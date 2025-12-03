from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from io import BytesIO
import base64
from scipy.interpolate import interp1d
from siphon.catalog import TDSCatalog
import warnings

app = Flask(__name__)

# Load data on startup
import os
data_dir = os.path.dirname(os.path.abspath(__file__))

soundings = pd.read_csv(os.path.join(data_dir, 'all_soundings_2024.csv'))
soundings['date'] = pd.to_datetime(soundings['date'])

flights = pd.read_csv(os.path.join(data_dir, 'xcontest_data.csv'))
flights['date'] = pd.to_datetime(flights['date'])

# Global variable for current forecast
current_forecast = None

# Standard pressure levels for comparison
STANDARD_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400]

def fetch_current_forecast():
    """Fetch today's GFS forecast for Denver and convert to match historical format"""
    print("Fetching current forecast from NOAA...")
    
    latitude = 39.74
    longitude = -104.99
    
    try:
        # Connect to catalog
        catalog_url = 'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml'
        catalog = TDSCatalog(catalog_url)
        latest_ds = list(catalog.datasets.values())[0]
        ncss = latest_ds.subset()
        
        # Query
        query = ncss.query()
        query.lonlat_point(longitude, latitude).accept('csv')
        query.variables('Geopotential_height_isobaric',
                        'Temperature_isobaric',
                        'Relative_humidity_isobaric',
                        'u-component_of_wind_isobaric',
                        'v-component_of_wind_isobaric')
        
        data = ncss.get_data(query)
        df = pd.DataFrame(data)
        
        # Rename columns
        df = df.rename(columns={
            'alt': 'pressure_Pa',
            'Geopotential_height_isobaric': 'height_m',
            'Temperature_isobaric': 'temperature_K',
            'Relative_humidity_isobaric': 'relative_humidity_%',
            'ucomponent_of_wind_isobaric': 'u_wind_ms',
            'vcomponent_of_wind_isobaric': 'v_wind_ms'
        })
        
        # Calculate dewpoint with warning suppression
        pressure_pa = df['pressure_Pa'].values * units.pascal
        temperature_k = df['temperature_K'].values * units.kelvin
        relative_humidity = df['relative_humidity_%'].values * units.percent
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dewpoint_k = mpcalc.dewpoint_from_relative_humidity(temperature_k, relative_humidity)
        
        df['dewpoint_K'] = dewpoint_k.to('kelvin').magnitude
        
        # CONVERT TO MATCH HISTORICAL FORMAT
        df_converted = pd.DataFrame({
            'date': pd.Timestamp.now(),
            'pressure': df['pressure_Pa'] / 100,  # Pa to hPa
            'height': df['height_m'],
            'temperature': df['temperature_K'] - 273.15,  # K to °C
            'dewpoint': df['dewpoint_K'] - 273.15,  # K to °C
            'u_wind': df['u_wind_ms'],
            'v_wind': df['v_wind_ms']
        })
        
        df_converted = df_converted.sort_values('pressure', ascending=False).reset_index(drop=True)
        
        # Only keep pressure levels we care about
        df_converted = df_converted[
            (df_converted['pressure'] <= 1000) & 
            (df_converted['pressure'] >= 400)
        ].reset_index(drop=True)
        
        print(f"✅ Forecast fetched successfully ({len(df_converted)} levels)")
        return df_converted
        
    except Exception as e:
        print(f"❌ Error fetching forecast: {e}")
        return None

def interpolate_to_standard_levels(data, levels=None):
    """Interpolate sounding data to standard pressure levels"""
    if levels is None:
        levels = STANDARD_LEVELS
    result = {'pressure': levels}
    
    for var in ['temperature', 'dewpoint', 'u_wind', 'v_wind']:
        valid = ~np.isnan(data[var].values)
        if valid.sum() < 2:
            return None
            
        p_valid = data['pressure'].values[valid]
        var_valid = data[var].values[valid]
        
        f = interp1d(p_valid, var_valid, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        result[var] = f(levels)
    
    return pd.DataFrame(result)

def calculate_similarity(date1, date2, is_forecast1=False, is_forecast2=False, 
                         weights=None, max_pressure=400):
    """Compare weather between two days, return distance score"""
    if is_forecast1:
        day1 = current_forecast.copy()
    else:
        day1 = soundings[soundings['date'].dt.date == pd.Timestamp(date1).date()].copy()
    
    if is_forecast2:
        day2 = current_forecast.copy()
    else:
        day2 = soundings[soundings['date'].dt.date == pd.Timestamp(date2).date()].copy()
    
    if day1 is None or day2 is None or len(day1) == 0 or len(day2) == 0:
        return float('inf')
    
    # Filter by altitude (max_pressure - higher hPa = lower altitude)
    levels_to_use = [p for p in STANDARD_LEVELS if p >= max_pressure]
    
    day1_std = interpolate_to_standard_levels(day1, levels_to_use)
    day2_std = interpolate_to_standard_levels(day2, levels_to_use)
    
    if day1_std is None or day2_std is None:
        return float('inf')
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'temperature': 1.5,
            'dewpoint': 1.0,
            'u_wind': 0.8,
            'v_wind': 0.8
        }
    
    total_distance = 0
    for var, weight in weights.items():
        if weight > 0:
            diff = day1_std[var].values - day2_std[var].values
            distance = np.sqrt(np.mean(diff**2))
            total_distance += distance * weight
    
    return total_distance

def find_similar_days(target_date, n_matches=10, is_forecast=False, 
                      weights=None, max_pressure=400):
    """Find historical days similar to target date"""
    if not is_forecast:
        target_date = pd.Timestamp(target_date)
    
    # Get all unique dates
    all_dates = soundings['date'].dt.date.unique()
    
    # Exclude target date if it's not a forecast
    if not is_forecast:
        all_dates = [d for d in all_dates if d != target_date.date()]
    
    # Calculate similarity for each
    similarities = []
    for date in all_dates:
        score = calculate_similarity(target_date if not is_forecast else None, date, 
                                     is_forecast1=is_forecast, is_forecast2=False,
                                     weights=weights, max_pressure=max_pressure)
        if not np.isinf(score):
            similarities.append((date, score))
    
    # Sort and return top matches
    similarities.sort(key=lambda x: x[1])
    return similarities[:n_matches]

def add_altitude_annotations(skew, day_data):
    """Add altitude annotations at key pressure levels"""
    altitude_markers = [(850, '5,000 ft'), (700, '10,000 ft'), (550, '15,000 ft')]
    for p_level, label in altitude_markers:
        skew.ax.text(1.02, p_level, label, fontsize=8, va='center', ha='left',
                    color='darkgray', transform=skew.ax.get_yaxis_transform(),
                    bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.7, edgecolor='none'))

def create_comparison_plot(target_date, match_date, is_forecast=False):
    """Create side-by-side Skew-T comparison plot"""
    # Get data
    if is_forecast:
        target_data = current_forecast.copy()
        target_label = f"Today's Forecast"
    else:
        target_data = soundings[soundings['date'].dt.date == pd.Timestamp(target_date).date()].copy()
        target_label = f'Target: {target_date}'
    
    match_data = soundings[soundings['date'].dt.date == pd.Timestamp(match_date).date()].copy()
    
    if target_data is None or len(target_data) == 0 or len(match_data) == 0:
        return None
    
    # Calculate similarity
    similarity = calculate_similarity(target_date if not is_forecast else None, match_date,
                                     is_forecast1=is_forecast, is_forecast2=False)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 8))
    
    # LEFT: Target date
    skew1 = SkewT(fig=fig, subplot=(1, 2, 1))
    
    p1 = target_data['pressure'].values * units.hPa
    T1 = target_data['temperature'].values * units.degC
    Td1 = target_data['dewpoint'].values * units.degC
    u1 = target_data['u_wind'].values * units('m/s')
    v1 = target_data['v_wind'].values * units('m/s')
    
    skew1.plot(p1, T1, 'r', linewidth=2, label='Temperature')
    skew1.plot(p1, Td1, 'g', linewidth=2, label='Dewpoint')
    skew1.plot_barbs(p1[::3], u1[::3], v1[::3])
    
    skew1.ax.set_ylim(1000, 400)
    skew1.ax.set_xlim(-30, 40)
    add_altitude_annotations(skew1, target_data)
    skew1.ax.legend(loc='upper left')
    skew1.ax.set_title(target_label, fontsize=12, fontweight='bold')
    
    # RIGHT: Match date
    skew2 = SkewT(fig=fig, subplot=(1, 2, 2))
    
    p2 = match_data['pressure'].values * units.hPa
    T2 = match_data['temperature'].values * units.degC
    Td2 = match_data['dewpoint'].values * units.degC
    u2 = match_data['u_wind'].values * units('m/s')
    v2 = match_data['v_wind'].values * units('m/s')
    
    skew2.plot(p2, T2, 'r', linewidth=2, label='Temperature')
    skew2.plot(p2, Td2, 'g', linewidth=2, label='Dewpoint')
    skew2.plot_barbs(p2[::3], u2[::3], v2[::3])
    
    skew2.ax.set_ylim(1000, 400)
    # No lables on match graph so it fits on screen better. 
    skew2.ax.set_yticklabels([])
    skew2.ax.set_ylabel('')
    skew2.ax.set_xlim(-30, 40)

    add_altitude_annotations(skew2, match_data)
    skew2.ax.legend(loc='upper left')
    skew2.ax.set_title(f'Match: {match_date} (Score: {similarity:.1f})', 
                      fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dates')
def get_dates():
    """Return list of available dates"""
    dates = sorted(soundings['date'].dt.date.unique())
    return jsonify([str(d) for d in dates])

@app.route('/api/matches/<date>')
def get_matches(date):
    """Get similar days for a given date"""
    from flask import request
    
    # Parse weight parameters
    weights = {
        'temperature': float(request.args.get('temp_weight', 1.5)),
        'dewpoint': float(request.args.get('dew_weight', 1.0)),
        'u_wind': float(request.args.get('u_weight', 0.8)),
        'v_wind': float(request.args.get('v_weight', 0.8))
    }
    max_pressure = int(request.args.get('max_pressure', 400))
    
    matches = find_similar_days(date, n_matches=10, weights=weights, max_pressure=max_pressure)
    
    result = []
    for match_date, score in matches:
        match_date_str = str(match_date)
        
        # Get flight data for this date
        flight_data = flights[flights['date'].dt.date == match_date]
        
        if len(flight_data) > 0 and flight_data['flights'].values[0] >= 0:
            num_pilots = int(flight_data['flights'].values[0])
            max_distance = float(flight_data['max_distance'].values[0])
        else:
            num_pilots = 0
            max_distance = 0
        
        # Get sounding data for sparkline (simplified for display)
        match_sounding = soundings[soundings['date'].dt.date == match_date]
        sounding_data = None
        if len(match_sounding) > 0:
            # Sample every few levels for sparkline
            sampled = match_sounding.iloc[::3]
            sounding_data = {
                'pressure': sampled['pressure'].tolist(),
                'temperature': sampled['temperature'].tolist(),
                'dewpoint': sampled['dewpoint'].tolist()
            }
        
        result.append({
            'date': match_date_str,
            'score': float(score),
            'num_pilots': num_pilots,
            'max_distance': max_distance,
            'sounding': sounding_data
        })
    
    return jsonify(result)

@app.route('/api/comparison/<target_date>/<match_date>')
def get_comparison(target_date, match_date):
    """Generate comparison plot"""
    is_forecast = target_date == 'forecast'
    img = create_comparison_plot(target_date, match_date, is_forecast=is_forecast)
    
    if img is None:
        return jsonify({'error': 'Could not generate plot'}), 400
    
    return jsonify({'image': img})

@app.route('/api/fetch_forecast')
def api_fetch_forecast():
    """Fetch current forecast from NOAA"""
    global current_forecast
    current_forecast = fetch_current_forecast()
    
    if current_forecast is None:
        return jsonify({'success': False, 'error': 'Could not fetch forecast'}), 500
    
    return jsonify({'success': True, 'message': f'Fetched {len(current_forecast)} levels'})

@app.route('/api/forecast_matches')
def get_forecast_matches():
    """Get similar historical days for current forecast"""
    global current_forecast
    from flask import request
    
    if current_forecast is None:
        current_forecast = fetch_current_forecast()
        if current_forecast is None:
            return jsonify({'error': 'Could not fetch forecast'}), 500
    
    # Parse weight parameters
    weights = {
        'temperature': float(request.args.get('temp_weight', 1.5)),
        'dewpoint': float(request.args.get('dew_weight', 1.0)),
        'u_wind': float(request.args.get('u_weight', 0.8)),
        'v_wind': float(request.args.get('v_weight', 0.8))
    }
    max_pressure = int(request.args.get('max_pressure', 400))
    
    matches = find_similar_days(None, n_matches=10, is_forecast=True, 
                                weights=weights, max_pressure=max_pressure)
    
    result = []
    for match_date, score in matches:
        match_date_str = str(match_date)
        
        # Get flight data for this date
        flight_data = flights[flights['date'].dt.date == match_date]
        
        if len(flight_data) > 0 and flight_data['flights'].values[0] >= 0:
            num_pilots = int(flight_data['flights'].values[0])
            max_distance = float(flight_data['max_distance'].values[0])
        else:
            num_pilots = 0
            max_distance = 0
        
        # Get sounding data for sparkline
        match_sounding = soundings[soundings['date'].dt.date == match_date]
        sounding_data = None
        if len(match_sounding) > 0:
            sampled = match_sounding.iloc[::3]
            sounding_data = {
                'pressure': sampled['pressure'].tolist(),
                'temperature': sampled['temperature'].tolist(),
                'dewpoint': sampled['dewpoint'].tolist()
            }
        
        result.append({
            'date': match_date_str,
            'score': float(score),
            'num_pilots': num_pilots,
            'max_distance': max_distance,
            'sounding': sounding_data
        })
    
    return jsonify(result)

@app.route('/api/scatter_data')
def get_scatter_data():
    """Get all flight data for scatter plot"""
    data = []
    for _, row in flights.iterrows():
        if row['flights'] >= 0 and row['max_distance'] >= 0:
            data.append({
                'date': str(row['date'].date()),
                'num_pilots': int(row['flights']),
                'max_distance': float(row['max_distance'])
            })
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
