# skymatch

Answers a question every paraglider pilot asks: **"the forecast looks like *that*... but will it actually fly?"**

Skymatch pulls today's GFS forecast for the Denver area from NOAA, finds the most
similar days in a year of historical atmospheric soundings, and shows how those
days actually flew, using logged flights from XContest.

## How it works

1. **Fetch** the latest GFS forecast for Denver (39.74, -104.99) from the
   NOAA THREDDS server via Siphon.
2. **Normalize** every sounding (forecast and historical) by interpolating
   temperature, dewpoint, and u/v wind onto standard pressure levels
   (1000 to 400 hPa).
3. **Compare** the forecast against each historical day using a weighted
   root-mean-square distance across those variables. Temperature is weighted
   highest, then dewpoint, then winds.
4. **Rank** the closest matches and join them against XContest flight logs:
   how many pilots flew that day, and how far.

The idea: if today's atmosphere closely resembles a day when 40 pilots flew
100 km, that says more than any single forecast index.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Requires `all_soundings_2024.csv` and `xcontest_data.csv` in the project root
(included).

## Stack

Flask · pandas · MetPy (Skew-T plots) · SciPy (interpolation) · Siphon (NOAA data access)

## Limitations and ideas

- One year of historical soundings; more history means better matches.
- Similarity weights are hardcoded; they could be fit against flight outcomes
  instead of guessed.
- Flight counts conflate weather with weekends. A day-of-week correction is
  on the list.
