# SkyMatch - Quick Setup

## Files Needed

Copy these 4 files to your existing skymatch directory (where your CSV files are):

1. **app.py** - Main Flask application
2. **index.html** - Web interface (put this in a `templates/` folder)
3. **requirements.txt** - Python dependencies  
4. **start.sh** - Quick start script (optional)

## Directory Structure

```
skymatch/
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
├── start.sh
├── all_soundings_2024.csv  ← You already have this
└── xcontest_data.csv        ← You already have this
```

## Setup Steps

### Option 1: Use start.sh (Easiest)

```bash
cd ~/Documents/dataVis/skymatch
chmod +x start.sh
./start.sh
```

### Option 2: Manual Setup

```bash
cd ~/Documents/dataVis/skymatch

# Create templates directory
mkdir -p templates

# Move index.html into templates/
mv index.html templates/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open: **http://localhost:5000**

## Deploying to VPS

See `DEPLOY_CADDY.md` for full deployment instructions.

Quick version:
```bash
# On VPS
cd ~/skymatch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service
sudo nano /etc/systemd/system/skymatch.service

# Add to Caddyfile
sudo nano /etc/caddy/Caddyfile
# Add: skymatch.yourdomain.com { reverse_proxy localhost:5000 }

# Start services
sudo systemctl start skymatch
sudo systemctl reload caddy
```

## Troubleshooting

**Module not found errors:**
- Make sure venv is activated: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**File not found errors:**
- Check CSV files are in the same directory as app.py
- Check index.html is in templates/ folder

**Port 5000 in use:**
- Edit app.py, change last line to: `app.run(debug=True, port=5001)`
