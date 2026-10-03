from pathlib import Path
import pandas as pd
import numpy as np

# Run this once in Colab where your full clean ML CSV exists.
# It creates lightweight files for the website without uploading 20M rows.

SOURCE = Path('/content/bloomdetect_clean_ml_data.csv')
OUT_DIR = Path('/content')

if not SOURCE.exists():
    raise FileNotFoundError(f'Full dataset not found: {SOURCE}')

USECOLS = [
    'date','latitude','longitude','chla','risk_label','risk_probability',
    'historical_baseline','chla_anomaly','chla_change'
]

print('Reading the full ML dataset...')
df = pd.read_csv(SOURCE, usecols=lambda c: c in USECOLS)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['chla'] = pd.to_numeric(df['chla'], errors='coerce')

def risk_num(s):
    if s.dtype == object:
        return s.astype(str).str.lower().str.contains('risk|flag|potential|bloom|yes|true').astype(int)
    return (pd.to_numeric(s, errors='coerce').fillna(0) > 0).astype(int)

df['risk'] = risk_num(df['risk_label']) if 'risk_label' in df else 0

# ------------------------------------------------------------
# 1. Map history: 2-degree cells by date.
# This is intentionally much smaller than the 20M-row source.
# ------------------------------------------------------------
df['lat_bin'] = np.floor(df['latitude'] / 2) * 2 + 1
df['lon_bin'] = np.floor(df['longitude'] / 2) * 2 + 1

map_history = (
    df.groupby(['date','lat_bin','lon_bin'], as_index=False)
      .agg(
          chla=('chla','mean'),
          risk=('risk','sum'),
          cells=('chla','size'),
          max_chla=('chla','max'),
      )
)
map_history.to_csv(OUT_DIR / 'bloomdetect_history.csv', index=False)

print('Created:', OUT_DIR / 'bloomdetect_history.csv')
print('Rows:', len(map_history))
print('Dates:', map_history['date'].nunique())

# ------------------------------------------------------------
# 2. Location history: keep the actual 0.25-degree grid values.
# This can still be large, so only keep columns needed by the app.
# If the resulting file is too large for GitHub, keep it in Drive and
# do not upload it to Streamlit Cloud.
# ------------------------------------------------------------
loc_cols = [
    'date','latitude','longitude','chla','risk',
    'risk_probability','historical_baseline','chla_anomaly','chla_change'
]
for c in loc_cols:
    if c not in df:
        df[c] = np.nan

location_history = df[loc_cols].dropna(subset=['date','latitude','longitude','chla']).copy()
location_history.to_csv(OUT_DIR / 'bloomdetect_location_history.csv', index=False)

print('Created:', OUT_DIR / 'bloomdetect_location_history.csv')
print('Rows:', len(location_history))
print('Dates:', location_history['date'].nunique())
print('\nUpload bloomdetect_history.csv beside app.py for the Risk Map timeline.')
print('Use bloomdetect_location_history.csv only if you also want full location history.')
