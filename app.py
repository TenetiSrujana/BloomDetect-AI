"""BloomDetect data and plotting helpers; importing this module loads no data."""
from pathlib import Path
import gzip
import io
import logging
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -40., 30., 20., 120.
MAX_HISTORY_ROWS = 2_000_000
MAX_HISTORY_BYTES = 180_000_000
LOGGER = logging.getLogger(__name__)
HISTORY_COLUMNS = ['date', 'latitude', 'longitude', 'chla', 'risk_flag',
                   'lat_bin', 'lon_bin', 'historical_baseline', 'chla_anomaly', 'chla_change',
                   'risk', 'stored_risk_flag']
PERSISTENCE_COLUMNS = ['latitude', 'longitude', 'observed_count', 'flagged_count',
                       'recurrence_pct', 'coverage_pct', 'max_gap_days', 'last_observed',
                       'current_run', 'historical_baseline', 'last_anomaly', 'last_change']


def normalize_lon(lon):
    """Normalize one finite longitude to [-180, 180); invalid input yields NaN."""
    try:
        value = float(lon)
        return (value + 180.) % 360. - 180. if np.isfinite(value) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _clean(frame):
    """Validate observations without interpreting zero as land or missing data."""
    frame = frame.copy()
    required = {'date', 'latitude', 'longitude', 'chla'}
    if not required.issubset(frame.columns):
        raise ValueError('Observation data is missing required fields.')
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    for col in ['latitude', 'longitude', 'chla']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    # Coordinate validation is not an ocean/land mask. Genuine zero chla is retained.
    valid = (frame['date'].notna() & np.isfinite(frame[['latitude', 'longitude', 'chla']]).all(axis=1)
             & frame['latitude'].between(LAT_MIN, LAT_MAX)
             & frame['longitude'].between(LON_MIN, LON_MAX) & frame['chla'].ge(0))
    frame = frame.loc[valid].copy()
    for col in ['historical_baseline', 'chla_anomaly', 'chla_change', 'previous_chla',
                'recent_mean', 'recent_max', 'history_count', 'risk_probability']:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frame


@st.cache_data(show_spinner=False)
def load_latest():
    """Load predictions, or the labels file only if predictions are absent.

    Source prediction/probability/label fields remain intact. risk_flag is a
    binary display alias of supplied results, never inferred from chlorophyll.
    """
    path = BASE_DIR / 'latest_bloom_risk_predictions.csv'
    if not path.is_file():
        path = BASE_DIR / 'latest_bloom_risk_labels.csv'
    original = pd.read_csv(path)
    frame = _clean(original)
    source = next((c for c in ['prediction', 'risk_flag', 'risk'] if c in frame), None)
    if source is not None:
        flags = pd.to_numeric(frame[source], errors='coerce')
    elif 'risk_label' in frame:
        source = 'risk_label'
        labels = frame[source].astype('string').str.strip().str.casefold()
        flags = labels.map({'normal': 0, 'potential bloom risk': 1})
    else:
        raise ValueError('Latest results have no supplied prediction or recognized label field.')
    if not flags.isin([0, 1]).all():
        raise ValueError('Latest predictions contain invalid risk flags.')
    frame['risk_flag'] = flags.astype('int8')
    frame = frame.reset_index(drop=True)
    frame.attrs.update(source_file=path.name, flag_source=source,
                       excluded_observation_rows=len(original) - len(frame))
    return frame


def _read_bounded_history(path):
    """Bound on-disk bytes, decompressed bytes and parsed rows for every format."""
    if path.stat().st_size > MAX_HISTORY_BYTES:
        raise ValueError('History exceeds the file byte limit.')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as source:
        payload = source.read(MAX_HISTORY_BYTES + 1)
    if len(payload) > MAX_HISTORY_BYTES:
        raise ValueError('History exceeds the uncompressed byte limit.')
    chunks, count = [], 0
    for chunk in pd.read_csv(io.BytesIO(payload), chunksize=100_000):
        count += len(chunk)
        if count > MAX_HISTORY_ROWS:
            raise ValueError('History exceeds the row limit.')
        chunks.append(chunk)
    if not chunks:
        raise ValueError('History is empty.')
    return pd.concat(chunks, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_history():
    """Load bounded compact history and calculate a prior-observation screening proxy.

    Baseline is the expanding per-bin mean of strictly earlier observations;
    change is from the preceding available observation. Both thresholds must
    be met (>=). This is a coarse-grid screening proxy, not model inference.
    """
    names = ['bloomdetect_history_web.csv.gz', 'bloomdetect_history.csv.gz',
             'bloomdetect_history.csv']
    for name in names:
        path = BASE_DIR / name
        if not path.is_file():
            continue
        try:
            return _prepare_history(_read_bounded_history(path), name)
        except Exception:
            LOGGER.warning('BloomDetect history could not be loaded from %s', name, exc_info=True)
    return (pd.DataFrame(columns=HISTORY_COLUMNS),
            'Historical context is unavailable. The latest stored model results remain separate and available.')


def _prepare_history(frame, source_name):
    """Validate compact 1° centers and recompute a strictly prior proxy."""
    original_count = len(frame)
    required = {'date', 'lat_bin', 'lon_bin', 'chla', 'risk'}
    if not required.issubset(frame.columns):
        raise ValueError('Compact history is missing required columns.')
    frame = frame.copy()
    frame['latitude'] = frame['lat_bin']
    frame['longitude'] = frame['lon_bin']
    frame = _clean(frame)
    frame['lat_bin'], frame['lon_bin'] = frame['latitude'], frame['longitude']
    centers = (np.isclose(frame['lat_bin'] % 1, .5) & np.isclose(frame['lon_bin'] % 1, .5))
    frame = frame.loc[centers].sort_values(['lat_bin', 'lon_bin', 'date']).copy()
    if frame.empty or frame.duplicated(['lat_bin', 'lon_bin', 'date']).any():
        raise ValueError('Compact history has no valid rows or duplicate bin dates.')
    groups = frame.groupby(['lat_bin', 'lon_bin'], sort=False)['chla']
    frame['historical_baseline'] = groups.transform(lambda values: values.shift(1).expanding().mean())
    frame['chla_anomaly'] = frame['chla'] - frame['historical_baseline']
    frame['chla_change'] = groups.diff()
    frame['stored_risk_flag'] = pd.to_numeric(frame['risk'], errors='coerce')
    frame['risk_flag'] = (frame['chla_anomaly'].ge(ANOMALY_THRESHOLD)
                          & frame['chla_change'].ge(CHANGE_THRESHOLD)).astype('int8')
    comparable = frame['stored_risk_flag'].isin([0, 1])
    mismatches = int((frame.loc[comparable, 'stored_risk_flag'] != frame.loc[comparable, 'risk_flag']).sum())
    stored_count = int(frame['stored_risk_flag'].eq(1).sum())
    proxy_count = int(frame['risk_flag'].sum())
    invalid_count = int((~comparable).sum())
    first_count = int(frame['historical_baseline'].isna().sum())
    message = ('Historical screening is recomputed from strictly earlier per-bin means and the preceding '
               'available observation, using both original thresholds (>=); it is a 1° proxy, not ML predictions. '
               f'Recomputed flags: {proxy_count:,}; stored flags: {stored_count:,}. '
               f'{mismatches:,} of {int(comparable.sum()):,} comparable stored flags differ. '
               f'{invalid_count:,} stored flags are unavailable or invalid; '
               f'{first_count:,} first observations lack prior context and are not proxy-flagged. '
               f'{original_count - len(frame):,} invalid observation rows were excluded. '
               'Zero chlorophyll values are retained; no verified land mask is available.')
    frame = frame.reset_index(drop=True)
    frame.attrs.update(source_file=source_name, screening_method='strictly-prior 1-degree proxy; both thresholds >=',
                       proxy_flagged_count=proxy_count, stored_flagged_count=stored_count,
                       comparable_stored_count=int(comparable.sum()), mismatch_count=mismatches,
                       invalid_stored_count=invalid_count, no_prior_context_count=first_count,
                       excluded_observation_rows=original_count - len(frame), screening_disclosure=message)
    return frame, message


def nearest_cell(latest, lat, lon):
    """Nearest valid study-window cell and great-circle distance in kilometres."""
    try:
        lat, lon = float(lat), normalize_lon(lon)
    except (TypeError, ValueError, OverflowError):
        return None, np.nan
    if not {'latitude', 'longitude'}.issubset(latest.columns):
        return None, np.nan
    if not (np.isfinite(lat) and np.isfinite(lon) and LAT_MIN <= lat <= LAT_MAX
            and LON_MIN <= lon <= LON_MAX) or latest.empty:
        return None, np.nan
    lats = pd.to_numeric(latest['latitude'], errors='coerce').to_numpy(dtype=float)
    lons = pd.to_numeric(latest['longitude'], errors='coerce').to_numpy(dtype=float)
    valid = (np.isfinite(lats) & np.isfinite(lons) & (lats >= LAT_MIN) & (lats <= LAT_MAX)
             & (lons >= LON_MIN) & (lons <= LON_MAX))
    positions = np.flatnonzero(valid)
    if not len(positions):
        return None, np.nan
    p1, p2 = np.radians(lat), np.radians(lats[valid])
    delta = np.radians(lons[valid] - lon)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(delta / 2) ** 2
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    index = int(np.argmin(distances))
    return latest.iloc[positions[index]], float(distances[index])


def location_history(history, lat, lon):
    """Return the containing 1° bin, never an arbitrary nearest history cell."""
    try:
        lat, lon = float(lat), normalize_lon(lon)
    except (TypeError, ValueError, OverflowError):
        return history.iloc[:0].copy()
    if (history.empty or not {'date', 'lat_bin', 'lon_bin'}.issubset(history.columns)
            or not (np.isfinite(lat) and np.isfinite(lon) and LAT_MIN <= lat <= LAT_MAX
                    and LON_MIN <= lon <= LON_MAX)):
        return history.iloc[:0].copy()
    return history.loc[(history['lat_bin'] == np.floor(lat) + .5)
                       & (history['lon_bin'] == np.floor(lon) + .5)].sort_values('date').copy()


@st.cache_data(show_spinner=False, max_entries=6)
def persistence_table(history, as_of, window_days=60):
    """Aggregate proxy recurrence over actual dataset dates in the inclusive window.

    current_run counts trailing flagged available dataset observations; a date
    missing for a cell breaks the run. max_gap_days is the largest elapsed gap
    between that cell's observations (zero if fewer than two observations).
    """
    if history.empty or not {'date', 'lat_bin', 'lon_bin', 'risk_flag'}.issubset(history.columns):
        return pd.DataFrame(columns=PERSISTENCE_COLUMNS)
    try:
        window_days = int(window_days)
    except (ValueError, TypeError, OverflowError):
        return pd.DataFrame(columns=PERSISTENCE_COLUMNS)
    end = pd.to_datetime(as_of, utc=True, errors='coerce')
    if pd.isna(end) or int(window_days) < 1:
        return pd.DataFrame(columns=PERSISTENCE_COLUMNS)
    end = end.tz_convert(None).normalize()
    start = end - pd.Timedelta(days=int(window_days) - 1)
    data = history.loc[history['date'].between(start, end)].sort_values('date')
    dates = pd.DatetimeIndex(data['date'].drop_duplicates().sort_values())
    if not len(dates):
        return pd.DataFrame(columns=PERSISTENCE_COLUMNS)
    rows = []
    for (lat, lon), cell in data.groupby(['lat_bin', 'lon_bin'], sort=False):
        observed = cell.drop_duplicates('date', keep='last')
        last = observed.iloc[-1]
        date_positions = dates.get_indexer(observed['date'])
        flags = observed['risk_flag'].eq(1).to_numpy()
        run, expected = 0, len(dates) - 1
        for position, flag in zip(date_positions[::-1], flags[::-1]):
            if position != expected or not flag:
                break
            run += 1
            expected -= 1
        count, flagged = len(observed), int(flags.sum())
        gaps = observed['date'].diff().dt.total_seconds().div(86400)
        rows.append({'latitude': lat, 'longitude': lon, 'observed_count': count,
                     'flagged_count': flagged, 'recurrence_pct': 100. * flagged / count,
                     'coverage_pct': 100. * count / len(dates),
                     'max_gap_days': float(gaps.max()) if count > 1 else 0.,
                     'last_observed': last['date'], 'current_run': run,
                     'historical_baseline': last.get('historical_baseline', np.nan),
                     'last_anomaly': last.get('chla_anomaly', np.nan),
                     'last_change': last.get('chla_change', np.nan)})
    result = pd.DataFrame(rows, columns=PERSISTENCE_COLUMNS)
    result.attrs['available_dataset_dates'] = len(dates)
    result.attrs['window_start'], result.attrs['window_end'] = start, end
    return result


def rank_investigations(latest, history, as_of):
    """Latest model flags only; descending lexicographic recurrence/change/coverage.

    Regional proxy metrics come from the containing 1° bin. Missing metrics
    sort last, equal keys preserve source order. This is not a severity score.
    """
    if latest.empty or not {'risk_flag', 'latitude', 'longitude'}.issubset(latest.columns):
        return pd.DataFrame(columns=list(dict.fromkeys(list(latest.columns) + ['reason'])))
    flagged = latest.loc[latest['risk_flag'].eq(1)].copy()
    flagged['lat_bin'] = np.floor(flagged['latitude']) + .5
    flagged['lon_bin'] = np.floor(flagged['longitude']) + .5
    metrics = persistence_table(history, as_of).rename(
        columns={'latitude': 'lat_bin', 'longitude': 'lon_bin',
                 'historical_baseline': 'history_baseline'})
    flagged = flagged.merge(metrics, on=['lat_bin', 'lon_bin'], how='left', validate='many_to_one')
    if 'chla_change' not in flagged:
        flagged['chla_change'] = np.nan
    flagged['chla_change'] = pd.to_numeric(flagged['chla_change'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    flagged = flagged.sort_values(['recurrence_pct', 'chla_change', 'coverage_pct'],
                                  ascending=False, na_position='last', kind='stable')
    def reason(row):
        if pd.isna(row['observed_count']):
            return ('Latest model-flagged cell; no containing-bin observations in the historical window. '
                    'Order: regional recurrence, latest change, available-date coverage (descending; missing last).')
        change = f"{row['chla_change']:+.4f}" if pd.notna(row['chla_change']) else 'unavailable'
        return (f"Latest model-flagged cell; 1° historical proxy recurrence {row['recurrence_pct']:.1f}% "
                f"({int(row['flagged_count'])}/{int(row['observed_count'])} observations); "
                f"latest change {change}; dataset-date coverage {row['coverage_pct']:.1f}%. "
                'Order: recurrence, latest change, coverage (descending; missing last); not a severity score.')
    flagged['reason'] = flagged.apply(reason, axis=1) if not flagged.empty else pd.Series(dtype=str)
    return flagged.reset_index(drop=True)


def _zone_label(lower, upper, positive, negative):
    def coordinate(value):
        return f'{abs(value):g}°{positive if value >= 0 else negative}'
    return f'{coordinate(lower)}–{coordinate(upper)}'


def hotspot_table(field):
    """All observed 2° zones, including unflagged zones; floor-based bounds."""
    columns = ['zone', 'latitude', 'longitude', 'lat_lower', 'lat_upper', 'lon_lower',
               'lon_upper', 'observed_count', 'flagged_count', 'flagged_pct', 'mean_chla', 'max_chla']
    if field.empty:
        return pd.DataFrame(columns=columns)
    data = field.copy()
    for col in ['latitude', 'longitude', 'chla', 'risk_flag']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.loc[np.isfinite(data[['latitude', 'longitude', 'chla']]).all(axis=1)
                    & data['latitude'].between(LAT_MIN, LAT_MAX)
                    & data['longitude'].between(LON_MIN, LON_MAX)].copy()
    data['lat_lower'] = np.floor(data['latitude'] / 2) * 2
    data['lon_lower'] = np.floor(data['longitude'] / 2) * 2
    data['flag'] = data['risk_flag'].eq(1).astype(int)
    result = data.groupby(['lat_lower', 'lon_lower'], as_index=False).agg(
        observed_count=('chla', 'size'), flagged_count=('flag', 'sum'),
        mean_chla=('chla', 'mean'), max_chla=('chla', 'max'))
    result['lat_upper'], result['lon_upper'] = result['lat_lower'] + 2, result['lon_lower'] + 2
    result['latitude'], result['longitude'] = result['lat_lower'] + 1, result['lon_lower'] + 1
    result['flagged_pct'] = 100 * result['flagged_count'] / result['observed_count']
    result['zone'] = [_zone_label(a, a + 2, 'N', 'S') + ', ' + _zone_label(b, b + 2, 'E', 'W')
                      for a, b in zip(result['lat_lower'], result['lon_lower'])]
    return result[columns].sort_values(['flagged_count', 'flagged_pct'], ascending=False).reset_index(drop=True)


def chart_style(fig, title='', height=380):
    """Apply readable dark text on a light native-chart canvas."""
    fig.update_layout(title={'text': html.escape(title), 'font': {'size': 18, 'color': '#142d3c'}},
                      height=height, font={'family': 'Arial, sans-serif', 'size': 13, 'color': '#142d3c'},
                      paper_bgcolor='#ffffff', plot_bgcolor='#f7fafc',
                      margin={'l': 40, 'r': 30, 't': 60, 'b': 40},
                      legend={'font': {'color': '#142d3c'}},
                      hoverlabel={'bgcolor': '#ffffff', 'font': {'color': '#142d3c'}})
    fig.update_xaxes(title_font={'color': '#142d3c'}, tickfont={'color': '#142d3c'}, gridcolor='#dce6ec')
    fig.update_yaxes(title_font={'color': '#142d3c'}, tickfont={'color': '#142d3c'}, gridcolor='#dce6ec')
    return fig


def make_map(field, mode='Screening'):
    """Coordinate-validated geographic display; sampling never removes flags."""
    modes = {'Screening': 'risk_flag', 'Chlorophyll-a': 'chla',
             'Historical anomaly': 'chla_anomaly', 'Recent change': 'chla_change'}
    if mode not in modes:
        raise ValueError(f'Unknown map mode: {mode}')
    data = field.copy()
    for col in ['latitude', 'longitude', 'risk_flag', 'chla']:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors='coerce')
    valid = (np.isfinite(data[['latitude', 'longitude']]).all(axis=1)
             & data['latitude'].between(LAT_MIN, LAT_MAX) & data['longitude'].between(LON_MIN, LON_MAX))
    data = data.loc[valid].copy()
    flags = data.loc[data['risk_flag'].eq(1)].copy()
    background = data.loc[~data['risk_flag'].eq(1)]
    step = max(1, int(np.ceil(len(background) / 4500)))
    sample = background.iloc[::step].copy()
    fig = go.Figure()
    if mode == 'Screening':
        fig.add_trace(go.Scattergeo(lat=sample['latitude'], lon=sample['longitude'], mode='markers',
                               name=f'Background rendering sample ({len(sample):,}/{len(background):,})',
                               marker={'size': 3, 'color': '#4595ce', 'opacity': .38},
                               hovertemplate='%{lat:.3f}°, %{lon:.3f}°<extra>Background sample</extra>'))
    if mode == 'Screening':
        zones = hotspot_table(data)
        zones = zones.loc[zones['flagged_count'].gt(0)]
        sizes = np.clip(pd.to_numeric(zones['flagged_count'], errors='coerce').fillna(0).to_numpy(dtype=float) * 1.2 + 8, 9, 28)
        fig.add_trace(go.Scattergeo(lat=zones['latitude'], lon=zones['longitude'], mode='markers',
                                   name='Flag concentration (2° zones)',
                                   marker={'size': sizes, 'color': '#22a879', 'opacity': .3},
                                   customdata=zones[['zone', 'flagged_count']].to_numpy(),
                                   hovertemplate='%{customdata[0]}<br>Flagged records: %{customdata[1]}<extra>Concentration</extra>'))
        # Explicit numeric arrays avoid pandas attribute collisions.
        values = flags['chla'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        denominator = max(float(values.quantile(.95)) if len(values) else 0., .001)
        flags['marker_size'] = (7 + 8 * np.sqrt((values / denominator).clip(0, 1))).clip(7, 15)
        fig.add_trace(go.Scattergeo(lat=flags['latitude'], lon=flags['longitude'], mode='markers',
                                   name=f'All flagged cells ({len(flags):,})',
                                   marker={'size': flags['marker_size'].to_numpy(dtype=float),
                                            'color': '#ef4f5e', 'opacity': .9,
                                            'line': {'color': '#ffffff', 'width': .5}},
                                   customdata=flags[['chla']].to_numpy(),
                                   hovertemplate='%{lat:.3f}°, %{lon:.3f}°<br>Chlorophyll-a %{customdata[0]:.4f}<extra>Flagged</extra>'))
    else:
        # All flags are retained alongside the sparse unflagged sample.
        shown = pd.concat([sample, flags], ignore_index=True)
        column = modes[mode]
        values = pd.to_numeric(shown[column], errors='coerce') if column in shown else pd.Series(np.nan, index=shown.index)
        finite = np.isfinite(values)
        shown, values = shown.loc[finite], values.loc[finite]
        marker = {'size': 5, 'color': values.to_numpy(dtype=float), 'colorscale': 'Viridis',
                  'showscale': True, 'colorbar': {'title': {'text': mode}, 'tickfont': {'color': '#142d3c'}}}
        if mode in ['Historical anomaly', 'Recent change']:
            extent = max(float(values.abs().max()) if len(values) else 0., .001)
            marker.update(colorscale='RdBu', reversescale=True, cmin=-extent, cmax=extent)
        fig.add_trace(go.Scattergeo(lat=shown['latitude'], lon=shown['longitude'], mode='markers',
                                   name=f'{mode}: background sample + all valid flags', marker=marker,
                                   customdata=np.column_stack([values, shown['risk_flag']]),
                                   hovertemplate='%{lat:.3f}°, %{lon:.3f}°<br>Value %{customdata[0]:.4f}<br>Flag %{customdata[1]}<extra></extra>'))
        # Flags with missing metric still remain explicitly visible.
        missing = flags.loc[~np.isfinite(pd.to_numeric(flags[column], errors='coerce'))] if column in flags else flags
        if not missing.empty:
            fig.add_trace(go.Scattergeo(lat=missing['latitude'], lon=missing['longitude'], mode='markers',
                                        name='Flagged; metric unavailable', marker={'size': 7, 'color': '#ef4f5e'}))
    fig.update_geos(projection_type='equirectangular', showland=True, landcolor='#eef0e8',
                    showocean=True, oceancolor='#f0f7fc', showcoastlines=True, coastlinecolor='#778996',
                    showcountries=True, countrycolor='#b2bdc5', bgcolor='#ffffff',
                    lataxis={'range': [LAT_MIN, LAT_MAX], 'showgrid': True, 'dtick': 10, 'gridcolor': '#c4d4df'},
                    lonaxis={'range': [LON_MIN, LON_MAX], 'showgrid': True, 'dtick': 10, 'gridcolor': '#c4d4df'})
    chart_style(fig, f'BloomDetect · {mode}', height=570)
    fig.add_annotation(x=0, y=-.10, xref='paper', yref='paper', showarrow=False, xanchor='left',
                       text='Background is a rendering sample; all flags retained. Coordinates validated; no verified land mask.',
                       font={'size': 11, 'color': '#334e60'})
    fig.update_layout(margin={'l': 20, 'r': 20, 't': 60, 'b': 75})
    return fig


# UI fragment: concatenate after backend.py; no backend imports required.
from datetime import datetime, timezone

st.set_page_config(page_title="BloomDetect AI", page_icon="🌊", layout="wide")

STYLES = """
<style>
:root { --bd-teal:#087f8c; --bd-ink:#153e50; }
.stApp { background:#f3f9fc; color:var(--bd-ink); color-scheme:light; }
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p { color:#153e50; }
.block-container { max-width:1380px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { color:#153e50; letter-spacing:-.025em; }
[data-testid="stVerticalBlockBorderWrapper"] { background:white; border-radius:18px; }
[data-testid="stMetric"] { background:#fff; border:1px solid #dbeaf0; border-radius:15px; padding:16px; }
[data-testid="stMetricLabel"] { color:#537080!important; }
[data-testid="stMetricValue"] { color:#153e50!important; font-size:1.65rem!important; }
.stNumberInput input { background:#fff!important; color:#153e50!important; }
.stButton>button,.stDownloadButton>button { border-radius:11px; border-color:#c9e2e8; }
.stButton>button[kind="primary"] { background:#087f8c; border-color:#087f8c; }
[data-testid="stImage"] img { border-radius:18px; width:100%; height:auto; object-fit:contain; }
.bd-hero { background:linear-gradient(110deg,#075766,#087f8c,#16a6a0); padding:20px 26px; border-radius:22px; margin:12px 0 22px; }
.bd-eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.75rem; color:#cff3f2; font-weight:700; }
.bd-hero h1 { margin:.25rem 0; font-size:2.3rem; color:white; }
.bd-hero p { margin:.3rem 0; max-width:850px; color:#e2f6f8; }
.bd-note { color:#537080; font-size:.9rem; }
header[data-testid="stHeader"] { background:rgba(243,249,252,.95); }
</style>
"""


def _ui_dates(frame):
    if frame is None or frame.empty or "date" not in frame:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _ui_numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ui_flagged(frame):
    return _ui_numeric(frame, "risk_flag").eq(1)


def _ui_date_label(value):
    return pd.Timestamp(value).strftime("%d %b %Y") if pd.notna(value) else "Unavailable"


def _ui_value(value, suffix="", digits=4):
    try:
        return f"{float(value):,.{digits}f}{suffix}" if np.isfinite(float(value)) else "Unavailable"
    except (ValueError, TypeError):
        return "Unavailable"


def _ui_csv(frame):
    return frame.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def _ui_history_cached():
    return load_history()


def _ui_history():
    try:
        history, message = _ui_history_cached()
        if history is None or history.empty:
            st.info("Historical context is unavailable. The latest model snapshot remains available.")
            return pd.DataFrame()
        st.caption("History is a 1° regional screening proxy using strictly earlier observations, not historical model predictions. Missing dates are not filled; no verified land mask is available.")
        if history.attrs.get('mismatch_count', 0):
            st.warning("Stored historical flags differ from the documented screening rule. Historical displays use the recomputed proxy; see Data for details.")
        return history
    except Exception:
        logging.exception("Unable to load optional historical context")
        st.info("Historical context could not be loaded. You can still explore the latest model snapshot.")
        return pd.DataFrame()


def _ui_latest_date(latest):
    dates = _ui_dates(latest)
    return dates.max() if not dates.empty else pd.NaT


def _ui_snapshot(latest):
    columns = st.columns(4)
    count = len(latest)
    flagged = int(_ui_flagged(latest).sum())
    columns[0].metric("Processed cells", f"{count:,}")
    columns[1].metric("Model-flagged cells", f"{flagged:,}")
    columns[2].metric("Risk share", f"{flagged/count:.2%}" if count else "Unavailable")
    columns[3].metric("Maximum chlorophyll-a", _ui_value(_ui_numeric(latest, "chla").max(), " mg/m³", 4))
    st.caption(f"Latest stored model result · {_ui_date_label(_ui_latest_date(latest))} UTC · Screening flags are investigation cues, not confirmed blooms.")


def _ui_plot(fig, title="", height=380):
    styled = chart_style(fig, title=title, height=height)
    if any(getattr(trace, 'type', '') == 'scattergeo' for trace in fig.data):
        styled.update_layout(margin=dict(l=20, r=20, t=25, b=80))
    st.plotly_chart(styled if styled is not None else fig, use_container_width=True)


def _ui_map(field, key):
    choices = ["Screening", "Chlorophyll-a", "Historical anomaly", "Recent change"]
    mode = st.radio("Map layer", choices, horizontal=True, key=key)
    column = {"Chlorophyll-a":"chla", "Historical anomaly":"chla_anomaly", "Recent change":"chla_change"}.get(mode, "risk_flag")
    if column not in field or _ui_numeric(field, column).notna().sum() == 0:
        st.info(f"{mode} is unavailable for this snapshot. Choose another layer.")
        return
    with st.container(border=True):
        _ui_plot(make_map(field, mode=mode), height=490)
    if mode == 'Screening':
        st.caption("Blue: processed-field rendering sample · Green: 2° flag concentration · Red: individual potential-risk screening record. Study region: Indian Ocean / Arabian Sea / Bay of Bengal, latitude −40° to 30°, longitude 20° to 120°.")
    st.caption("Coordinates in decimal degrees. Chlorophyll-a, historical anomaly and recent change: mg/m³. Missing values are not evidence of low risk.")


def _ui_home(latest):
    _ui_snapshot(latest)
    st.markdown("### From ocean colour to an investigation cue")
    a, b = st.columns([1.15, 1])
    with a:
        st.write("BloomDetect AI brings a stored chlorophyll-a model snapshot and regional historical screening context into one scientific workspace. Inspect a flag, compare its context, then decide where independent verification is needed.")
        st.markdown("**Risk Map** — spatial layers and available-date playback.\n\n**Location** — nearest processed cell and containing-bin history.\n\n**Insights** — changes, regional patterns and investigation context.\n\n**Data** — sources, coverage and downloadable records.")
        st.caption("Chlorophyll-a is a phytoplankton proxy. Satellite retrievals and screening alone cannot establish species, toxicity or a confirmed bloom.")
    with b:
        image = Path(BASE_DIR) / "bloomdetect_bloom_process.png"
        if image.exists():
            st.image(str(image), caption="Ocean-colour observation → chlorophyll-a → screening → investigation", use_container_width=True)
        else:
            illustration = '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340" viewBox="0 0 760 340">
<rect width="760" height="340" rx="24" fill="#e7f4fa"/>
<circle cx="638" cy="67" r="28" fill="#ffdd94"/>
<path d="M0 194 Q95 154 190 194 T380 194 T570 194 T760 194 V340 H0Z" fill="#9ed8e1"/>
<path d="M0 241 Q95 205 190 241 T380 241 T570 241 T760 241 V340 H0Z" fill="#087f8c"/>
<g fill="#80d9b2"><circle cx="331" cy="257" r="9"/><circle cx="361" cy="277" r="13"/><circle cx="391" cy="246" r="8"/><circle cx="423" cy="273" r="11"/></g>
<g transform="translate(188 76) rotate(-18)" stroke="#153e50" stroke-width="3"><rect x="-65" y="-16" width="42" height="32" fill="#489bcc"/><rect x="23" y="-16" width="42" height="32" fill="#489bcc"/><rect x="-20" y="-22" width="40" height="44" rx="7" fill="white"/></g>
<path d="M195 109 L331 211 M210 103 L420 211" stroke="#4595ce" stroke-width="2" stroke-dasharray="6 6"/>
<g font-family="Arial,sans-serif" fill="#153e50"><text x="30" y="35" font-size="20" font-weight="bold">Ocean colour → investigation</text><text x="30" y="60" font-size="15">Satellite observations reveal chlorophyll-a patterns</text><text x="30" y="319" font-size="15" fill="white">Observe · Screen · Compare context · Verify independently</text></g></svg>'''
            st.image(illustration, caption="Illustrative workflow; chlorophyll-a patterns are screening evidence, not bloom confirmation.", use_container_width=True)
    with st.container(border=True):
        st.markdown("#### Read the snapshot responsibly")
        st.write("This is a dated data product, not a real-time monitoring feed. A model flag prioritizes review; an unflagged cell does not guarantee safe water. Historical screening is a separate, coarser source of context.")


def _ui_historical_field(history, selected):
    dates = _ui_dates(history)
    frame = history.loc[dates.eq(pd.Timestamp(selected))].copy()
    if frame.empty:
        return frame
    aggregations = {c:("max" if c == "risk_flag" else "mean") for c in ["chla", "risk_flag", "historical_baseline", "chla_anomaly", "chla_change"] if c in frame}
    return frame.groupby(["latitude", "longitude"], as_index=False).agg(aggregations).assign(date=pd.Timestamp(selected))


def _ui_risk_map(latest):
    st.markdown("### Risk Map")
    history = _ui_history()
    latest_date = _ui_latest_date(latest)
    all_dates = sorted(set(_ui_dates(history).dropna().tolist()) | set(_ui_dates(latest).dropna().tolist()))
    if not all_dates:
        st.info("No dated observations are available for the map.")
        return
    default = latest_date if pd.notna(latest_date) else all_dates[-1]
    if st.session_state.get("bd_map_date") not in all_dates:
        st.session_state["bd_map_date"] = default
    def move_date(step):
        current = st.session_state.get("bd_map_date", default)
        index = all_dates.index(current)
        st.session_state["bd_map_date"] = all_dates[max(0, min(len(all_dates)-1, index+step))]
    with st.container(border=True):
        if len(all_dates) > 1:
            selected = st.select_slider("Observation date (UTC)", options=all_dates, format_func=_ui_date_label, key="bd_map_date")
        else:
            selected = all_dates[0]
            st.write(f"Observation date (UTC): {_ui_date_label(selected)}")
        prev, nxt, label = st.columns([1, 1, 4])
        prev.button("← Previous date", on_click=move_date, args=(-1,), disabled=selected == all_dates[0])
        nxt.button("Next date →", on_click=move_date, args=(1,), disabled=selected == all_dates[-1])
        label.caption("Historical playback: step through the available observation dates. Gaps are not filled.")
    is_latest = selected == latest_date
    field = latest if is_latest else _ui_historical_field(history, selected)
    if field.empty:
        st.info("This date has no available spatial records.")
        return
    if is_latest:
        st.caption("Latest stored model result · native processed-cell resolution. Flags are the supplied model outputs.")
        _ui_snapshot(field)
    else:
        st.caption("Compact historical screening · aggregated 1° regional bins · these flags are not historical ML predictions.")
        a,b,c = st.columns(3)
        a.metric("Observed regional bins", len(field))
        b.metric("Flagged regional bins", int(_ui_flagged(field).sum()))
        c.metric("Flagged bin share", f"{_ui_flagged(field).mean():.1%}")
        st.caption("Counts and share describe observed 1° bins. Original fine-grid cell counts are unavailable.")
    _ui_map(field, "bd_map_layer")
    with st.expander("Flagged records for this date"):
        flagged = field.loc[_ui_flagged(field)]
        if flagged.empty:
            st.info("No supplied screening flags for this date.")
        else:
            st.dataframe(flagged, hide_index=True, use_container_width=True)
            st.download_button("Download flagged records CSV", _ui_csv(flagged), "bloomdetect_flagged_records.csv", "text/csv")


def _ui_probability(row):
    for name in ("risk_probability", "bloom_probability", "probability", "risk_prob", "prediction_probability", "bloom_prob"):
        if name in row.index:
            try:
                probability = float(row[name])
                if np.isfinite(probability) and 0 <= probability <= 1:
                    return f"{probability:.1%}", name
            except (TypeError, ValueError):
                pass
            return "Unavailable", name
    return "Unavailable", None


def _ui_evidence(row, distance, series, history, latest_date):
    regional_metrics = None
    st.markdown("#### Evidence & provenance")
    stamp = pd.to_datetime(row.get("date", latest_date), errors="coerce", utc=True)
    age = (pd.Timestamp.now(tz="UTC").normalize() - stamp.normalize()).days if pd.notna(stamp) else None
    a,b,c = st.columns(3)
    a.metric("Observation date (UTC)", _ui_date_label(stamp))
    b.metric("Age at review (UTC days)", str(age) if age is not None else "Unavailable")
    c.metric("Nearest-cell distance", _ui_value(distance, " km", 1))
    st.caption("Dated stored output; not real-time. Distance is to a processed coordinate, not proof of ocean coverage. No land/ocean mask is available.")
    if not history.empty:
        with st.container(border=True):
            table = persistence_table(history, as_of=latest_date, window_days=60)
            lat, lon = float(row["latitude"]), float(row["longitude"])
            # Match the regional series coordinate rather than assuming fine-grid history.
            if not series.empty and not table.empty:
                regional_lat, regional_lon = series.iloc[-1]["latitude"], series.iloc[-1]["longitude"]
                match = table.loc[np.isclose(_ui_numeric(table,"latitude"),regional_lat) & np.isclose(_ui_numeric(table,"longitude"),regional_lon)]
            else:
                match = pd.DataFrame()
            if not match.empty:
                metrics = match.iloc[0]
                regional_metrics = metrics
                x,y,z = st.columns(3)
                x.metric("Observed dates / 60-day window", _ui_value(metrics.get("observed_count"), digits=0))
                y.metric("Available-date coverage", _ui_value(metrics.get("coverage_pct"), "%", 1))
                z.metric("Largest observation gap", _ui_value(metrics.get("max_gap_days"), " days", 0))
                st.write(f"Flagged recurrence: {_ui_value(metrics.get('recurrence_pct'), '%', 1)} · Current run: {_ui_value(metrics.get('current_run'), digits=0)} successive flagged observations.")
            st.caption("Coverage is relative to dates available in the historical dataset, not expected daily observations. Runs describe successive available observations, not uninterrupted bloom days. Historical resolution: 1° regional bins; latest resolution: source processed cells.")
    return regional_metrics


def _ui_location(latest):
    st.markdown("### Location")
    st.caption("Enter latitude and longitude in decimal degrees. The result is the nearest processed cell, with 1° regional historical context where available.")
    input_column, result_column = st.columns([.85, 1.15], gap="large")
    with input_column, st.container(border=True):
        st.markdown("#### Your input")
        with st.form("bd_location_form"):
            a,b = st.columns(2)
            lat = a.number_input("Latitude (°; north positive)", min_value=-90.0, max_value=90.0, value=15.0, step=0.25)
            lon = b.number_input("Longitude (°; east positive)", min_value=-180.0, max_value=180.0, value=65.0, step=0.25)
            submitted = st.form_submit_button("Inspect location", type="primary")
    if submitted:
        st.session_state["bd_location"] = (lat,lon)
    lat,lon = st.session_state.get("bd_location", (15.0,65.0))
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        st.warning(f"This coordinate is outside the processed region ({LAT_MIN}–{LAT_MAX}° latitude; {LON_MIN}–{LON_MAX}° longitude). Choose a coordinate within the data bounds.")
        return
    row, distance = nearest_cell(latest,lat,lon)
    if row is None:
        st.info("No processed cell is available for this location.")
        return
    with result_column, st.container(border=True):
        st.markdown("#### Nearest processed cell")
        st.caption("Ocean validity is unverified: no land/ocean mask is supplied.")
        a,b = st.columns(2)
        a.metric("Processed latitude", f"{row['latitude']:.4f}°")
        b.metric("Processed longitude", f"{row['longitude']:.4f}°")
        st.write(f"Your submitted input: **{lat:.4f}°, {lon:.4f}°**")
        a,b = st.columns(2)
        a.metric("Cell chlorophyll-a", _ui_value(row.get('chla'), ' mg/m³'))
        probability, probability_column = _ui_probability(row)
        b.metric("Model screening probability", probability)
        flag = pd.to_numeric(row.get("risk_flag"),errors="coerce")
        if flag == 1:
            st.error("Potential bloom-risk screening flag — review the supporting evidence.")
        else:
            st.success("Not flagged by the stored model. This does not establish safe water.")
    if distance is not None and distance > 50:
        st.warning("The nearest processed cell is more than 50 km away. This display heuristic indicates a weak location match; it is not a scientific screening threshold.")
    chla = pd.to_numeric(row.get("chla"), errors="coerce")
    if chla == 0:
        st.warning("The source reports zero chlorophyll-a. Check source retrieval and missing-value conventions before interpreting it; zero does not establish land or absence of phytoplankton.")
    st.caption("Probability is shown only for a finite source value on the 0–1 scale. A flag or probability does not establish toxicity, a confirmed bloom or safe water. Land/ocean classification is unknown.")
    with st.container(border=True):
        st.markdown("#### Supporting model signals")
        signals = [(name, label) for name, label in [("historical_baseline", "Source historical baseline"), ("chla_anomaly", "Source anomaly"), ("chla_change", "Source recent change"), ("previous_chla", "Previous source observation")] if name in row.index]
        if signals:
            columns = st.columns(len(signals))
            for column, (name, label) in zip(columns, signals):
                column.metric(label, _ui_value(row.get(name), " mg/m³"))
            st.caption("Signals from the latest supplied model record; they are distinct from the coarser regional history below.")
        else:
            st.info("Supporting model signals are not supplied for this record.")
    history = _ui_history()
    latest_date = _ui_latest_date(latest)
    series = location_history(history,float(row["latitude"]),float(row["longitude"])) if not history.empty else pd.DataFrame()
    if not series.empty:
        series = series.copy()
        series["date"] = _ui_dates(series)
        series = series.sort_values("date")
        with st.container(border=True):
            _ui_plot(px.line(series, x="date", y="chla", markers=True, labels={"chla":"Chlorophyll-a (mg/m³)","date":"Observation date (UTC)"}), "Containing 1° bin: regional chlorophyll-a context")
            st.caption("Regional bin averages are not the exact history of the latest fine-grid cell. The latest model probability is not inferred from this series.")
        last = series.iloc[-1]
        a,b,c = st.columns(3)
        a.metric("Regional historical baseline", _ui_value(last.get("historical_baseline")," mg/m³"))
        b.metric("Regional anomaly", _ui_value(last.get("chla_anomaly")," mg/m³"))
        c.metric("Regional recent change", _ui_value(last.get("chla_change")," mg/m³"))
    else:
        st.info("No regional history is available for the containing bin.")
    regional_metrics = _ui_evidence(row,distance,series,history,latest_date)
    report = ["BloomDetect AI — investigation evidence report", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",f"Requested coordinate: {lat}, {lon}",f"Nearest processed coordinate: {row['latitude']}, {row['longitude']}",f"Distance km: {distance}",f"Stored model observation UTC: {row.get('date',latest_date)}",f"Chlorophyll-a mg/m³: {row.get('chla')}",f"Stored model risk flag: {row.get('risk_flag')}",f"Probability: {probability}; source field: {probability_column or 'unavailable'}",f"Regional history observations: {len(series)}",f"Regional history range: {_ui_date_label(_ui_dates(series).min())} to {_ui_date_label(_ui_dates(series).max())}","Provenance: EOS-06 / OCM-3 E06OCM_L4_AC ocean-colour chlorophyll-a; supplied latest model output and compact historical screening.","Limitations: not real-time; not a confirmed bloom, toxicity diagnosis or ocean/land verification. History is 1-degree regional context, not exact cell history. Coverage refers to available observation dates; successive flagged observations are not continuous days."]
    if not series.empty:
        for field in ("historical_baseline","chla_anomaly","chla_change"):
            report.append(f"Last regional {field}: {series.iloc[-1].get(field, 'unavailable')}")
    report.append(f"Recurrence window UTC (inclusive): {_ui_date_label(latest_date - pd.Timedelta(days=59))} to {_ui_date_label(latest_date)}")
    report.append(history.attrs.get('screening_disclosure', 'Historical screening unavailable.'))
    if regional_metrics is not None:
        for field in ("observed_count", "flagged_count", "recurrence_pct", "coverage_pct", "max_gap_days", "current_run", "last_observed"):
            report.append(f"Containing-bin {field}: {regional_metrics.get(field, 'unavailable')}")
    else:
        report.append("Containing-bin recurrence: unavailable; no missing values are treated as zero.")
    with st.container(border=True):
        st.markdown("#### Investigation downloads")
        st.download_button("Download evidence report TXT", "\n".join(report),"bloomdetect_investigation.txt","text/plain")
        evidence = row.to_frame().T.copy()
        evidence["requested_latitude"], evidence["requested_longitude"] = lat,lon
        evidence["distance_km"] = distance
        evidence["history_observation_count"] = len(series)
        evidence["history_resolution"] = "1-degree regional bins; not exact cell history"
        evidence["provenance_limitations"] = "Stored model; not real-time or confirmed bloom; no land/ocean verification"
        st.download_button("Download investigation CSV",_ui_csv(evidence),"bloomdetect_investigation.csv","text/csv")
        if not series.empty:
            st.download_button("Download containing-bin history CSV",_ui_csv(series),"bloomdetect_regional_history.csv","text/csv")


def _ui_insights(latest):
    st.markdown("### Insights")
    st.caption("Explore current model signals, spatial patterns and coarse historical evidence for investigation.")
    a, b, c = st.columns(3)
    change = _ui_numeric(latest, "chla_change")
    anomaly = _ui_numeric(latest, "chla_anomaly")
    a.metric("Cells with positive change", f"{int(change.gt(0).sum()):,}" if change.notna().any() else "Unavailable")
    b.metric("Cells at/above anomaly threshold", f"{int(anomaly.ge(ANOMALY_THRESHOLD).sum()):,}" if anomaly.notna().any() else "Unavailable")
    c.metric("Current model-flagged cells", f"{int(_ui_flagged(latest).sum()):,}")
    st.caption("Latest source cells: positive change > 0; anomaly ≥ the supplied anomaly threshold. These signals alone are not confirmed blooms.")
    zones = hotspot_table(latest)
    with st.container(border=True):
        st.markdown("#### Observed 2° zones")
        st.caption("Floor-based 2° boundaries include all observed cells, including unflagged cells. Counts depend on sampling coverage; zones are not harm rankings.")
        if not zones.empty:
            top = zones.head(12).sort_values("flagged_count")
            _ui_plot(px.bar(top, x="flagged_count", y="zone", orientation="h", text="flagged_count", labels={"flagged_count":"Model-flagged cells", "zone":"2° coordinate bounds"}, color_discrete_sequence=["#087f8c"]), "Flagged-cell counts by 2° zone", height=470)
            st.dataframe(zones.head(20), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left, st.container(border=True):
        _ui_plot(px.histogram(latest, x="chla", nbins=40, labels={"chla":"Chlorophyll-a (mg/m³)"}, color_discrete_sequence=["#4595ce"]), "Latest concentration distribution")
        st.caption("All valid source cells, including zeros. A zero is not evidence of safe water.")
    with right, st.container(border=True):
        groups = latest.copy()
        groups["Screening group"] = np.where(_ui_flagged(groups), "Flagged", "Not flagged")
        signals = [column for column in ["chla_anomaly", "chla_change"] if _ui_numeric(groups, column).notna().any()]
        if signals:
            for column in signals:
                groups[column] = _ui_numeric(groups, column)
            means = groups.groupby("Screening group", as_index=False)[signals].mean().melt(id_vars="Screening group", var_name="Signal", value_name="Mean (mg/m³)")
            means["Signal"] = means["Signal"].map({"chla_anomaly":"Anomaly", "chla_change":"Recent change"})
            _ui_plot(px.bar(means, x="Signal", y="Mean (mg/m³)", color="Screening group", barmode="group", color_discrete_map={"Flagged":"#ef4f5e", "Not flagged":"#4595ce"}), "Supporting signals by screening group")
            st.caption("Group means over available source values; screening groups do not establish biological severity.")
        else:
            st.info("Anomaly and change signals are not supplied for group comparison.")
    history = _ui_history()
    _ui_regional_insights(latest, history, _ui_latest_date(latest))


def _ui_regional_insights(latest,history,latest_date):
    st.markdown("### Regional investigation context")
    if history.empty:
        st.caption("Regional recurrence and historical charts need the optional history dataset.")
        _ui_shortlist(latest, history, latest_date)
        return
    field = _ui_historical_field(history,_ui_dates(history).max())
    a,b = st.columns(2)
    a.metric("Bins with positive anomaly",int(_ui_numeric(field,"chla_anomaly").gt(0).sum()))
    b.metric("Bins with positive recent change",int(_ui_numeric(field,"chla_change").gt(0).sum()))
    st.caption(f"Latest available historical-screening date: {_ui_date_label(_ui_dates(history).max())}. Counts refer to 1° bins, not source cells.")
    with st.container(border=True):
        _ui_plot(px.histogram(field,x="chla",nbins=30,labels={"chla":"Regional-bin chlorophyll-a (mg/m³)"}),"Current historical-bin concentrations")
    records = history.copy()
    records["date"] = _ui_dates(records)
    records["chla"] = _ui_numeric(records,"chla")
    records["flagged"] = _ui_flagged(records).astype(float)
    per_bin = records.groupby(["date","latitude","longitude"],as_index=False).agg(chla=("chla","mean"),flagged=("flagged","max"))
    summaries = per_bin.groupby("date",as_index=False).agg(mean_chla=("chla","mean"),flagged_share=("flagged","mean"))
    with st.container(border=True):
        _ui_plot(px.line(summaries,x="date",y="flagged_share",markers=True,labels={"flagged_share":"Share of observed regional bins flagged","date":"Observation date (UTC)"}),"Historical screening share")
        st.caption("Historical screening only; no latest ML flags are mixed into this series. The observed-bin denominator can vary by date.")
    with st.container(border=True):
        _ui_plot(px.line(summaries,x="date",y="mean_chla",markers=True,labels={"mean_chla":"Mean regional-bin chlorophyll-a (mg/m³)","date":"Observation date (UTC)"}),"Historical screening: mean chlorophyll-a")
        st.caption("Unweighted mean over available regional bins; changing coverage can affect the mean.")
    with st.container(border=True):
        st.markdown("#### Regional proxy recurrence")
        window_days = st.selectbox("Lookback window (calendar days)", [30, 60, 90], index=1, key="bd_recurrence_window")
        persistence = persistence_table(history,as_of=latest_date,window_days=window_days)
        start = latest_date - pd.Timedelta(days=window_days - 1)
        st.caption(f"Inclusive window: {_ui_date_label(start)}–{_ui_date_label(latest_date)} UTC. Coverage uses available dataset dates, not expected daily observations. Recurrence is flagged / observed dates; missing observations are not unflagged observations.")
        if not persistence.empty:
            ranked = persistence.sort_values([c for c in ["recurrence_pct","observed_count"] if c in persistence],ascending=False)
            st.dataframe(ranked.head(20),hide_index=True,use_container_width=True)
            st.caption(f"{persistence.attrs.get('available_dataset_dates', 0):,} available dataset dates. Ranked by recurrence, then observed count. Largest gaps are between recorded observations; runs count successive available dates, not continuous bloom days. Sparse coverage limits interpretation.")
            st.download_button("Download regional recurrence CSV", _ui_csv(persistence), "bloomdetect_regional_recurrence.csv", "text/csv")
        else:
            st.info("No observations are available within this window.")
    _ui_shortlist(latest, history, latest_date)


def _ui_shortlist(latest, history, latest_date):
    with st.container(border=True):
        st.markdown("#### Latest flagged-cell investigation shortlist")
        shortlist = rank_investigations(latest,history,as_of=latest_date)
        st.caption("Current model flags associated with their containing 1° bins; this coarse evidence is not exact-cell history. Review order: 60-day regional proxy recurrence, latest cell change, then available-date coverage (descending; missing context last). This is an investigation order, not a harm or severity ranking.")
        if shortlist.empty:
            st.info("No latest flagged cells are available for a shortlist.")
        else:
            columns = [name for name in ["latitude", "longitude", "chla", "chla_change", "lat_bin", "lon_bin", "recurrence_pct", "observed_count", "coverage_pct", "max_gap_days", "reason"] if name in shortlist]
            st.dataframe(shortlist[columns].head(25),hide_index=True,use_container_width=True)
            st.download_button("Download full investigation shortlist CSV",_ui_csv(shortlist),"bloomdetect_investigation_shortlist.csv","text/csv")
        report = ["BloomDetect AI — regional investigation context", f"Latest model date UTC: {_ui_date_label(latest_date)}", f"Current model-flagged cells: {int(_ui_flagged(latest).sum()):,}", "Source: EOS-06 / OCM-3 E06OCM_L4_AC ocean-colour chlorophyll-a; supplied model output.", "Shortlist context: containing 1-degree historical proxy bins over the inclusive 60-day window; not exact-cell history.", "Order: recurrence, latest change, coverage; no harm score. Coverage refers to available dataset dates, not daily completeness. Missing history does not imply zero recurrence.", "Not real-time; flags do not confirm a bloom, toxicity or safe water. No verified land mask.", "", shortlist.head(25).to_csv(index=False)]
        st.download_button("Download insights report TXT", "\n".join(report), "bloomdetect_insights_report.txt", "text/plain")


def _ui_data(latest):
    st.markdown("### Data & provenance")
    st.write("**Source product:** EOS-06 / OCM-3, **E06OCM_L4_AC**, distributed through MOSDAC. Ocean-colour chlorophyll-a is a proxy for phytoplankton concentration.")
    st.markdown("[MOSDAC data portal](https://www.mosdac.gov.in/) · [Ocean Colour Monitor information](https://www.isro.gov.in/Oceansat3.html)")
    st.markdown("[Source product algorithm document](https://www.mosdac.gov.in/docs/ATBD_AnalysedCHL_GlobalOcean.pdf)")
    history = _ui_history()
    with st.container(border=True):
        st.markdown("#### Coverage: latest model output")
        st.write(f"{len(latest):,} processed records · latest date {_ui_date_label(_ui_latest_date(latest))} UTC.")
        st.caption("Native processed coordinates, approximately 0.25° where supplied; inspect the source grid. Original retained columns and ML output are preserved in the download. This file is distinct from the compact regional history.")
        st.dataframe(latest.head(20),hide_index=True,use_container_width=True)
        st.download_button("Download latest model output CSV",_ui_csv(latest),"bloomdetect_latest_model_output.csv","text/csv")
        st.download_button("Download potential-risk cells CSV",_ui_csv(latest.loc[_ui_flagged(latest)]),"bloomdetect_potential_risk.csv","text/csv")
    with st.container(border=True):
        st.markdown("#### Coverage: compact historical screening")
        if history.empty:
            st.info("Optional historical dataset unavailable.")
        else:
            dates = _ui_dates(history)
            st.write(f"{len(history):,} regional-bin records · {dates.nunique():,} available dates · {_ui_date_label(dates.min())}–{_ui_date_label(dates.max())} UTC.")
            st.caption("Aggregated 1° regional bins. Original fine-cell counts are unavailable in this compact dataset. Historical screening flags are not stored ML predictions. Missing dates and bins are not interpolated.")
            st.dataframe(history.head(20),hide_index=True,use_container_width=True)
            st.caption(history.attrs.get('screening_disclosure', ''))
            source = Path(BASE_DIR) / history.attrs.get('source_file', 'bloomdetect_history_web.csv.gz')
            if source.exists():
                st.download_button("Download original compact history file", source.read_bytes(), source.name, "application/gzip" if source.suffix == '.gz' else 'text/csv')
    with st.container(border=True):
        st.markdown("#### Concise field dictionary")
        dictionary = pd.DataFrame([
            ("date","Observation date, interpreted in UTC; not a live-feed timestamp"),
            ("latitude / longitude","Decimal degrees; latest processed coordinates or historical 1° bin coordinates"),
            ("chla","Chlorophyll-a concentration in mg/m³; source zeros require provenance review"),
            ("risk_flag","Latest: supplied model flag. History: compact screening flag. 1=flagged, 0=not flagged"),
             ("historical_baseline","History: mean of strictly earlier observations in the same 1° bin, mg/m³. Latest: supplied model feature; see source pipeline"),
             ("chla_anomaly / chla_change","History: current minus strictly prior baseline / preceding available observation, mg/m³. Latest: supplied source features"),
             ("lat_bin / lon_bin","Containing 1° bin centres; not exact fine-grid locations"),
             ("stored_risk_flag","Original compact-file flag retained for comparison; historical displays use the recomputed prior-observation proxy"),
            ("recurrence_pct","Flagged observations / observed observations, within the window"),
            ("coverage_pct","Observed bin dates / available dataset dates in the window; not daily completeness"),
            ("max_gap_days / current_run","Gap between observations in days / successive flagged observations, not continuous days"),
            ("model probability, if supplied","Shown only when finite and in [0,1]; never guessed from percentage-like values"),
        ],columns=["Field","Meaning"])
        st.dataframe(dictionary,hide_index=True,use_container_width=True)
        st.caption(f"Historical screening thresholds supplied by the pipeline: anomaly {ANOMALY_THRESHOLD}; recent change {CHANGE_THRESHOLD}. These contextual indicators do not independently confirm a bloom.")
        st.download_button("Download field dictionary CSV",_ui_csv(dictionary),"bloomdetect_dictionary.csv","text/csv")


def render_page():
    st.markdown(STYLES,unsafe_allow_html=True)
    if "bd_page" not in st.session_state:
        st.session_state["bd_page"] = "Home"
    pages = {"Home":_ui_home,"Risk Map":_ui_risk_map,"Location":_ui_location,"Insights":_ui_insights,"Data":_ui_data}
    if st.session_state["bd_page"] not in pages:
        st.session_state["bd_page"] = "Home"
    navigation = st.columns(5)
    for column, name in zip(navigation,pages):
        if column.button(name,key=f"bd_nav_{name}",type="primary" if st.session_state["bd_page"] == name else "secondary",use_container_width=True):
            st.session_state["bd_page"] = name
            st.rerun()
    st.markdown('<div class="bd-hero"><div class="bd-eyebrow">Ocean colour · scientific screening</div><h1>BloomDetect AI</h1><p>Explore chlorophyll-a patterns, investigate screening flags and understand the evidence behind each observation.</p></div>',unsafe_allow_html=True)
    try:
        latest = load_latest()
        if not latest.empty:
            latest = latest.loc[latest['date'].eq(latest['date'].max())].copy()
    except Exception:
        logging.exception("Unable to load latest model snapshot")
        st.error("The latest snapshot could not be loaded. Please check that the supplied data file is available and try again.")
        return
    if latest is None or latest.empty:
        st.info("There are no processed observations to display yet. Add the latest prediction dataset to begin.")
        return
    if not {"latitude","longitude","chla","risk_flag"}.issubset(latest.columns):
        st.info("The latest snapshot is missing required screening fields. Please provide the complete prediction dataset.")
        return
    page = st.session_state["bd_page"]
    try:
        pages[page](latest)
    except Exception:
        LOGGER.exception("BloomDetect %s page rendering failed", page)
        st.error("This view could not be displayed. Try another page or reload the snapshot.")
    st.caption("BloomDetect AI · dated satellite-derived screening evidence · independent verification required")


try:
    render_page()
except Exception:
    logging.exception("BloomDetect page rendering failed")
    st.error("This view could not be displayed. Try another view or reload the snapshot. Your source data has not been changed.")
