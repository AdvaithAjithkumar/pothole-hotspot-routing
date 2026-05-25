"""
Large-Scale Pothole Hotspot Analysis & Safe Route Recommendation
KJ Somaiya BDA SEM VI — Real Chicago 311 Dataset Version
Run: python pothole_pipeline_chicago.py
"""

import os, time, warnings, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
warnings.filterwarnings('ignore')

# ── Set JAVA_HOME and HADOOP_HOME ────────────────────────────────────────────
if os.name == 'nt':
    os.environ['JAVA_HOME']   = r'C:\Java\jdk-11.0.30+7'
    os.environ['HADOOP_HOME'] = r'C:\hadoop'
    os.environ['PATH']        = os.environ['PATH'] + r';C:\hadoop\bin'
    os.environ['PYSPARK_PYTHON']       = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
else:
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
    
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import osmnx as ox
import networkx as nx
import folium

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR  = './pothole_pipeline_output/'
FETCH_LIMIT = 50000
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f'✓ Output directory: {OUTPUT_DIR}')

# ════════════════════════════════════════════════════════════════════════════
# SPARK SESSION
# ════════════════════════════════════════════════════════════════════════════
spark = SparkSession.builder \
    .appName('PotholeHotspotAnalysis_Chicago') \
    .master('local[*]') \
    .config('spark.driver.memory', '8g') \
    .config('spark.sql.shuffle.partitions', '16') \
    .config('spark.ui.showConsoleProgress', 'false') \
    .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
print(f'✓ Spark {spark.version} ready — {spark.sparkContext.defaultParallelism} cores')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — FETCH REAL CHICAGO 311 POTHOLE DATA VIA API
# ════════════════════════════════════════════════════════════════════════════
print('\n── Phase 1: Fetching Real Chicago 311 Pothole Data ───────────────────')

def fetch_chicago_potholes(limit=50000):
    """
    Fetch real pothole service requests from Chicago Open Data Portal.
    API: Socrata SODA — no authentication required.
    Source: https://data.cityofchicago.org/resource/7as2-ds3y.json
    """
    url    = "https://data.cityofchicago.org/resource/7as2-ds3y.json"
    params = {
        '$limit' : limit,
        '$where' : "latitude IS NOT NULL AND longitude IS NOT NULL",
        '$order' : 'creation_date DESC'
    }
    print(f'  Fetching {limit:,} records from Chicago Open Data API...')
    t0       = time.time()
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    raw = pd.DataFrame(response.json())
    print(f'  ✓ Fetched {len(raw):,} records in {time.time()-t0:.1f}s')

    # ── Map to pipeline schema ───────────────────────────────────────────────
    df = pd.DataFrame()
    df['report_id']   = raw['service_request_number']
    df['lat']         = raw['latitude'].astype(float)
    df['lon']         = raw['longitude'].astype(float)
    df['damage_type'] = 'D40'
    df['source']      = 'Chicago_311_Real'
    df['timestamp']   = pd.to_datetime(raw['creation_date']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # ── Derive severity from status ──────────────────────────────────────────
    # Open-Dup = lower priority (already reported) → severity 2
    # Open     = new report → severity 3
    # Closed   = resolved but was real → severity 4 (confirmed damage)
    status_map = {
        'Open - Dup' : 2,
        'Open'       : 3,
        'Closed'     : 4,
    }
    df['severity'] = raw['status'].map(status_map).fillna(3).astype(int)

    # Drop any remaining nulls
    df = df.dropna(subset=['lat','lon','severity','report_id'])
    df = df.reset_index(drop=True)

    print(f'  ✓ Schema mapped: {len(df):,} clean records')
    print(f'    Severity distribution: {df.severity.value_counts().sort_index().to_dict()}')
    print(f'    Date range: {df.timestamp.min()} → {df.timestamp.max()}')
    print(f'    Lat range : {df.lat.min():.4f} → {df.lat.max():.4f}')
    print(f'    Lon range : {df.lon.min():.4f} → {df.lon.max():.4f}')
    return df


real_df = fetch_chicago_potholes(limit=FETCH_LIMIT)

# ── Save ONLY the 7 schema columns to CSV — no extra columns ────────────────
csv_path = OUTPUT_DIR + 'pothole_reports.csv'
real_df[['report_id','lat','lon','severity','damage_type','source','timestamp']].to_csv(
    csv_path, index=False
)
print(f'✓ Real dataset saved → {csv_path}')

# ── Dataset overview plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Chicago 311 Real Pothole Dataset Overview', fontsize=13, fontweight='bold')

sev_counts = real_df.severity.value_counts().sort_index()
colors_sev = ['#2ecc71','#f1c40f','#e67e22','#e74c3c','#8e44ad']
axes[0].bar(sev_counts.index, sev_counts.values,
            color=colors_sev[:len(sev_counts)], edgecolor='white')
axes[0].set_title('Severity Distribution')
axes[0].set_xlabel('Severity'); axes[0].set_ylabel('Count')

# Year only used for plotting — never saved to CSV
year_series = pd.to_datetime(real_df['timestamp']).dt.year
year_counts = year_series.value_counts().sort_index()
axes[1].bar(year_counts.index.astype(str), year_counts.values,
            color='#2E5FA3', edgecolor='white')
axes[1].set_title('Reports by Year')
axes[1].set_xlabel('Year'); axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

sample = real_df.sample(min(5000, len(real_df)), random_state=42)
axes[2].scatter(sample.lon, sample.lat, s=0.8, alpha=0.4, color='#E87722')
axes[2].set_title('Geographic Distribution (Chicago)')
axes[2].set_xlabel('Longitude'); axes[2].set_ylabel('Latitude')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'dataset_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Dataset overview chart saved')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — PYSPARK ETL
# ════════════════════════════════════════════════════════════════════════════
print('\n── Phase 3: PySpark ETL ───────────────────────────────────────────────')

REPORT_SCHEMA = StructType([
    StructField('report_id',   StringType(),  nullable=False),
    StructField('lat',         DoubleType(),  nullable=False),
    StructField('lon',         DoubleType(),  nullable=False),
    StructField('severity',    IntegerType(), nullable=False),
    StructField('damage_type', StringType(),  nullable=True),
    StructField('source',      StringType(),  nullable=True),
    StructField('timestamp',   StringType(),  nullable=True),
])

t_start = time.time()
raw_df = spark.read \
    .option('header', 'true') \
    .option('mode', 'PERMISSIVE') \
    .schema(REPORT_SCHEMA) \
    .csv(csv_path)
raw_count = raw_df.count()
print(f'✓ CSV ingested in {time.time()-t_start:.2f}s — {raw_count:,} records')

# Cleaning rules
after_r1   = raw_df.dropna(subset=['report_id','lat','lon','severity'])
r1_removed = raw_count - after_r1.count()
after_r2   = after_r1.filter((F.col('lat') >= -90)  & (F.col('lat') <= 90))
r2_removed = after_r1.count() - after_r2.count()
after_r3   = after_r2.filter((F.col('lon') >= -180) & (F.col('lon') <= 180))
r3_removed = after_r2.count() - after_r3.count()
after_r4   = after_r3.filter((F.col('severity') >= 1) & (F.col('severity') <= 5))
r4_removed = after_r3.count() - after_r4.count()
print(f'  R1 nulls removed       : {r1_removed:,}')
print(f'  R2 invalid lat removed : {r2_removed:,}')
print(f'  R3 invalid lon removed : {r3_removed:,}')
print(f'  R4 invalid sev removed : {r4_removed:,}')

# Deduplication — 5m grid
DEDUP_PRECISION = 0.00005
deduped_df = after_r4 \
    .withColumn('lat_grid', (F.floor(F.col('lat') / DEDUP_PRECISION) * DEDUP_PRECISION)) \
    .withColumn('lon_grid', (F.floor(F.col('lon') / DEDUP_PRECISION) * DEDUP_PRECISION))
window_dedup = Window.partitionBy('lat_grid','lon_grid').orderBy(F.col('severity').desc())
deduped_df = deduped_df \
    .withColumn('rank', F.row_number().over(window_dedup)) \
    .filter(F.col('rank') == 1) \
    .drop('rank','lat_grid','lon_grid')
dedup_count = deduped_df.count()
print(f'  Near-duplicates removed: {after_r4.count()-dedup_count:,}')
print(f'  Final clean records    : {dedup_count:,}')

clean_df = deduped_df \
    .withColumn('timestamp_parsed',
                F.to_timestamp(F.col('timestamp'), 'yyyy-MM-dd HH:mm:ss')) \
    .withColumn('is_pothole', F.col('damage_type') == 'D40') \
    .drop('timestamp') \
    .withColumnRenamed('timestamp_parsed', 'timestamp')

PARQUET_CLEAN = OUTPUT_DIR + 'cleaned_reports.parquet'
clean_df.write.mode('overwrite').parquet(PARQUET_CLEAN)
print(f'✓ Cleaned data written to Parquet')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — HOTSPOT DETECTION
# ════════════════════════════════════════════════════════════════════════════
print('\n── Phase 4: Spatial Hotspot Detection ────────────────────────────────')

GRID_PREC = 0.001
clean_df  = spark.read.parquet(PARQUET_CLEAN)

gridded_df = clean_df \
    .withColumn('lat_cell', F.round(F.floor(F.col('lat') / GRID_PREC) * GRID_PREC, 3)) \
    .withColumn('lon_cell', F.round(F.floor(F.col('lon') / GRID_PREC) * GRID_PREC, 3))

t0 = time.time()
cell_agg = gridded_df.groupBy('lat_cell','lon_cell').agg(
    F.count('*').alias('report_count'),
    F.avg('severity').alias('avg_severity'),
    F.sum(F.col('is_pothole').cast('int')).alias('pothole_count')
)
total_cells = cell_agg.count()
print(f'✓ Aggregation complete in {time.time()-t0:.2f}s — {total_cells:,} grid cells')

COUNT_THRESHOLD    = 3
SEVERITY_THRESHOLD = 2.0
hotspot_cells = cell_agg.filter(
    (F.col('report_count') >= COUNT_THRESHOLD) &
    (F.col('avg_severity') >= SEVERITY_THRESHOLD)
)
hotspot_count = hotspot_cells.count()
print(f'✓ Hotspot cells identified: {hotspot_count:,}')
print(f'  Coverage: {hotspot_count/total_cells*100:.1f}% of active cells')

max_count = hotspot_cells.agg(F.max('report_count')).collect()[0][0]
MAX_SEV   = 5.0
hotspot_scored = hotspot_cells \
    .withColumn('norm_count',    F.col('report_count') / float(max_count)) \
    .withColumn('norm_severity', F.col('avg_severity') / MAX_SEV) \
    .withColumn('cell_danger_score',
                F.round(0.6 * F.col('norm_count') + 0.4 * F.col('norm_severity'), 6))

PARQUET_HOTSPOTS = OUTPUT_DIR + 'hotspot_cells.parquet'
hotspot_scored.write.mode('overwrite').parquet(PARQUET_HOTSPOTS)
hotspot_pd = hotspot_scored.toPandas()
print(f'✓ Hotspot cells written to Parquet')
print(f'  Max danger score : {hotspot_pd.cell_danger_score.max():.4f}')
print(f'  Mean danger score: {hotspot_pd.cell_danger_score.mean():.4f}')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — ROAD SEGMENT DANGER SCORING
# ════════════════════════════════════════════════════════════════════════════
print('\n── Phase 5: Road Segment Danger Scoring ──────────────────────────────')

LAT_MIN = real_df.lat.min(); LAT_MAX = real_df.lat.max()
LON_MIN = real_df.lon.min(); LON_MAX = real_df.lon.max()
print(f'  Chicago bounds: lat [{LAT_MIN:.4f}, {LAT_MAX:.4f}], lon [{LON_MIN:.4f}, {LON_MAX:.4f}]')
print('Downloading Chicago road network (may take 60-120s)...')

t0 = time.time()
G  = ox.graph_from_place('Chicago, Illinois, USA', network_type='drive')
print(f'✓ Network downloaded in {time.time()-t0:.1f}s — {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges')

hotspot_lookup = {
    (round(row['lat_cell'], 3), round(row['lon_cell'], 3)): row['cell_danger_score']
    for _, row in hotspot_pd.iterrows()
}

def snap_to_cell(lat, lon, prec=0.001):
    return (round(np.floor(lat / prec) * prec, 3),
            round(np.floor(lon / prec) * prec, 3))

BASE_DANGER = 0.05
mapped = 0
for u, v, key, data in G.edges(keys=True, data=True):
    mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
    mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
    danger  = hotspot_lookup.get(snap_to_cell(mid_lat, mid_lon), BASE_DANGER)
    G[u][v][key]['danger_weight'] = float(danger) + 1e-9
    if danger > BASE_DANGER:
        mapped += 1
print(f'✓ Danger weights assigned — {mapped:,} / {G.number_of_edges():,} edges mapped to hotspots')

edge_records = [
    {
        'edge_u'       : int(u), 'edge_v': int(v), 'edge_key': int(k),
        'length_m'     : float(d.get('length', 0)),
        'danger_weight': float(d.get('danger_weight', BASE_DANGER)),
        'highway'      : str(d.get('highway', 'unclassified'))
            if not isinstance(d.get('highway'), list) else str(d['highway'][0])
    }
    for u, v, k, d in G.edges(keys=True, data=True)
]
edges_pd    = pd.DataFrame(edge_records)
EDGE_SCHEMA = StructType([
    StructField('edge_u',        LongType(),    True),
    StructField('edge_v',        LongType(),    True),
    StructField('edge_key',      IntegerType(), True),
    StructField('length_m',      DoubleType(),  True),
    StructField('danger_weight', DoubleType(),  True),
    StructField('highway',       StringType(),  True),
])
edges_spark = spark.createDataFrame(edges_pd, schema=EDGE_SCHEMA)
PARQUET_EDGES = OUTPUT_DIR + 'scored_edges.parquet'
edges_spark.write.mode('overwrite').parquet(PARQUET_EDGES)
print(f'✓ Scored edges written to Parquet')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 6 — ROUTE PLANNING
# ════════════════════════════════════════════════════════════════════════════
print('\n── Phase 6: Safe Route Planning ──────────────────────────────────────')

ORIGIN_COORDS = (41.8827, -87.6233)   # The Loop, Downtown Chicago
DEST_COORDS   = (41.8500, -87.6500)   # Pilsen neighbourhood, SW Chicago

orig_node = ox.nearest_nodes(G, ORIGIN_COORDS[1], ORIGIN_COORDS[0])
dest_node = ox.nearest_nodes(G, DEST_COORDS[1],   DEST_COORDS[0])
print(f'✓ Origin node: {orig_node}  |  Dest node: {dest_node}')

try:
    shortest_path = nx.astar_path(G, orig_node, dest_node, weight='length')
    print(f'✓ Shortest path: {len(shortest_path)} nodes')
except nx.NetworkXNoPath:
    shortest_path = []
    print('  No shortest path found')

try:
    safe_path = nx.astar_path(G, orig_node, dest_node, weight='danger_weight')
    print(f'✓ Safe path    : {len(safe_path)} nodes')
except nx.NetworkXNoPath:
    safe_path = []
    print('  No safe path found')

def route_metrics(G, path):
    total_len, total_dng, n = 0.0, 0.0, 0
    for u, v in zip(path[:-1], path[1:]):
        best_edge  = min(G[u][v].values(), key=lambda d: d.get('length', 0))
        total_len += best_edge.get('length', 0)
        total_dng += best_edge.get('danger_weight', BASE_DANGER)
        n += 1
    return total_len, (total_dng / n if n else 0), n

sp_len, sp_dng, sp_edges = route_metrics(G, shortest_path)
sf_len, sf_dng, sf_edges = route_metrics(G, safe_path)
AVG_SPEED_MPS = 8.33

danger_reduction_pct  = (sp_dng - sf_dng) / sp_dng * 100 if sp_dng else 0
distance_overhead_pct = (sf_len - sp_len) / sp_len * 100 if sp_len else 0
time_overhead_s       = (sf_len - sp_len) / AVG_SPEED_MPS

print(f'\n  {"Metric":<28} {"Shortest":>12} {"Safe":>12} {"Delta":>10}')
print(f'  {"-"*64}')
print(f'  {"Distance (m)":<28} {sp_len:>12.0f} {sf_len:>12.0f} {distance_overhead_pct:>+9.1f}%')
print(f'  {"Mean danger score":<28} {sp_dng:>12.5f} {sf_dng:>12.5f} {danger_reduction_pct:>+9.1f}%')
print(f'  {"Travel time (s)":<28} {sp_len/AVG_SPEED_MPS:>12.0f} {sf_len/AVG_SPEED_MPS:>12.0f} {time_overhead_s:>+9.0f}s')

# Folium map
map_centre = [(ORIGIN_COORDS[0]+DEST_COORDS[0])/2, (ORIGIN_COORDS[1]+DEST_COORDS[1])/2]
m = folium.Map(location=map_centre, zoom_start=13, tiles='OpenStreetMap')

def path_coords(G, path):
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]

if shortest_path:
    folium.PolyLine(path_coords(G, shortest_path),
                    color='#2E5FA3', weight=6, opacity=0.85,
                    tooltip=f'Shortest Path | {sp_len:.0f}m | danger: {sp_dng:.4f}').add_to(m)
if safe_path:
    folium.PolyLine(path_coords(G, safe_path),
                    color='#27AE60', weight=6, opacity=0.85,
                    tooltip=f'Safe Path | {sf_len:.0f}m | danger: {sf_dng:.4f}').add_to(m)

folium.Marker(ORIGIN_COORDS, tooltip='Origin — The Loop, Chicago',
              icon=folium.Icon(color='blue', icon='play', prefix='fa')).add_to(m)
folium.Marker(DEST_COORDS,   tooltip='Destination — Pilsen, Chicago',
              icon=folium.Icon(color='red',  icon='flag', prefix='fa')).add_to(m)

top_hs = hotspot_pd.nlargest(300, 'cell_danger_score')
for _, hs in top_hs.iterrows():
    folium.CircleMarker(
        location=[hs['lat_cell']+GRID_PREC/2, hs['lon_cell']+GRID_PREC/2],
        radius=max(3, float(hs['cell_danger_score'])*14),
        color='#E74C3C', fill=True, fill_opacity=0.45,
        tooltip=(f"Danger: {hs['cell_danger_score']:.3f} | "
                 f"Reports: {int(hs['report_count'])} | "
                 f"Avg sev: {hs['avg_severity']:.2f}")
    ).add_to(m)

legend_html = '''
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:12px 16px;border-radius:8px;
            border:2px solid #ccc;font-size:13px;line-height:1.8;">
  <b>Pothole-Aware Routing — Chicago</b><br>
  <span style="color:#2E5FA3">&#9644;&#9644;</span>&nbsp;Shortest Path<br>
  <span style="color:#27AE60">&#9644;&#9644;</span>&nbsp;Safe Path (danger-minimised)<br>
  <span style="color:#E74C3C">&#9679;</span>&nbsp;Real 311 Pothole Hotspot
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

MAP_PATH = OUTPUT_DIR + 'pothole_route_map_chicago.html'
m.save(MAP_PATH)
print(f'✓ Interactive map saved → {MAP_PATH}')
print('  Open pothole_route_map_chicago.html in your browser')

# ════════════════════════════════════════════════════════════════════════════
# PHASE 7 — SPARK vs PANDAS BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
print('\n── Phase 7: Spark vs Pandas Benchmark ────────────────────────────────')

def generate_benchmark_data(n, seed=99):
    np.random.seed(seed)
    LAT_C, LON_C = 41.85, -87.65
    lats = np.clip(np.random.normal(LAT_C, 0.08, n), LAT_C-0.15, LAT_C+0.15)
    lons = np.clip(np.random.normal(LON_C, 0.10, n), LON_C-0.20, LON_C+0.20)
    return pd.DataFrame({
        'report_id'  : [f'BENCH_{i:07d}' for i in range(n)],
        'lat'        : lats, 'lon': lons,
        'severity'   : np.random.choice([1,2,3,4,5], n, p=[0.10,0.25,0.35,0.20,0.10]),
        'damage_type': 'D40',
        'source'     : 'benchmark',
        'timestamp'  : '2022-06-01 00:00:00'
    })

BENCH_SCHEMA = StructType([
    StructField('report_id',   StringType(),  True),
    StructField('lat',         DoubleType(),  True),
    StructField('lon',         DoubleType(),  True),
    StructField('severity',    IntegerType(), True),
    StructField('damage_type', StringType(),  True),
    StructField('source',      StringType(),  True),
    StructField('timestamp',   StringType(),  True),
])

SIZES = [100_000, 500_000, 1_000_000, 5_000_000]
print('Pre-generating benchmark CSVs...')
bench_paths = {}
for sz in SIZES:
    p = OUTPUT_DIR + f'bench_{sz}.csv'
    generate_benchmark_data(sz).to_csv(p, index=False)
    bench_paths[sz] = p
    print(f'  ✓ {sz:>9,} rows → {p}')

spark_times, pandas_times = [], []
print(f'\n{"Size":>12}  {"Spark (s)":>10}  {"Pandas (s)":>10}  {"Speedup":>8}')
print('-' * 48)

for sz in SIZES:
    path = bench_paths[sz]

    # Spark
    t0   = time.time()
    s_df = (spark.read
                 .option('header', 'true')
                 .schema(BENCH_SCHEMA)
                 .csv(path)
                 .dropna(subset=['report_id','lat','lon','severity'])
                 .filter(
                     F.col('lat').between(-90, 90) &
                     F.col('lon').between(-180, 180) &
                     F.col('severity').between(1, 5)
                 ))
    _ = (s_df
             .withColumn('lat_cell', F.floor(F.col('lat') / GRID_PREC))
             .withColumn('lon_cell', F.floor(F.col('lon') / GRID_PREC))
             .groupBy('lat_cell','lon_cell')
             .agg(F.count('*'), F.avg('severity'))
             .count())
    t_spark = time.time() - t0
    spark_times.append(t_spark)

    # Pandas
    t0   = time.time()
    p_df = pd.read_csv(path)
    p_df = p_df.dropna(subset=['report_id','lat','lon','severity'])
    p_df = p_df[p_df['lat'].between(-90,90) & p_df['lon'].between(-180,180) & p_df['severity'].between(1,5)]
    p_df['lat_cell'] = (p_df['lat'] // GRID_PREC)
    p_df['lon_cell'] = (p_df['lon'] // GRID_PREC)
    _ = p_df.groupby(['lat_cell','lon_cell']).agg(count=('severity','count'), mean_sev=('severity','mean'))
    t_pandas = time.time() - t0
    pandas_times.append(t_pandas)

    speedup = t_pandas / t_spark if t_spark > 0 else 0
    print(f'{sz:>12,}  {t_spark:>10.3f}  {t_pandas:>10.3f}  {speedup:>7.2f}x')

speedups = [p/s for p, s in zip(pandas_times, spark_times)]

# ════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
print('\n── Generating Evaluation Dashboard ───────────────────────────────────')

fig = plt.figure(figsize=(17, 10))
fig.suptitle(
    'Evaluation Dashboard — Large-Scale Pothole Hotspot Analysis\n'
    'Real Chicago 311 Dataset | KJ Somaiya BDA SEM VI',
    fontsize=13, fontweight='bold', y=0.99
)

# 1. Benchmark bar chart
ax1 = fig.add_subplot(2, 3, 1)
labels = [f'{s//1000}K' for s in SIZES]
x = np.arange(len(SIZES)); w = 0.35
b1 = ax1.bar(x-w/2, spark_times,  w, label='PySpark', color='#2E5FA3')
b2 = ax1.bar(x+w/2, pandas_times, w, label='Pandas',  color='#E87722')
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_xlabel('Dataset Size'); ax1.set_ylabel('Time (s)')
ax1.set_title('ETL + GroupBy Benchmark'); ax1.legend(fontsize=8)
for bar in list(b1)+list(b2):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=7)

# 2. Speedup trend
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(SIZES, speedups, 'o-', color='#8E44AD', linewidth=2, markersize=8)
ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='Break-even (1x)')
ax2.set_xscale('log')
ax2.set_title('Spark Speedup vs Dataset Size (log scale)')
ax2.set_xlabel('Dataset Size'); ax2.set_ylabel('Speedup (Pandas/Spark)')
ax2.legend(fontsize=8)
for x_val, y_val in zip(SIZES, speedups):
    ax2.annotate(f'{y_val:.2f}x', (x_val, y_val),
                 textcoords='offset points', xytext=(6,4), fontsize=8)

# 3. Route comparison
ax3 = fig.add_subplot(2, 3, 3)
metrics = ['Distance (m)', 'Avg Danger\n(×10⁴)', 'Travel Time\n(min)']
sp_vals = [sp_len, sp_dng*1e4, sp_len/AVG_SPEED_MPS/60]
sf_vals = [sf_len, sf_dng*1e4, sf_len/AVG_SPEED_MPS/60]
x3 = np.arange(len(metrics))
ax3.bar(x3-0.2, sp_vals, 0.38, label='Shortest Path', color='#2E5FA3')
ax3.bar(x3+0.2, sf_vals, 0.38, label='Safe Path',     color='#27AE60')
ax3.set_xticks(x3); ax3.set_xticklabels(metrics, fontsize=9)
ax3.set_title('Route Comparison — Chicago'); ax3.legend(fontsize=8)

# 4. Pipeline data funnel
ax4 = fig.add_subplot(2, 3, 4)
stages   = ['Raw API Data', 'After ETL Clean', 'After Dedup', 'Hotspot Cells', 'Scored Edges']
values   = [raw_count, raw_count-r1_removed, dedup_count, hotspot_count, G.number_of_edges()]
colors_f = ['#2E5FA3','#4A7FC1','#6896C8','#E87722','#27AE60']
bars = ax4.barh(range(len(stages)), values, color=colors_f, height=0.55)
ax4.set_yticks(range(len(stages))); ax4.set_yticklabels(stages, fontsize=9)
ax4.set_xlabel('Count'); ax4.set_title('Pipeline Data Flow'); ax4.invert_yaxis()
for bar, val in zip(bars, values):
    ax4.text(bar.get_width()+50, bar.get_y()+bar.get_height()/2,
             f'{val:,}', va='center', fontsize=8)

# 5. Hotspot danger distribution
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(hotspot_pd['cell_danger_score'], bins=35,
         color='#E87722', edgecolor='white', alpha=0.8)
ax5.axvline(sp_dng, color='#2E5FA3', linestyle='--', linewidth=2,
            label=f'Shortest: {sp_dng:.4f}')
ax5.axvline(sf_dng, color='#27AE60', linestyle='--', linewidth=2,
            label=f'Safe:     {sf_dng:.4f}')
ax5.set_title('Danger Score Distribution vs Route Averages')
ax5.set_xlabel('Cell Danger Score'); ax5.set_ylabel('Cell Count')
ax5.legend(fontsize=7)

# 6. Summary table
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
rows = [
    ['Data source',                 'Chicago 311 Real API'],
    ['Input records (raw)',         f'{raw_count:,}'],
    ['After ETL + dedup',           f'{dedup_count:,}'],
    ['Hotspot cells detected',      f'{hotspot_count:,}'],
    ['Road edges scored',           f'{G.number_of_edges():,}'],
    ['Danger reduction (safe path)',f'{danger_reduction_pct:.1f} %'],
    ['Distance overhead',           f'{distance_overhead_pct:+.1f} %'],
    ['Time overhead',               f'{time_overhead_s/60:+.1f} min'],
    ['Speedup at 5M rows',          f'{speedups[-1]:.2f}x'],
    ['PySpark version',             f'{spark.version}'],
]
tbl = ax6.table(cellText=rows, colLabels=['Metric', 'Value'],
                loc='center', cellLoc='left')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2E5FA3')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EEF2F7')
ax6.set_title('Pipeline Summary', fontweight='bold', pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.97])
DASHBOARD_PATH = OUTPUT_DIR + 'evaluation_dashboard.png'
plt.savefig(DASHBOARD_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f'✓ Dashboard saved → {DASHBOARD_PATH}')

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*62)
print('  PIPELINE COMPLETE — REAL CHICAGO 311 DATASET')
print('='*62)
print(f'  Data source    : Chicago Open Data 311 API (real)')
print(f'  Records fetched: {raw_count:,}')
print(f'  After ETL+dedup: {dedup_count:,}')
print(f'  Hotspot cells  : {hotspot_count:,}')
print(f'  Edges scored   : {G.number_of_edges():,}')
print(f'  Danger reduced : {danger_reduction_pct:.1f}%')
print(f'  Spark speedup  : {speedups[-1]:.2f}x at 5M rows')
print(f'  Map saved      : {MAP_PATH}')
print(f'  Dashboard      : {DASHBOARD_PATH}')
print('='*62)

spark.stop()