import os
import json
import numpy as np
import geopandas as gpd

# Directories for bird and energy infrastructure data
BIRD_DIR = "birds"
INFRA_DIR = "infrastructure"

# List bird and energy files
bird_files = sorted([f for f in os.listdir(BIRD_DIR) if f.endswith('.gpkg')])
infra_files = sorted([f for f in os.listdir(INFRA_DIR) if f.endswith('.geojson')])

def compute_correlation_per_layer(bird_path, infra_path):
    try:
        bird_gdf = gpd.read_file(bird_path)
        infra_gdf = gpd.read_file(infra_path)
        # Reproject to a projected CRS for accurate distance calculations
        if bird_gdf.crs != "EPSG:3857":
            bird_gdf = bird_gdf.to_crs(epsg=3857)
        if infra_gdf.crs != "EPSG:3857":
            infra_gdf = infra_gdf.to_crs(epsg=3857)
        distances = []
        for geom in bird_gdf.geometry:
            dists = infra_gdf.geometry.distance(geom)
            distances.append(dists.min())
        if "abd" in bird_gdf.columns:
            abd = bird_gdf["abd"].values
            if len(abd) == len(distances) and len(abd) > 1:
                r = np.corrcoef(abd, distances)[0, 1]
                return r
    except Exception as e:
        print(f"Error computing correlation for {bird_path} and {infra_path}: {e}")
    return None

# Dictionary to hold results: {bird_file: {infra_file: correlation_value, ...}, ...}
correlation_results = {}

for bird_file in bird_files:
    bird_path = os.path.join(BIRD_DIR, bird_file)
    print(f"Processing bird file: {bird_file}")
    correlation_results[bird_file] = {}
    for infra_file in infra_files:
        infra_path = os.path.join(INFRA_DIR, infra_file)
        print(f"    Processing energy file: {infra_file}")
        r_value = compute_correlation_per_layer(bird_path, infra_path)
        correlation_results[bird_file][infra_file] = r_value

# Save results to a JSON file.
with open("correlation_results.json", "w") as f:
    json.dump(correlation_results, f, indent=4)

print("Correlation results computed and stored in 'correlation_results.json'")
