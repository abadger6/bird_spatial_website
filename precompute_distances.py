import os
import json
import numpy as np
import geopandas as gpd

# Directories for bird and energy infrastructure data
BIRD_DIR = "birds"
INFRA_DIR = "infrastructure"

# List all bird and energy files
bird_files = sorted([f for f in os.listdir(BIRD_DIR) if f.endswith('.gpkg')])
infra_files = sorted([f for f in os.listdir(INFRA_DIR) if f.endswith('.geojson')])

def compute_distances(bird_path, infra_path):
    try:
        bird_gdf = gpd.read_file(bird_path)
        infra_gdf = gpd.read_file(infra_path)
        # Reproject to EPSG:3857 for accurate distance computations.
        if bird_gdf.crs != "EPSG:3857":
            bird_gdf = bird_gdf.to_crs(epsg=3857)
        if infra_gdf.crs != "EPSG:3857":
            infra_gdf = infra_gdf.to_crs(epsg=3857)
        distances = []
        # For each bird feature, compute the minimum distance to any energy feature.
        for geom in bird_gdf.geometry:
            dists = infra_gdf.geometry.distance(geom)
            distances.append(dists.min())
        return distances  # returns list of distances (in meters)
    except Exception as e:
        print(f"Error computing distances for {bird_path} and {infra_path}: {e}")
        return None

# Dictionary to hold results: {bird_file: {infra_file: distances, ...}, ...}
distance_results = {}

for bird_file in bird_files:
    bird_path = os.path.join(BIRD_DIR, bird_file)
    print(f"Processing bird file: {bird_file}")
    distance_results[bird_file] = {}
    for infra_file in infra_files:
        infra_path = os.path.join(INFRA_DIR, infra_file)
        print(f"    Processing energy file: {infra_file}")
        dists = compute_distances(bird_path, infra_path)
        distance_results[bird_file][infra_file] = dists

# Save the precomputed distances to a JSON file.
with open("distance_results.json", "w") as f:
    json.dump(distance_results, f, indent=4)

print("Distance results computed and stored in 'distance_results.json'")
