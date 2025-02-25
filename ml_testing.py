import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

# Load energy infrastructure data
infra_gdf = gpd.read_file('infrastructure/Coal Power Plants.geojson')
# Reproject to ESRI:54012 if needed
infra_gdf = infra_gdf.to_crs("ESRI:54012")

# Define TIF files for each season
tif_paths = {
    'breeding': 'birds_ml/amekes/amekes_abundance_seasonal_breeding_mean_2022.tif',
    'non_breeding': 'birds_ml/amekes/amekes_abundance_seasonal_nonbreeding_mean_2022.tif',
    'pre_breeding': 'birds_ml/amekes/amekes_abundance_seasonal_prebreeding-migration_mean_2022.tif',
    'post_breeding': 'birds_ml/amekes/amekes_abundance_seasonal_postbreeding-migration_mean_2022.tif'
}

# Extract zonal statistics (mean abundance) for each season
for season, tif_path in tif_paths.items():
    stats = zonal_stats(infra_gdf, tif_path, stats="mean", nodata=-9999)
    infra_gdf[season] = [stat['mean'] for stat in stats]

def get_max_season(row):
    season_values = {season: row[season] for season in tif_paths.keys()}
    return max(season_values, key=season_values.get)

infra_gdf['max_season'] = infra_gdf.apply(get_max_season, axis=1)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define feature columns and label
features = list(tif_paths.keys())
X = infra_gdf[features]
y = infra_gdf['max_season']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print("Model accuracy: ", accuracy)



import dash
from dash import dcc, html
import plotly.express as px

# Compute centroids for plotting (assuming polygon features)
infra_gdf['lon'] = infra_gdf.geometry.centroid.x
infra_gdf['lat'] = infra_gdf.geometry.centroid.y

# Create a Plotly Express scatter mapbox
fig = px.scatter_mapbox(
    infra_gdf,
    lat='lat',
    lon='lon',
    color='max_season',
    mapbox_style="carto-positron",
    zoom=4,
    title="Predicted Season with Highest Bird Abundance"
)

# Build the Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Bird Abundance Prediction Near Energy Infrastructure"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
