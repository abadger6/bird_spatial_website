import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define directories containing your geospatial files
BIRD_DIR = "birds"
INFRA_DIR = "infrastructure"

# Get list of available files:
# For birds, we continue to use .gpkg files.
# For infrastructure, we now look for .geojson files.
bird_files = sorted([f for f in os.listdir(BIRD_DIR) if f.endswith('.gpkg')])
infra_files = sorted([f for f in os.listdir(INFRA_DIR) if f.endswith('.geojson')])

# Create dropdown options
bird_options = [{"label": os.path.splitext(f)[0], "value": f} for f in bird_files]
infra_options = [{"label": os.path.splitext(f)[0], "value": f} for f in infra_files]

# Define the tab20 colors (hex codes) from the Tableau 20 palette
tab20_colors = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5"
]

# Helper function to convert hex color to rgba string with a given opacity (e.g., 0.8)
def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"

# Helper function to compute correlation for one energy layer.
def compute_correlation_per_layer(bird_gdf, infra_gdf):
    bird_proj = bird_gdf.to_crs(epsg=3857)
    infra_proj = infra_gdf.to_crs(epsg=3857)
    distances = []
    for geom in bird_proj.geometry:
        dists = infra_proj.geometry.distance(geom)
        distances.append(dists.min())
    if "abd" in bird_gdf.columns:
        abd = bird_gdf["abd"].values
        if len(abd) == len(distances) and len(abd) > 1:
            r = np.corrcoef(abd, distances)[0, 1]
            return r
    return None

# Helper function to compute distances (in meters) from each bird record to the nearest energy feature.
def compute_distances(bird_gdf, infra_gdf):
    bird_proj = bird_gdf.to_crs(epsg=3857)
    infra_proj = infra_gdf.to_crs(epsg=3857)
    distances = []
    for geom in bird_proj.geometry:
        distances.append(infra_proj.geometry.distance(geom).min())
    return np.array(distances)

# A helper function to convert a GeoDataFrame to lists of lat/lon (using centroid for non‐point geometries)
def extract_lat_lon(gdf):
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    lats, lons = [], []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "Point":
            point = geom
        else:
            point = geom.centroid
        lats.append(point.y)
        lons.append(point.x)
    return lats, lons

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Bird & Energy Infrastructure Mapping"

# Layout of the app, now with an extra Graph for the scatter plots.
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Bird & Energy Infrastructure Mapping Suite"), width=8),
        dbc.Col(html.H4("by Your Name"), width=4, className="text-end align-self-center")
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select a Bird Species:"),
            dcc.Dropdown(id="bird-dropdown", options=bird_options,
                         placeholder="Choose a species...", clearable=False),
            html.Br(),
            html.Label("Select Energy Infrastructure (multiple allowed):"),
            dcc.Dropdown(id="infra-dropdown", options=infra_options,
                         multi=True, placeholder="Select infrastructure...")
        ], width=8),
        dbc.Col(html.Img(src="/assets/logo.png", style={"maxWidth": "100%", "height": "auto"}),
                width=4, className="text-end")
    ], className="my-3"),
    
    dbc.Row([
        dbc.Col(html.Button("Generate Map", id="map-button", n_clicks=0, className="btn btn-primary"),
                width={"size": 4, "offset": 4}, className="text-center")
    ], className="my-3"),
    
    # Row to display the map and status/analysis text.
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-map",
                type="default",
                children=dcc.Graph(id="map-graph", config={"displayModeBar": True}, style={"height": "80vh"})
            ),
            html.Div(id="status-message", className="mt-3", style={"fontSize": "18px"})
        ])
    ]),
    # New row to display the individual scatter plots.
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-scatter",
                type="default",
                children=dcc.Graph(id="scatter-plot", config={"displayModeBar": True})
            )
        )
    ])
], fluid=True)

# Callback to update the map, perform correlation analysis, and generate individual scatter plots.
@app.callback(
    Output("map-graph", "figure"),
    Output("status-message", "children"),
    Output("scatter-plot", "figure"),
    Input("map-button", "n_clicks"),
    State("bird-dropdown", "value"),
    State("infra-dropdown", "value")
)
def update_map(n_clicks, selected_bird, selected_infra):
    if n_clicks is None or n_clicks == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            mapbox=dict(style="open-street-map", center={"lat": 39, "lon": -98}, zoom=3, layers=[]),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return empty_fig, "Select layers and click 'Generate Map' to display the map.", go.Figure()
    
    # Create the base map with extra top margin for legend/colorbar.
    fig = go.Figure()
    fig.update_layout(
        mapbox=dict(style="open-street-map", center={"lat": 39, "lon": -98}, zoom=3, layers=[]),
        margin={"r": 0, "t": 80, "l": 0, "b": 0}
    )
    
    status_message = ""
    correlation_results = {}  # To hold correlation for each energy layer.
    scatter_data = []         # To store scatter data for each energy layer.
    
    # Bird layer plotting with "abd" based marker size and color.
    if selected_bird:
        bird_path = os.path.join(BIRD_DIR, selected_bird)
        try:
            bird_gdf = gpd.read_file(bird_path)
            if bird_gdf.crs != "EPSG:4326":
                bird_gdf = bird_gdf.to_crs("EPSG:4326")
            bird_lats, bird_lons = extract_lat_lon(bird_gdf)
            
            if "abd" in bird_gdf.columns:
                abd_values = bird_gdf["abd"]
                min_abd = abd_values.min()
                max_abd = abd_values.max()
                if max_abd - min_abd > 0:
                    sizes = 5 + 15 * ((abd_values - min_abd) / (max_abd - min_abd))
                else:
                    sizes = [10] * len(abd_values)
            else:
                sizes = 10
                abd_values = None

            # Position the colorbar above the map.
            colorbar_settings = dict(title="abd", x=0.5, y=1.05, orientation="h", xanchor="center", yanchor="bottom")
            marker_kwargs = dict(size=sizes, colorscale="Viridis", showscale=True, colorbar=colorbar_settings)
            if abd_values is not None:
                marker_kwargs["color"] = abd_values
            else:
                marker_kwargs["color"] = "blue"
            
            fig.add_trace(go.Scattermapbox(
                lat=bird_lats, lon=bird_lons,
                mode="markers",
                marker=go.scattermapbox.Marker(**marker_kwargs),
                name=os.path.splitext(selected_bird)[0]
            ))
        except Exception as e:
            print("Error loading bird file:", e)
            status_message += "Error loading bird data. "
    else:
        status_message += "Please select a bird species. "
        return fig, status_message, go.Figure()

    # Process each selected energy infrastructure layer.
    if selected_infra:
        for i, infra_file in enumerate(selected_infra):
            infra_path = os.path.join(INFRA_DIR, infra_file)
            try:
                infra_gdf = gpd.read_file(infra_path)
                if infra_gdf.crs != "EPSG:4326":
                    infra_gdf = infra_gdf.to_crs("EPSG:4326")
                
                # Choose a color from tab20 with 80% opacity.
                chosen_color = hex_to_rgba(tab20_colors[i % len(tab20_colors)], 0.8)
                geom_types = infra_gdf.geom_type.unique()
                
                # For point-type infrastructures, plot markers; for non-points, add as layers.
                if all(g in ["Point", "MultiPoint"] for g in geom_types):
                    infra_lats, infra_lons = extract_lat_lon(infra_gdf)
                    fig.add_trace(go.Scattermapbox(
                        lat=infra_lats, lon=infra_lons,
                        mode="markers",
                        marker=go.scattermapbox.Marker(size=8, color=chosen_color),
                        name=os.path.splitext(infra_file)[0]
                    ))
                else:
                    geojson = json.loads(infra_gdf.to_json())
                    if any(g in ["Polygon", "MultiPolygon"] for g in geom_types):
                        layer_type = "fill"
                    else:
                        layer_type = "line"
                    
                    current_layers = list(fig.layout.mapbox.layers)
                    current_layers.append(dict(
                        sourcetype="geojson", source=geojson,
                        type=layer_type, color=chosen_color
                    ))
                    fig.layout.mapbox.layers = current_layers
                    
                    # Add a dummy legend trace for non-point layers.
                    dummy_trace = go.Scattermapbox(
                        lat=[0, 0], lon=[0, 0],
                        mode="lines",
                        line=dict(color=chosen_color, width=3),
                        name=os.path.splitext(infra_file)[0],
                        hoverinfo="none",
                        visible="legendonly"
                    )
                    fig.add_trace(dummy_trace)
                
                # Perform correlation analysis for this energy layer (if "abd" exists).
                if "abd" in bird_gdf.columns:
                    r = compute_correlation_per_layer(bird_gdf, infra_gdf)
                    if r is not None:
                        correlation_results[os.path.splitext(infra_file)[0]] = r
                        
                    # Compute scatter data: x = bird abd, y = distances (in meters).
                    distances = compute_distances(bird_gdf, infra_gdf)
                    scatter_data.append({
                        "name": os.path.splitext(infra_file)[0],
                        "x": bird_gdf["abd"].values,
                        "y": distances,
                        "color": chosen_color
                    })
            except Exception as e:
                print(f"Error loading infrastructure file {infra_file}:", e)
                status_message += f"Error loading {infra_file}. "
    else:
        status_message += "Please select at least one energy infrastructure layer. "
    
    # Update legend layout to display horizontally above the map.
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    
    # Create individual scatter plots for each energy layer using subplots.
    if scatter_data:
        # Determine the global y-axis range across all energy layers.
        global_y_min = min(np.min(d["y"]) for d in scatter_data)
        global_y_max = max(np.max(d["y"]) for d in scatter_data)
        n_plots = len(scatter_data)
        subplot_titles = [f"Scatter Plot: {d['name']}" for d in scatter_data]
        scatter_fig = make_subplots(rows=n_plots, cols=1, shared_yaxes=True, subplot_titles=subplot_titles)
        
        for idx, d in enumerate(scatter_data, start=1):
            scatter_fig.add_trace(
                go.Scatter(
                    x=d["x"],
                    y=d["y"],
                    mode="markers",
                    marker=dict(color=d["color"], size=8),
                    name=d["name"]
                ),
                row=idx, col=1
            )
            # Set the y-axis range for this subplot.
            scatter_fig.update_yaxes(range=[global_y_min, global_y_max], row=idx, col=1)
        
        scatter_fig.update_layout(
            title="Individual Scatter Plots: Bird Abundance (abd) vs. Distance to Energy Infrastructure",
            xaxis_title="Bird Abundance (abd)",
            yaxis_title="Distance (meters)",
            margin={"r": 20, "t": 80, "l": 40, "b": 40}
        )
    else:
        scatter_fig = go.Figure()
    
    # Append correlation analysis results to the status message.
    if correlation_results:
        analysis_text = "\nCorrelation Analysis (Pearson r between bird 'abd' and distance):\n"
        for layer_name, r_value in correlation_results.items():
            analysis_text += f"• {layer_name}: {r_value:.2f}\n"
        strongest_layer = min(correlation_results, key=lambda k: correlation_results[k])
        analysis_text += f"\nLayer with strongest (most negative) correlation: {strongest_layer}"
        status_message += analysis_text

    if status_message == "":
        status_message = "Map generated successfully."

    return fig, status_message, scatter_fig

if __name__ == '__main__':
    app.run_server(debug=True)
