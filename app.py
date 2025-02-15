import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load precomputed correlation results.
try:
    with open("correlation_results.json", "r") as f:
        precomputed_correlations = json.load(f)
except Exception as e:
    precomputed_correlations = {}
    print("Error loading precomputed correlation results:", e)

# Load precomputed distance results.
try:
    with open("distance_results.json", "r") as f:
        precomputed_distances = json.load(f)
except Exception as e:
    precomputed_distances = {}
    print("Error loading precomputed distance results:", e)

# Define directories for geospatial files.
BIRD_DIR = "birds"
INFRA_DIR = "infrastructure"

# List available files.
bird_files = sorted([f for f in os.listdir(BIRD_DIR) if f.endswith('.gpkg')])
infra_files = sorted([f for f in os.listdir(INFRA_DIR) if f.endswith('.geojson')])

# Create dropdown options.
bird_options = [{"label": os.path.splitext(f)[0], "value": f} for f in bird_files]
infra_options = [{"label": os.path.splitext(f)[0], "value": f} for f in infra_files]

# Define the tab20 colors from the Tableau 20 palette.
tab20_colors = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5"
]

# Helper: Convert hex color to rgba string with given opacity.
def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"

# Helper: Extract lat/lon lists from a GeoDataFrame.
def extract_lat_lon(gdf):
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    lats, lons = [], []
    for geom in gdf.geometry:
        if geom is None:
            continue
        point = geom if geom.geom_type == "Point" else geom.centroid
        lats.append(point.y)
        lons.append(point.x)
    return lats, lons

# Initialize the Dash app.
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Bird & Energy Infrastructure Mapping"
server = app.server

# Define the app layout.
app.layout = dbc.Container([
    # Header.
    dbc.Row([
        dbc.Col(html.H1("Bird & Energy Infrastructure Mapping Suite"), width=8),
        dbc.Col(html.H4("by Alex Badgett"), width=4, className="text-end align-self-center")
    ], className="my-3"),
    # Bulleted notes.
    dbc.Row([
        dbc.Col(
            html.Ul([
                html.Li("This app is an illustrative example of my interest in the intersection of bird conservation and existing and future energy infrastructure."),
                html.Li("It uses precomputed spatial correlations and distances combined with relative bird abundance to analyze areas of critical abundance with close proximity to energy infrastructure."),
                html.Li("Below the interactive map, additional plots provide further analysis and visualization examples."),
                html.Li("Please note that this is an illustrative example only! Results are not peer reviewed and should not be used for any other purposes."),
                html.Li("I would love to hear from you if you have thoughts or suggestions! badgett.alex@gmail.com"),
            ]),
            width=12
        )
    ], className="my-3"),
    # Dropdowns.
    dbc.Row([
        dbc.Col([
            html.Label("Select a Bird Species:"),
            dcc.Dropdown(id="bird-dropdown", options=bird_options,
                         placeholder="Choose a species...", clearable=False),
            html.Br(),
            html.Label("Select Energy Infrastructure (multiple allowed):"),
            dcc.Dropdown(id="infra-dropdown", options=infra_options,
                         multi=True, placeholder="Select infrastructure...")
        ], width=12)
    ], className="my-3"),
    # Generate Map button.
    dbc.Row([
        dbc.Col(html.Button("Generate Map", id="map-button", n_clicks=0, className="btn btn-primary"),
                width={"size": 4, "offset": 4}, className="text-center")
    ], className="my-3"),
    # Main interactive map.
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-map",
                type="default",
                children=dcc.Graph(id="map-graph", config={"displayModeBar": True}, style={"height": "80vh"})
            )
        )
    ]),
    # Status message row.
    dbc.Row([
        dbc.Col(
            html.Div(id="status-message", style={"fontSize": "18px", "marginTop": "10px"}),
            width=12
        )
    ]),
    # Section: Parallel Coordinates Plot.
    dbc.Row([
        dbc.Col(html.H5("Parallel Coordinates Plot: Comparing Abundance and Distances. Use the sliders for each vertical line to select a subset of variables to isolate. This may not work as well for mobile users. "), width=12)
    ], className="my-2"),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-parcoords",
                type="default",
                children=dcc.Graph(id="parcoords-plot", config={"displayModeBar": True}, style={"height": "80vh"})
            )
        )
    ]),
    # Section: Scatter Plot Matrix.
    dbc.Row([
        dbc.Col(html.H5("Scatter Plot Matrix: Abundance and Distances"), width=12)
    ], className="my-2"),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-splom",
                type="default",
                children=dcc.Graph(id="scatter-matrix", config={"displayModeBar": True}, style={"height": "80vh"})
            )
        )
    ]),
    # Section: Distance Scatter Plots.
    dbc.Row([
        dbc.Col(html.H5("Distance Scatter Plots: Abundance vs. Distance (km)"), width=12)
    ], className="my-2"),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-scatter",
                type="default",
                children=dcc.Graph(id="scatter-plot", config={"displayModeBar": True}, style={"height": "80vh"})
            )
        )
    ]),
    # Section: P90 Maps (each as a separate plot with title).
    dbc.Row([
        dbc.Col(html.H5("P90 Maps: Top 10% Abundance with Shortest Distances"), width=12)
    ], className="my-2"),
    dbc.Row([
        dbc.Col(
            html.Div(id="p90-map-container")
        )
    ]),
    # Data Sources.
    dbc.Row([
        dbc.Col(
            html.Div([
                html.P("Data Sources:", style={"fontWeight": "bold", "fontSize": "16px"}),
                html.Ul([
                    html.Li("Bird data downloaded from eBird"),
                    html.Li("Energy infrastructure data downloaded from United States Energy Information Administration"),
                    html.Li("Additional processing details available upon request")
                ])
            ], style={"textAlign": "center", "marginTop": "20px", "marginBottom": "20px"}),
            width=12
        )
    ])
], fluid=True)

# Callback to update all plots.
@app.callback(
    Output("map-graph", "figure"),
    Output("status-message", "children"),
    Output("scatter-plot", "figure"),
    Output("parcoords-plot", "figure"),
    Output("scatter-matrix", "figure"),
    Output("p90-map-container", "children"),
    Input("map-button", "n_clicks"),
    State("bird-dropdown", "value"),
    State("infra-dropdown", "value")
)
def update_plots(n_clicks, selected_bird, selected_infra):
    if n_clicks is None or n_clicks == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            mapbox=dict(style="open-street-map", center={"lat":39, "lon":-98}, zoom=3, layers=[]),
            margin={"r":0, "t":0, "l":0, "b":0}
        )
        return empty_fig, "Select layers and click 'Generate Map' to display the map.", go.Figure(), go.Figure(), go.Figure(), []
    
    # --- Main Interactive Map ---
    map_fig = go.Figure()
    map_fig.update_layout(
        mapbox=dict(style="open-street-map", center={"lat":39, "lon":-98}, zoom=3, layers=[]),
        margin={"r":0, "t":80, "l":0, "b":0}
    )
    
    status_message = ""
    correlation_results = {}
    scatter_data = []  # For individual distance scatter subplots.
    
    # Load bird data.
    try:
        bird_path = os.path.join(BIRD_DIR, selected_bird)
        bird_gdf = gpd.read_file(bird_path)
        if bird_gdf.crs != "EPSG:4326":
            bird_gdf = bird_gdf.to_crs("EPSG:4326")
        bird_lats, bird_lons = extract_lat_lon(bird_gdf)
    except Exception as e:
        status_message += "Error loading bird data. "
        return map_fig, status_message, go.Figure(), go.Figure(), go.Figure(), []
    
    # Plot bird points on the main map.
    if "abd" in bird_gdf.columns:
        abd_values = bird_gdf["abd"]
        min_abd, max_abd = abd_values.min(), abd_values.max()
        sizes = 5 + 15 * ((abd_values - min_abd) / (max_abd - min_abd)) if max_abd > min_abd else [10]*len(abd_values)
    else:
        sizes = 10
        abd_values = None

    colorbar_settings = dict(title="Abundance", x=0.5, y=1.05, orientation="h", xanchor="center", yanchor="bottom")
    marker_kwargs = dict(size=sizes, colorscale="Viridis", showscale=True, colorbar=colorbar_settings,
                         color=abd_values if abd_values is not None else "blue")
    map_fig.add_trace(go.Scattermapbox(
        lat=bird_lats, lon=bird_lons,
        mode="markers",
        marker=go.scattermapbox.Marker(**marker_kwargs),
        name=os.path.splitext(selected_bird)[0]
    ))
    
    # Process each selected energy infrastructure layer.
    if selected_infra:
        for i, infra_file in enumerate(selected_infra):
            try:
                infra_path = os.path.join(INFRA_DIR, infra_file)
                infra_gdf = gpd.read_file(infra_path)
                if infra_gdf.crs != "EPSG:4326":
                    infra_gdf = infra_gdf.to_crs("EPSG:4326")
                chosen_color = hex_to_rgba(tab20_colors[i % len(tab20_colors)], 0.8)
                geom_types = infra_gdf.geom_type.unique()
                if all(g in ["Point", "MultiPoint"] for g in geom_types):
                    infra_lats, infra_lons = extract_lat_lon(infra_gdf)
                    map_fig.add_trace(go.Scattermapbox(
                        lat=infra_lats, lon=infra_lons,
                        mode="markers",
                        marker=go.scattermapbox.Marker(size=8, color=chosen_color),
                        name=os.path.splitext(infra_file)[0]
                    ))
                else:
                    geojson = json.loads(infra_gdf.to_json())
                    layer_type = "fill" if any(g in ["Polygon", "MultiPolygon"] for g in geom_types) else "line"
                    current_layers = list(map_fig.layout.mapbox.layers)
                    current_layers.append(dict(
                        sourcetype="geojson", source=geojson,
                        type=layer_type, color=chosen_color
                    ))
                    map_fig.layout.mapbox.layers = current_layers
                    dummy_trace = go.Scattermapbox(
                        lat=[0,0], lon=[0,0],
                        mode="lines",
                        line=dict(color=chosen_color, width=3),
                        name=os.path.splitext(infra_file)[0],
                        hoverinfo="none",
                        visible="legendonly"
                    )
                    map_fig.add_trace(dummy_trace)
                    
                # Lookup precomputed correlations.
                if "abd" in bird_gdf.columns:
                    r = precomputed_correlations.get(selected_bird, {}).get(infra_file, None)
                    if r is not None:
                        correlation_results[os.path.splitext(infra_file)[0]] = r
                    # Lookup precomputed distances.
                    distances = precomputed_distances.get(selected_bird, {}).get(infra_file, None)
                    if distances is not None:
                        distances = np.array(distances)
                    else:
                        distances = np.array([])
                    scatter_data.append({
                        "name": os.path.splitext(infra_file)[0],
                        "x": bird_gdf["abd"].values,
                        "y": distances,  # in meters; will convert to km
                        "color": chosen_color
                    })
            except Exception as e:
                status_message += f"Error loading {infra_file}. "
    else:
        status_message += "Please select at least one energy infrastructure layer. "

    map_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    
    # --- Parallel Coordinates Plot ---
    if selected_infra and "abd" in bird_gdf.columns and scatter_data:
        data = {"Abundance": bird_gdf["abd"].values}
        for d in scatter_data:
            data[f"Distance_{d['name']}"] = d["y"] / 1000
        df = pd.DataFrame(data)
        parcoords_fig = go.Figure(data=go.Parcoords(
            line=dict(color=df["Abundance"], colorscale='Viridis', showscale=True, colorbar=dict(title="Abundance")),
            dimensions=[{"label": "Abundance", "values": df["Abundance"]}] +
                       [{"label": f"Distance ({d['name']}) (km)", "values": df[f"Distance_{d['name']}"]}
                        for d in scatter_data]
        ))
        parcoords_fig.update_layout(title="Parallel Coordinates: Abundance and Distances (km)",
                                      margin={"r":20, "t":80, "l":40, "b":40},
                                      plot_bgcolor="#f5f5f5", paper_bgcolor="#f5f5f5")
    else:
        parcoords_fig = go.Figure()
    
    # --- Scatter Plot Matrix ---
    if selected_infra and "Abundance" in df.columns and scatter_data:
        splom_fig = px.scatter_matrix(df,
                                      dimensions=df.columns,
                                      title="Scatter Plot Matrix: Abundance and Distances (km)",
                                      labels={col: col for col in df.columns},
                                      color="Abundance",
                                      color_continuous_scale="Viridis")
        # Automatically adjust tick font size based on number of dimensions.
        num_dims = len(df.columns)
        tick_font_size = max(8, 16 - num_dims)
        for key in splom_fig.layout:
            if key.startswith("xaxis"):
                splom_fig.layout[key].update(tickfont=dict(size=tick_font_size))
            if key.startswith("yaxis"):
                splom_fig.layout[key].update(tickfont=dict(size=tick_font_size))
        splom_fig.update_layout(margin={"r":20, "t":80, "l":40, "b":40},
                                plot_bgcolor="#f5f5f5", paper_bgcolor="#f5f5f5")
    else:
        splom_fig = go.Figure()
    
    # --- Distance Scatter Subplots ---
    if scatter_data:
        global_y_min = min(np.min(d["y"]) for d in scatter_data) / 1000
        global_y_max = max(np.max(d["y"]) for d in scatter_data) / 1000
        n_plots = len(scatter_data)
        subplot_titles = [f"{d['name']}" for d in scatter_data]
        scatter_fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=subplot_titles)
        for idx, d in enumerate(scatter_data, start=1):
            scatter_fig.add_trace(
                go.Scatter(
                    x=d["x"],
                    y=d["y"] / 1000,  # convert m to km
                    mode="markers",
                    marker=dict(color=d["color"], size=8),
                    name=d["name"]
                ),
                row=idx, col=1
            )
            scatter_fig.update_xaxes(title_text="Bird Abundance", gridcolor="rgba(0,0,0,0.2)", gridwidth=1, row=idx, col=1)
            scatter_fig.update_yaxes(title_text="Distance (km)", range=[global_y_min, global_y_max], gridcolor="rgba(0,0,0,0.2)", gridwidth=1, row=idx, col=1)
        scatter_fig.update_layout(title="Individual Scatter Plots: Abundance vs. Distance (km)",
                                  margin={"r":20, "t":80, "l":40, "b":40},
                                  plot_bgcolor="#f5f5f5", paper_bgcolor="#f5f5f5")
    else:
        scatter_fig = go.Figure()
    
    # --- P90 Maps as Separate Plots ---
    p90_graphs = []
    if selected_infra and "abd" in bird_gdf.columns:
        for i, infra_file in enumerate(selected_infra):
            try:
                distances = precomputed_distances.get(selected_bird, {}).get(infra_file, None)
                if distances is not None:
                    distances = np.array(distances)
                else:
                    distances = np.array([])
                p90_threshold = np.percentile(bird_gdf["abd"], 90)
                p90_idx = np.where(bird_gdf["abd"] >= p90_threshold)[0]
                if len(p90_idx) > 0 and len(distances) == len(bird_gdf):
                    p90_distances = distances[p90_idx]
                    dist_threshold = np.percentile(p90_distances, 25)
                    final_idx = p90_idx[p90_distances <= dist_threshold]
                else:
                    final_idx = np.array([])
                if len(final_idx) > 0:
                    p90_bird = bird_gdf.iloc[final_idx]
                    p90_lats, p90_lons = extract_lat_lon(p90_bird)
                else:
                    p90_lats, p90_lons = [], []
                p90_fig = go.Figure()
                p90_fig.update_layout(
                    title=f"P90 Map: {os.path.splitext(infra_file)[0]}",
                    mapbox=dict(style="open-street-map", center={"lat":39, "lon":-98}, zoom=3),
                    margin={"r":0, "t":60, "l":0, "b":30}  # extra bottom margin for title separation
                )
                trace = go.Scattermapbox(
                    lat=p90_lats,
                    lon=p90_lons,
                    mode="markers",
                    marker=go.scattermapbox.Marker(size=10, color=hex_to_rgba(tab20_colors[i % len(tab20_colors)], 0.8)),
                    name=f"P90: {os.path.splitext(infra_file)[0]}"
                )
                p90_fig.add_trace(trace)
                p90_graphs.append(dcc.Graph(figure=p90_fig, config={"displayModeBar": True}, style={"height": "80vh", "marginBottom": "30px"}))
            except Exception as e:
                status_message += f"Error processing P90 for {infra_file}. "
    # --- End P90 Maps ---
    
    if correlation_results:
        analysis_text = "\nCorrelation Analysis (Pearson r between bird 'Abundance' and distance):\n"
        for layer_name, r_value in correlation_results.items():
            analysis_text += f"• {layer_name}: {r_value:.2f}\n"
        strongest_layer = min(correlation_results, key=lambda k: correlation_results[k])
        analysis_text += f"\nLayer with strongest (most negative) correlation: {strongest_layer}"
        status_message += analysis_text

    if status_message == "":
        status_message = "Map generated successfully."
    
    return map_fig, status_message, scatter_fig, parcoords_fig, splom_fig, p90_graphs

if __name__ == '__main__':
    app.run_server(debug=True)
