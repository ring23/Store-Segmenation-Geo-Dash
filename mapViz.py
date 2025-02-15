import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap, MiniMap, MeasureControl
import branca.colormap as cm
import plotly.express as px
import numpy as np
from geopy.distance import geodesic
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from shapely.geometry import Point, MultiPoint
from scipy.spatial import ConvexHull
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, mapping
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import folium.plugins as plugins

# Load your dataset
df = pd.read_csv("updated_store_location_v3.csv")
print(df.head())  # Verify the renaming worked

# Filter data: Only include rows where LAT and LNG are not null and IMPUTED_FLAG is 'Y'
df_map = df[(df['IMPUTED_FLAG'] == 'Y') | (~df['LAT'].isna() & ~df['LNG'].isna())]

def create_cluster_colors(df_map: pd.DataFrame) -> dict:
    """
    Creates a color mapping for clusters using distinct, vibrant colors.
    
    Args:
        df_map: DataFrame containing the store location data
    
    Returns:
        dict: Mapping of clusters to colors
    """
    # High-contrast, distinct colors optimized for visualization
    colors = [
        '#E41A1C',  # Bright red
        '#377EB8',  # Blue
        '#4DAF4A',  # Green
        '#984EA3',  # Purple
        '#FF7F00',  # Orange
        '#FFFF33',  # Yellow
        '#A65628',  # Brown
        '#F781BF',  # Pink
        '#1B9E77',  # Teal
        '#7570B3',  # Slate blue
        '#E6AB02',  # Gold
        '#66A61E'   # Lime green
    ]
    
    # Sort clusters and convert to strings to ensure consistent color assignment
    sorted_clusters = sorted(df_map["CLUSTERS"].unique())
    return {str(int(cluster)): colors[i % len(colors)] for i, cluster in enumerate(sorted_clusters)}

# Create color mapping
cluster_colors = create_cluster_colors(df_map)

def create_enhanced_map(df_map: pd.DataFrame, cluster_colors: dict) -> folium.Map:
    """
    Creates an enhanced map visualization with multiple layers and controls.
    
    Args:
        df_map: DataFrame containing the store location data
        cluster_colors: Dictionary mapping clusters to colors
    
    Returns:
        folium.Map: The configured map object
    """
    map_center = [df_map["LAT"].mean(), df_map["LNG"].mean()]
    m = folium.Map(
        location=map_center, 
        zoom_start=5,
        tiles=None,  # Remove default tiles
        prefer_canvas=True,
        zoom_control=True,
        scrollWheelZoom=True,
        dragging=True,
        zoom_snap=0.25,
        zoom_delta=0.25,
        max_zoom=18,
        min_zoom=2
    )

    # Add base layers with full opacity
    folium.TileLayer(
        tiles='cartodbpositron',
        name='Light Map',
        control=True,
        opacity=1.0,
        attr='CartoDB'
    ).add_to(m)

    # Add state boundaries layer
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        name='With Boundaries',
        control=True,
        opacity=1.0,
        attr='CartoDB'
    ).add_to(m)

    # Force white background and ensure tiles are visible
    folium.Element("""
        <style>
            .leaflet-container {
                background-color: white !important;
            }
            .leaflet-tile-container {
                opacity: 1 !important;
            }
            .leaflet-tile {
                opacity: 1 !important;
            }
        </style>
    """).add_to(m)

    # Create separate feature groups for different visualization options
    markers = folium.FeatureGroup(name='Individual Stores', show=True)
    clustered = MarkerCluster(name='Clustered View', show=False)
    heatmap_layer = folium.FeatureGroup(name='Heat Map', show=False)  # Set heatmap off by default

    # Add heatmap layer
    heat_data = df_map[['LAT', 'LNG']].values.tolist()
    HeatMap(heat_data).add_to(heatmap_layer)

    # Add individual markers with custom styling
    for _, row in df_map.iterrows():
        # Create a more detailed popup
        popup_html = f"""
        <div style="font-family: Arial; width: 150px;">
            <h4 style="color: #2c3e50;">Store Details</h4>
            <b>Unit ID:</b> {row['UNIT_ID']}<br>
            <b>Cluster:</b> {row['CLUSTERS']}<br>
        </div>
        """
        
        # Convert cluster to string when accessing cluster_colors
        cluster_str = str(int(row["CLUSTERS"]))
        
        # Add circle markers with improved styling for individual view
        folium.CircleMarker(
            location=[row["LAT"], row["LNG"]],
            radius=6,
            weight=2,
            color=cluster_colors[cluster_str],
            fill=True,
            fill_color=cluster_colors[cluster_str],
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"Unit ID: {row['UNIT_ID']}"
        ).add_to(markers)

        # Add markers to clustered view
        folium.CircleMarker(
            location=[row["LAT"], row["LNG"]],
            radius=6,
            weight=2,
            color=cluster_colors[cluster_str],
            fill=True,
            fill_color=cluster_colors[cluster_str],
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"Unit ID: {row['UNIT_ID']}"
        ).add_to(clustered)

    # Add all layers to map
    markers.add_to(m)
    clustered.add_to(m)
    heatmap_layer.add_to(m)

    # Add minimap
    minimap = MiniMap(
        toggle_display=True,
        position='bottomright',
        tile_layer=folium.TileLayer(
            tiles='cartodbpositron',
            attr='CartoDB',
            opacity=1.0
        ),
        zoom_level_offset=-5  # Show broader context
    )
    m.add_child(minimap)

    # Add measurement tool
    measure = MeasureControl(
        position='topleft',
        primary_length_unit='miles',
        secondary_length_unit='kilometers',
        primary_area_unit='sq-miles',
        secondary_area_unit='sq-kilometers',
        active_color='red',
        completed_color='red',
        popup_options={'className': 'measure-popup'}
    )
    m.add_child(measure)

    # Add CSS to style the measurement tool
    folium.Element("""
        <style>
            .measure-popup {
                font-family: Arial;
                font-size: 12px;
            }
            .leaflet-control-measure {
                background-color: white;
                padding: 5px;
                border-radius: 4px;
            }
            .leaflet-control-measure .leaflet-control-measure-interaction {
                background-color: #f8f9fa;
            }
        </style>
    """).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m

def calculate_cluster_spread_optimized(df_map: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average and maximum distances between stores in each cluster using optimized methods.
    
    Args:
        df_map: DataFrame containing store locations with LAT, LNG, and CLUSTERS columns
    
    Returns:
        DataFrame with cluster spread metrics in miles
    """
    spreads = []
    
    # Convert degrees to approximate miles (using average miles per degree)
    # This is an approximation but much faster than geodesic
    lat_mile_factor = 69
    lng_mile_factor = 54.6  # approximate for mid-latitudes
    
    for cluster in df_map["CLUSTERS"].unique():
        cluster_stores = df_map[df_map["CLUSTERS"] == cluster]
        
        if len(cluster_stores) > 1:
            # Convert lat/lng to approximate miles
            coords = cluster_stores[["LAT", "LNG"]].values
            coords_miles = np.column_stack([
                coords[:, 0] * lat_mile_factor,
                coords[:, 1] * lng_mile_factor
            ])
            
            # Calculate pairwise distances efficiently using scipy
            distances = pdist(coords_miles)
            
            spreads.append({
                "Cluster": cluster,
                "Average_Distance": round(np.mean(distances), 2),
                "Max_Distance": round(np.max(distances), 2),
                "Store_Count": len(cluster_stores)
            })
        else:
            spreads.append({
                "Cluster": cluster,
                "Average_Distance": 0,
                "Max_Distance": 0,
                "Store_Count": 1
            })
    
    return pd.DataFrame(spreads)

def calculate_cluster_distance_matrix(df_map: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average distances between clusters using optimized methods.
    
    Args:
        df_map: DataFrame containing store locations
    
    Returns:
        DataFrame: Symmetrical matrix of cluster-to-cluster distances
    """
    clusters = sorted(df_map["CLUSTERS"].unique())
    matrix = pd.DataFrame(index=clusters, columns=clusters)
    
    # Calculate cluster centroids for efficiency
    cluster_centroids = df_map.groupby("CLUSTERS")[["LAT", "LNG"]].mean()
    
    # Convert to miles factors (approximate)
    lat_mile_factor = 69
    lng_mile_factor = 54.6
    
    for c1 in clusters:
        for c2 in clusters:
            if c1 >= c2:  # Only calculate half (symmetrical matrix)
                if c1 == c2:
                    matrix.loc[c1, c2] = 0
                else:
                    # Convert lat/lng differences to approximate miles
                    lat_diff = (cluster_centroids.loc[c1, "LAT"] - cluster_centroids.loc[c2, "LAT"]) * lat_mile_factor
                    lng_diff = (cluster_centroids.loc[c1, "LNG"] - cluster_centroids.loc[c2, "LNG"]) * lng_mile_factor
                    dist = round(np.sqrt(lat_diff**2 + lng_diff**2), 1)
                    matrix.loc[c1, c2] = dist
                    matrix.loc[c2, c1] = dist
    
    return matrix

def plot_distance_matrix(distance_matrix: pd.DataFrame, cluster_colors: dict) -> go.Figure:
    """
    Create an interactive heatmap of cluster distances.
    
    Args:
        distance_matrix: DataFrame containing cluster-to-cluster distances
        cluster_colors: Dictionary mapping clusters to colors
    
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=distance_matrix.values,
        x=distance_matrix.columns,
        y=distance_matrix.index,
        hoverongaps=False,
        hovertemplate='Cluster %{x} to Cluster %{y}<br>Distance: %{z:.1f} miles<extra></extra>',
        colorscale='RdYlBu_r'
    ))
    
    # Update layout
    fig.update_layout(
        title="Inter-Cluster Distances (miles)",
        xaxis_title="Cluster",
        yaxis_title="Cluster",
        width=600,
        height=600,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}  # Match matrix orientation
    )
    
    return fig

def plot_enhanced_distance_matrix(distance_matrix: pd.DataFrame, geo_spread: pd.DataFrame, cluster_colors: dict) -> go.Figure:
    """
    Create an enhanced heatmap showing both inter-cluster distances and within-cluster spread.
    """
    # Create heatmap
    fig = go.Figure()
    
    # Add the distance matrix heatmap
    fig.add_trace(go.Heatmap(
        z=distance_matrix.values,
        x=distance_matrix.columns,
        y=distance_matrix.index,
        hoverongaps=False,
        hovertemplate='<b>Between Clusters:</b><br>' +
                      'Cluster %{x} to Cluster %{y}<br>' +
                      'Center Distance: %{z:.1f} miles<br>' +
                      '<extra></extra>',
        colorscale='RdYlBu_r'
    ))
    
    # Add annotations for within-cluster spread
    for cluster in distance_matrix.index:
        cluster_spread = geo_spread[geo_spread['Cluster'] == cluster]['Average_Distance'].values[0]
        fig.add_annotation(
            x=cluster,
            y=cluster,
            text=f'Spread: {cluster_spread:.1f} mi',
            showarrow=False,
            font=dict(size=8, color='black')
        )
    
    # Update layout
    fig.update_layout(
        title="Cluster Distance Analysis<br><sup>Diagonal shows average spread within clusters</sup>",
        xaxis_title="Cluster",
        yaxis_title="Cluster",
        width=700,
        height=600,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

# Create Streamlit interface
st.set_page_config(layout="wide")
st.title("Store Location Analysis Dashboard")

# Add description and instructions
st.markdown("""
### Interactive Store Cluster Map
Use the layer control in the top right to switch between:
- Individual stores (color-coded by cluster)
- Clustered view (stores grouped by proximity)
- Heat map (density visualization)

Click on markers for detailed information.
""")

# Create two columns for filters
col1, col2 = st.columns(2)

with col1:
    # Cluster filter with default handling
    available_clusters = sorted(df_map["CLUSTERS"].unique())
    selected_clusters = st.multiselect(
        "Select Clusters to Display",
        options=available_clusters,
        default=available_clusters
    )
    
    # If no clusters are selected, show all clusters
    if not selected_clusters:
        selected_clusters = available_clusters

with col2:
    # Store ID search
    store_id = st.text_input(
        "Search Store ID",
        placeholder="Enter Store ID..."
    )

# Filter the dataframe based on selections
filtered_df = df_map[df_map["CLUSTERS"].isin(selected_clusters)]
if store_id:
    filtered_df = filtered_df[filtered_df["UNIT_ID"].astype(str).str.contains(store_id, case=False)]

# Update map with filtered data
m = create_enhanced_map(filtered_df, cluster_colors)
folium_static(m, width=1200, height=600)

# Update the statistics section to be more efficient
# Cluster Statistics Section
st.markdown("### Cluster Statistics")

# Calculate all statistics at once to avoid multiple computations
@st.cache_data  # Cache the results
def calculate_all_statistics(df):
    """Calculate all statistics for the dashboard in one pass."""
    # Cluster counts
    cluster_counts = df["CLUSTERS"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    
    # Geographic spread
    geo_spread = calculate_cluster_spread_optimized(df)
    
    # Store details
    display_df = df[["UNIT_ID", "CLUSTERS", "LAT", "LNG"]].copy().round(4)
    
    return cluster_counts, geo_spread, display_df

# Calculate all statistics
cluster_counts, geo_spread, display_df = calculate_all_statistics(filtered_df)

# Display statistics in columns
col1, col2, col3 = st.columns(3)

with col1:
    # Convert cluster numbers to strings for consistent color mapping
    cluster_counts = df_map["CLUSTERS"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    cluster_counts['Cluster_Str'] = cluster_counts['Cluster'].apply(lambda x: str(int(x)))
    cluster_counts = cluster_counts.sort_values('Cluster')  # Sort to maintain consistent order

    fig1 = px.bar(
        cluster_counts,
        x="Cluster_Str",
        y="Count",
        title="Stores per Cluster",
        labels={'Count': 'Number of Stores', 'Cluster_Str': 'Cluster'},
        color="Cluster_Str",
        color_discrete_map=cluster_colors,
        category_orders={"Cluster_Str": sorted(cluster_counts['Cluster_Str'])}
    )
    fig1.update_layout(
        showlegend=False,
        xaxis_title="Cluster",
        yaxis_title="Number of Stores"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    if len(geo_spread) > 0:
        # Convert cluster numbers to strings for consistent color mapping
        geo_spread['Cluster_Str'] = geo_spread['Cluster'].apply(lambda x: str(int(x)))
        geo_spread = geo_spread.sort_values('Cluster')  # Sort to maintain consistent order

        fig2 = px.bar(
            geo_spread,
            x="Cluster_Str",
            y="Average_Distance",
            title="Average Distance Between Stores (miles)",
            labels={'Average_Distance': 'Miles', 'Cluster_Str': 'Cluster'},
            color="Cluster_Str",
            color_discrete_map=cluster_colors,
            category_orders={"Cluster_Str": sorted(geo_spread['Cluster_Str'])}
        )
        fig2.update_layout(
            showlegend=False,
            yaxis_title="Miles",
            xaxis_title="Cluster"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("""
        📊 **Geographic Spread Explanation**
        - Average Distance: Mean distance between all pairs of stores in each cluster
        - Larger values indicate stores are more spread out
        - Smaller values indicate stores are more tightly grouped
        """)

with col3:
    st.markdown("#### Cluster Details")
    if len(geo_spread) > 0:
        cluster_details = geo_spread[["Cluster", "Store_Count", "Average_Distance", "Max_Distance"]]
        cluster_details.columns = [
            "Cluster",
            "Store Count",
            "Avg Distance (mi)",
            "Max Distance (mi)"
        ]
        st.dataframe(
            cluster_details.set_index("Cluster"),
            use_container_width=True
        )
    else:
        st.warning("No data available for selected filters")

# Display store details
st.markdown("### Store Details")
if len(display_df) > 0:
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "UNIT_ID": "Store ID",
            "CLUSTERS": "Cluster",
            "LAT": "Latitude",
            "LNG": "Longitude"
        }
    )
else:
    st.warning("No stores match the selected filters.")

# Add download button for filtered data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_stores.csv",
    mime="text/csv"
)

# Add legend
st.sidebar.markdown("### Cluster Legend")
for cluster, color in cluster_colors.items():
    st.sidebar.markdown(
        f'<div style="display: flex; align-items: center;">'
        f'<div style="width: 20px; height: 20px; background-color: {color}; '
        f'margin-right: 10px; border-radius: 50%;"></div>'
        f'<span>Cluster {cluster}</span></div>',
        unsafe_allow_html=True
    )

# Add this after the statistics section
st.markdown("### Cluster Distance Analysis")

# Calculate and display the distance matrix
@st.cache_data
def get_distance_matrix(df):
    return calculate_cluster_distance_matrix(df)

distance_matrix = get_distance_matrix(filtered_df)

# Create two columns for visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(
        plot_enhanced_distance_matrix(distance_matrix, geo_spread, cluster_colors), 
        use_container_width=True
    )

with col2:
    st.markdown("""
    #### Distance Matrix Explanation
    - Values show approximate distances between cluster centers in miles
    - Darker colors indicate greater distances
    - Diagonal shows zero (same cluster)
    - Matrix is symmetrical
    
    **How to read:**
    1. Find a cluster pair (row and column)
    2. Color intensity shows relative distance
    3. Hover for exact distance values
    4. Use for understanding cluster relationships
    """)
    
    # Add download button for distance matrix
    csv_matrix = distance_matrix.to_csv()
    st.download_button(
        label="Download Distance Matrix",
        data=csv_matrix,
        file_name="cluster_distances.csv",
        mime="text/csv",
    )

    # Show summary statistics
    st.markdown("#### Quick Stats")
    max_dist = distance_matrix.max().max()
    avg_dist = distance_matrix.mean().mean()
    
    st.markdown(f"""
    - Average inter-cluster distance: {avg_dist:.1f} miles
    - Maximum inter-cluster distance: {max_dist:.1f} miles
    - Number of clusters: {len(distance_matrix)}
    """)

def add_state_analysis(df_map: pd.DataFrame, cluster_colors: dict) -> None:
    """
    Create state-level analysis visualizations.
    
    Args:
        df_map: DataFrame containing store locations with STATE and CLUSTERS columns
        cluster_colors: Dictionary mapping clusters to colors
    """
    # Create state-level breakdown
    state_breakdown = df_map.groupby(['STATE', 'CLUSTERS']).size().unstack(fill_value=0)
    state_totals = state_breakdown.sum(axis=1).sort_values(ascending=False)
    
    # Calculate percentages
    state_percentages = state_breakdown.div(state_breakdown.sum(axis=1), axis=0) * 100
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Total stores by state
        fig1 = px.bar(
            state_totals,
            title="Total Stores by State",
            labels={'index': 'State', 'value': 'Number of Stores'},
            height=400
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cluster composition by state
        fig2 = px.bar(
            state_breakdown,
            title="Cluster Composition by State",
            barmode='stack',
            color_discrete_map=cluster_colors,
            labels={'value': 'Number of Stores'},
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Add detailed breakdown table
    st.markdown("#### Detailed State Breakdown")
    state_details = pd.DataFrame({
        'Total Stores': state_totals,
        'Clusters Present': state_breakdown.apply(
            lambda x: ','.join(sorted(x[x > 0].index.astype(str))), 
            axis=1
        ),
        'Dominant Cluster': state_breakdown.idxmax(axis=1)
    }).round(2)
    
    st.dataframe(
        state_details,
        use_container_width=True,
        column_config={
            "Total Stores": st.column_config.NumberColumn(
                "Total Stores",
                help="Number of stores in each state"
            ),
            "Clusters Present": st.column_config.TextColumn(
                "Clusters Present",
                help="All clusters that have stores in this state"
            ),
            "Dominant Cluster": st.column_config.TextColumn(
                "Dominant Cluster",
                help="Cluster with the most stores in this state"
            )
        }
    )

def calculate_store_density(df_map: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate store density metrics.
    
    Args:
        df_map: DataFrame containing store locations
    
    Returns:
        DataFrame with added density metrics
    """
    from scipy.spatial import cKDTree
    
    # Create KD-tree for efficient nearest neighbor calculations
    coords = df_map[['LAT', 'LNG']].values
    tree = cKDTree(coords)
    
    # Calculate stores within different radii
    radii = [10, 25, 50]  # miles
    
    # Initialize columns
    for radius in radii:
        df_map[f'stores_within_{radius}mi'] = 0
    
    # Calculate for each point
    for radius in radii:
        radius_deg = radius / 69  # approximate conversion to degrees
        # Query all points at once instead of loop
        all_neighbors = tree.query_ball_point(coords, r=radius_deg)
        
        # Count neighbors for each point (subtract 1 to exclude self)
        df_map[f'stores_within_{radius}mi'] = [len(n) - 1 for n in all_neighbors]
    
    return df_map

def analyze_store_isolation(df_map: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Identify isolated stores and calculate nearest neighbor statistics.
    
    Args:
        df_map: DataFrame containing store locations
        n_neighbors: Number of nearest neighbors to consider
    
    Returns:
        DataFrame with isolation metrics added
    """
    from scipy.spatial import cKDTree
    
    coords = df_map[['LAT', 'LNG']].values
    tree = cKDTree(coords)
    
    # Find N nearest neighbors for each store
    distances, indices = tree.query(coords, k=n_neighbors+1)  # +1 because first point is self
    
    # Convert distances to miles (approximate)
    distances_miles = distances * 69
    
    # Calculate average distance to N nearest neighbors
    df_map['avg_distance_to_neighbors'] = np.mean(distances_miles[:, 1:], axis=1)
    
    # Identify isolated stores
    isolation_threshold = np.percentile(df_map['avg_distance_to_neighbors'], 90)
    df_map['is_isolated'] = df_map['avg_distance_to_neighbors'] > isolation_threshold
    
    return df_map

# Add this after the existing statistics section
st.markdown("### Geographic Analysis")

# Add separate cluster filter for geographic analysis
geo_analysis_clusters = st.multiselect(
    "Select Clusters for Geographic Analysis",
    options=available_clusters,
    default=available_clusters,
    key="geo_analysis_filter"  # Unique key to separate from main filter
)

# If no clusters are selected, show all clusters
if not geo_analysis_clusters:
    geo_analysis_clusters = available_clusters

# After loading the data and before creating the interface
# Initialize the dataframe with all metrics
@st.cache_data
def initialize_dataframe_with_metrics(df):
    """
    Initialize the dataframe with all required metrics.
    """
    df = df.copy()
    df = calculate_store_density(df)
    df = analyze_store_isolation(df)
    return df

# Initialize the full dataset with metrics
df_map = initialize_dataframe_with_metrics(df_map)

# Filter the pre-calculated metrics instead of recalculating
filtered_df_with_metrics = df_map[df_map["CLUSTERS"].isin(geo_analysis_clusters)]

# Add state analysis with the new filtered dataframe
add_state_analysis(filtered_df_with_metrics, cluster_colors)

# Add density analysis section
st.markdown("### Store Density Analysis")
col1, col2 = st.columns(2)

with col1:
    # Average stores within different radii by cluster
    radii = [10, 25, 50]
    cluster_density_data = []
    
    for cluster in filtered_df_with_metrics["CLUSTERS"].unique():
        cluster_data = filtered_df_with_metrics[filtered_df_with_metrics["CLUSTERS"] == cluster]
        for radius in radii:
            cluster_density_data.append({
                'Cluster': cluster,
                'Radius (miles)': radius,
                'Average Stores': cluster_data[f'stores_within_{radius}mi'].mean()
            })
    
    density_stats = pd.DataFrame(cluster_density_data)
    
    fig = px.bar(
        density_stats,
        x='Radius (miles)',
        y='Average Stores',
        color='Cluster',
        barmode='group',
        title="Average Number of Nearby Stores by Cluster",
        text='Average Stores',
        color_discrete_map=cluster_colors
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Radius (miles)",
        yaxis_title="Average Number of Stores",
        legend_title="Cluster",
        height=500  # Make the chart taller to accommodate the legend
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.info("""
    📊 **Density Analysis Explanation**
    - Shows average number of stores within different radii for each cluster
    - Higher values indicate denser store placement
    - Lower values indicate more dispersed stores
    """)

with col2:
    # Isolated stores analysis by cluster
    st.markdown("#### Isolation Analysis by Cluster")
    
    # Calculate isolation metrics for all stores, grouped by cluster
    isolation_by_cluster = df_map.groupby('CLUSTERS').agg({
        'is_isolated': 'sum',
        'avg_distance_to_neighbors': 'mean'
    }).round(1)
    
    isolation_by_cluster.columns = ['Isolated Stores', 'Avg Distance to 5 Nearest (mi)']
    
    # Add percentage of isolated stores
    cluster_sizes = df_map['CLUSTERS'].value_counts()
    isolation_by_cluster['% Isolated'] = (
        isolation_by_cluster['Isolated Stores'] / cluster_sizes * 100
    ).round(1)
    
    # Sort by percentage of isolated stores
    isolation_by_cluster = isolation_by_cluster.sort_values('% Isolated', ascending=False)
    
    # Display the metrics
    st.dataframe(
        isolation_by_cluster,
        use_container_width=True,
        column_config={
            "Isolated Stores": st.column_config.NumberColumn(
                "Isolated Stores",
                help="Number of stores identified as isolated in each cluster"
            ),
            "Avg Distance to 5 Nearest (mi)": st.column_config.NumberColumn(
                "Avg Distance to 5 Nearest (mi)",
                help="Average distance to the 5 nearest stores",
                format="%.1f"
            ),
            "% Isolated": st.column_config.NumberColumn(
                "% Isolated",
                help="Percentage of stores in the cluster that are isolated",
                format="%.1f%%"
            )
        }
    )
    
    # Add summary metrics for context
    st.markdown("#### Overall Isolation Metrics")
    total_isolated = df_map['is_isolated'].sum()
    total_stores = len(df_map)
    overall_avg_distance = df_map['avg_distance_to_neighbors'].mean()
    
    st.metric(
        "Total Isolated Stores", 
        f"{total_isolated} ({(total_isolated/total_stores*100):.1f}%)"
    )
    st.metric(
        "Overall Average Distance to 5 Nearest", 
        f"{overall_avg_distance:.1f} miles"
    )

# Add detailed store view with new metrics
st.markdown("### Store Details with Geographic Metrics")
if len(filtered_df_with_metrics) > 0:
    display_cols = [
        "UNIT_ID", "CLUSTERS", "STATE",
        "stores_within_10mi", "stores_within_25mi",
        "avg_distance_to_neighbors", "is_isolated"
    ]
    
    st.dataframe(
        filtered_df_with_metrics[display_cols],
        use_container_width=True,
        column_config={
            "UNIT_ID": "Store ID",
            "CLUSTERS": "Cluster",
            "stores_within_10mi": "Stores within 10mi",
            "stores_within_25mi": "Stores within 25mi",
            "avg_distance_to_neighbors": st.column_config.NumberColumn(
                "Avg Distance to Neighbors (mi)",
                format="%.1f"
            ),
            "is_isolated": "Is Isolated"
        }
    )

def create_cluster_comparison(df_map: pd.DataFrame, cluster_colors: dict) -> None:
    """
    Create an interactive cluster comparison tool.
    """
    st.markdown("### Cluster Comparison Tool")
    
    # Select clusters to compare
    col1, col2 = st.columns(2)
    with col1:
        cluster1 = st.selectbox(
            "Select First Cluster",
            options=sorted(df_map["CLUSTERS"].unique()),
            key="compare_cluster1"
        )
    with col2:
        cluster2 = st.selectbox(
            "Select Second Cluster",
            options=sorted(df_map["CLUSTERS"].unique()),
            key="compare_cluster2"
        )
    
    if cluster1 and cluster2:
        # Convert numpy integers to strings for dictionary lookup
        cluster1_str = str(int(cluster1))
        cluster2_str = str(int(cluster2))
        
        # Get data for each cluster
        cluster1_data = df_map[df_map["CLUSTERS"] == cluster1]
        cluster2_data = df_map[df_map["CLUSTERS"] == cluster2]
        
        # Create comparison metrics
        comparison = pd.DataFrame({
            'Metric': [
                'Number of Stores',
                'States Present',
                'Average Distance Between Stores (mi)',
                'Max Distance Between Stores (mi)',
                'Average Stores within 10mi',
                'Average Stores within 25mi',
                'Number of Isolated Stores',
                '% Isolated Stores',
                'Average Distance to 5 Nearest (mi)'
            ],
            f'Cluster {cluster1}': [
                len(cluster1_data),
                len(cluster1_data['STATE'].unique()),
                cluster1_data['avg_distance_to_neighbors'].mean(),
                cluster1_data['avg_distance_to_neighbors'].max(),
                cluster1_data['stores_within_10mi'].mean(),
                cluster1_data['stores_within_25mi'].mean(),
                cluster1_data['is_isolated'].sum(),
                (cluster1_data['is_isolated'].mean() * 100),
                cluster1_data['avg_distance_to_neighbors'].mean()
            ],
            f'Cluster {cluster2}': [
                len(cluster2_data),
                len(cluster2_data['STATE'].unique()),
                cluster2_data['avg_distance_to_neighbors'].mean(),
                cluster2_data['avg_distance_to_neighbors'].max(),
                cluster2_data['stores_within_10mi'].mean(),
                cluster2_data['stores_within_25mi'].mean(),
                cluster2_data['is_isolated'].sum(),
                (cluster2_data['is_isolated'].mean() * 100),
                cluster2_data['avg_distance_to_neighbors'].mean()
            ]
        })
        
        # Display comparison table
        st.dataframe(
            comparison.set_index('Metric').round(2),
            use_container_width=True
        )
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # State distribution comparison
            state_dist = pd.DataFrame({
                f'Cluster {cluster1}': cluster1_data['STATE'].value_counts(),
                f'Cluster {cluster2}': cluster2_data['STATE'].value_counts()
            }).fillna(0)
            
            fig1 = px.bar(
                state_dist,
                barmode='group',
                title=f"State Distribution Comparison",
                labels={'value': 'Number of Stores', 'index': 'State'},
                color_discrete_map={
                    f'Cluster {cluster1}': cluster_colors[cluster1_str],
                    f'Cluster {cluster2}': cluster_colors[cluster2_str]
                }
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Density comparison
            density_comparison = pd.DataFrame({
                'Radius (miles)': [10, 25, 50],
                f'Cluster {cluster1}': [
                    cluster1_data[f'stores_within_{r}mi'].mean()
                    for r in [10, 25, 50]
                ],
                f'Cluster {cluster2}': [
                    cluster2_data[f'stores_within_{r}mi'].mean()
                    for r in [10, 25, 50]
                ]
            })
            
            fig2 = px.line(
                density_comparison,
                x='Radius (miles)',
                y=[f'Cluster {cluster1}', f'Cluster {cluster2}'],
                title="Store Density Comparison",
                labels={'value': 'Average Nearby Stores'},
                color_discrete_map={
                    f'Cluster {cluster1}': cluster_colors[cluster1_str],
                    f'Cluster {cluster2}': cluster_colors[cluster2_str]
                }
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Add a map showing both clusters
        comparison_map = create_enhanced_map(
            pd.concat([cluster1_data, cluster2_data]),
            cluster_colors
        )
        st.markdown("#### Geographic Comparison")
        folium_static(comparison_map, width=1200, height=400)

# Add this at the end of the script, after the existing sections
create_cluster_comparison(df_map, cluster_colors)

# After loading the data, convert date columns to datetime
df['OPENING_DATE'] = pd.to_datetime(df['OPENING_DATE'])
df['LAST_REMODEL'] = pd.to_datetime(df['LAST_REMODEL'])

# Performance Overview
st.markdown("### Store Performance Analysis")
col1, col2 = st.columns(2)

with col1:
    # Revenue by Format boxplot with better formatting
    fig = px.box(df_map, 
                 x='STORE_FORMAT', 
                 y='ANNUAL_REVENUE',
                 title='Revenue Distribution by Store Format',
                 labels={'ANNUAL_REVENUE': 'Annual Revenue', 
                        'STORE_FORMAT': 'Store Format'})
    
    fig.update_layout(
        showlegend=False,
        yaxis=dict(
            title='Annual Revenue (Millions)',
            tickformat='$,.0f',
            # Convert to millions for cleaner display
            tickvals=np.arange(0, df_map['ANNUAL_REVENUE'].max(), 2000000),
            ticktext=[f'${x/1000000:.1f}M' for x in np.arange(0, df_map['ANNUAL_REVENUE'].max(), 2000000)]
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Revenue per sqft by region
    fig = px.scatter(df_map,
                     x='STORE_SIZE_SQFT',
                     y='REVENUE_PER_SQFT',
                     color='CLUSTERS',
                     title='Revenue Efficiency by Store Size',
                     labels={'STORE_SIZE_SQFT': 'Store Size (sq ft)', 
                            'REVENUE_PER_SQFT': 'Revenue per sq ft ($)'})
    st.plotly_chart(fig, use_container_width=True)

# Store Age Analysis
st.markdown("### Store Network Evolution")
timeline_fig = px.histogram(df_map,
                           x='OPENING_DATE',
                           color='STORE_FORMAT',
                           title='Store Opening Timeline',
                           labels={'OPENING_DATE': 'Opening Date',
                                  'count': 'Number of Stores'})
st.plotly_chart(timeline_fig, use_container_width=True)

# Market Analysis
st.markdown("### Market Demographics")
col1, col2 = st.columns(2)

with col1:
    income_fig = px.box(df_map,
                        x='CLUSTERS',
                        y='MEDIAN_HOUSEHOLD_INCOME',
                        title='Income Distribution by Cluster',
                        labels={'MEDIAN_HOUSEHOLD_INCOME': 'Median Household Income ($)',
                               'CLUSTERS': 'Cluster'})
    st.plotly_chart(income_fig, use_container_width=True)

with col2:
    competition_fig = px.histogram(df_map,
                                  x='COMPETITORS_5MI',
                                  color='CLUSTERS',
                                  title='Competition Density',
                                  labels={'COMPETITORS_5MI': 'Competitors within 5 miles',
                                         'count': 'Number of Stores'})
    st.plotly_chart(competition_fig, use_container_width=True)

# Calculate cluster metrics before the Operational Efficiency section
# Convert CLUSTERS to strings for consistent color mapping
df_map['CLUSTERS_STR'] = df_map['CLUSTERS'].astype(str)

# Calculate metrics by cluster
cluster_metrics = df_map.groupby('CLUSTERS').agg({
    'ANNUAL_REVENUE': 'mean',
    'EMPLOYEE_COUNT': 'mean',
    'REVENUE_PER_SQFT': 'mean',
    'CUSTOMER_SATISFACTION': 'mean',
    'STORE_SIZE_SQFT': 'mean'
}).round(2)

# Make cluster_metrics include the index for plotting
cluster_metrics = cluster_metrics.reset_index()
# Convert CLUSTERS to strings to match color mapping
cluster_metrics['CLUSTERS_STR'] = cluster_metrics['CLUSTERS'].astype(str)

# Operational Efficiency Section
st.markdown("### Operational Efficiency by Cluster")

# Create tabs for different operational metrics
op_tab1, op_tab2, op_tab3, op_tab4 = st.tabs([
    "Revenue Metrics",
    "Store Operations",
    "Customer Metrics",
    "Detailed Metrics"
])

with op_tab1:
    st.markdown("#### Revenue Analysis")
    
    # Create two columns for revenue charts
    rev_col1, rev_col2 = st.columns(2)
    
    with rev_col1:
        # Average Revenue by Cluster
        fig1 = px.bar(
            cluster_metrics,
            x='CLUSTERS_STR',
            y='ANNUAL_REVENUE',
            title='Average Revenue by Cluster',
            color='CLUSTERS_STR',
            color_discrete_map=cluster_colors,
            labels={'ANNUAL_REVENUE': 'Annual Revenue', 'CLUSTERS_STR': 'Cluster'}
        )
        fig1.update_layout(showlegend=False, yaxis_tickformat='$,.0f')
        st.plotly_chart(fig1, use_container_width=True)
    
    with rev_col2:
        # Revenue per Sq Ft by Cluster
        fig2 = px.bar(
            cluster_metrics,
            x='CLUSTERS_STR',
            y='REVENUE_PER_SQFT',
            title='Revenue per Sq Ft by Cluster',
            color='CLUSTERS_STR',
            color_discrete_map=cluster_colors,
            labels={'REVENUE_PER_SQFT': 'Revenue per Sq Ft', 'CLUSTERS_STR': 'Cluster'}
        )
        fig2.update_layout(showlegend=False, yaxis_tickformat='$,.2f')
        st.plotly_chart(fig2, use_container_width=True)

with op_tab2:
    st.markdown("#### Store Operations")
    
    # Create two columns for operations charts
    ops_col1, ops_col2 = st.columns(2)
    
    with ops_col1:
        # Average Employee Count by Cluster
        fig3 = px.bar(
            cluster_metrics,
            x='CLUSTERS_STR',
            y='EMPLOYEE_COUNT',
            title='Average Employee Count by Cluster',
            color='CLUSTERS_STR',
            color_discrete_map=cluster_colors,
            labels={'EMPLOYEE_COUNT': 'Employees', 'CLUSTERS_STR': 'Cluster'}
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with ops_col2:
        # Average Store Size by Cluster
        fig4 = px.bar(
            cluster_metrics,
            x='CLUSTERS_STR',
            y='STORE_SIZE_SQFT',
            title='Average Store Size by Cluster',
            color='CLUSTERS_STR',
            color_discrete_map=cluster_colors,
            labels={'STORE_SIZE_SQFT': 'Square Feet', 'CLUSTERS_STR': 'Cluster'}
        )
        fig4.update_layout(showlegend=False, yaxis_tickformat=',d')
        st.plotly_chart(fig4, use_container_width=True)

with op_tab3:
    st.markdown("#### Customer Satisfaction")
    
    # Customer Satisfaction Chart
    fig5 = px.bar(
        cluster_metrics,
        x='CLUSTERS_STR',
        y='CUSTOMER_SATISFACTION',
        title='Customer Satisfaction by Cluster (0-10 Scale)',
        color='CLUSTERS_STR',
        color_discrete_map=cluster_colors,
        labels={'CUSTOMER_SATISFACTION': 'Average Rating (0-10)', 'CLUSTERS_STR': 'Cluster'}
    )
    fig5.update_layout(
        showlegend=False,
        yaxis=dict(
            range=[0, 10],  # Updated range
            tickformat='.1f'  # Show one decimal place
        )
    )
    
    # Add reference lines for context
    fig5.add_hline(
        y=8.0,
        line_dash="dash",
        line_color="green",
        annotation_text="Excellent (8.0+)",
        annotation_position="right"
    )
    fig5.add_hline(
        y=7.0,
        line_dash="dash",
        line_color="orange",
        annotation_text="Average (7.0)",
        annotation_position="right"
    )
    fig5.add_hline(
        y=6.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Needs Improvement (<6.0)",
        annotation_position="right"
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Add explanation of the rating scale
    st.markdown("""
    **Customer Satisfaction Scale:**
    - 9-10: Outstanding service
    - 8-9: Excellent service
    - 7-8: Good service
    - 6-7: Fair service
    - <6: Needs improvement
    
    Ratings are based on customer surveys and feedback across multiple service dimensions.
    """)

with op_tab4:
    st.markdown("#### Detailed Cluster Metrics")
    
    # Display the detailed metrics table
    st.dataframe(
        cluster_metrics,
        use_container_width=True,
        column_config={
            "ANNUAL_REVENUE": st.column_config.NumberColumn(
                "Annual Revenue",
                help="Average annual revenue per store",
                format="$%d"
            ),
            "REVENUE_PER_SQFT": st.column_config.NumberColumn(
                "Revenue per Sq Ft",
                help="Average revenue per square foot",
                format="$%.2f"
            ),
            "CUSTOMER_SATISFACTION": st.column_config.NumberColumn(
                "Customer Satisfaction",
                help="Average customer satisfaction rating",
                format="%.2f"
            ),
            "EMPLOYEE_COUNT": st.column_config.NumberColumn(
                "Employees",
                help="Average number of employees per store",
                format="%d"
            ),
            "STORE_SIZE_SQFT": st.column_config.NumberColumn(
                "Store Size (sq ft)",
                help="Average store size in square feet",
                format="%d"
            )
        }
    )

# Add after the existing statistics section
st.markdown("### Market Opportunity Analysis")

# Create tabs for different analyses
market_tab1, market_tab2, market_tab3 = st.tabs([
    "Population Density", 
    "Competition Analysis",
    "Revenue Potential"
])

with market_tab1:
    st.markdown("#### Population vs Store Coverage")
    
    # Create two columns - map on left, explanation on right
    map_col1, guide_col1 = st.columns([2, 1])
    
    with map_col1:
        # Create population density heatmap
        population_map = folium.Map(
            location=[df_map["LAT"].mean(), df_map["LNG"].mean()],
            zoom_start=4,
            tiles='CartoDB positron'
        )
        
        # Prepare population data for heatmap
        valid_data = df_map.dropna(subset=['LAT', 'LNG', 'POPULATION_5MI'])
        
        # Add store markers FIRST (before heatmap)
        for _, row in valid_data.iterrows():
            try:
                cluster_str = str(int(row['CLUSTERS']))
                folium.CircleMarker(
                    location=[float(row['LAT']), float(row['LNG'])],
                    radius=8,  # Increased size
                    color='black',
                    weight=2,
                    fill=True,
                    fill_color=cluster_colors[cluster_str],
                    fill_opacity=1.0,
                    popup=f"""
                        <div style='font-family: Arial; width: 150px;'>
                            <b>Store ID:</b> {row['UNIT_ID']}<br>
                            <b>Cluster:</b> {row['CLUSTERS']}<br>
                            <b>Population:</b> {int(row['POPULATION_5MI']):,}
                        </div>
                    """
                ).add_to(population_map)
            except (ValueError, TypeError) as e:
                st.write(f"Error with row {row['UNIT_ID']}: {str(e)}")
                continue
        
        # Create the heatmap data
        population_data = []
        for _, row in valid_data.iterrows():
            try:
                lat = float(row['LAT'])
                lng = float(row['LNG'])
                weight = float(row['POPULATION_5MI']) / float(valid_data['POPULATION_5MI'].max())
                if not (pd.isna(lat) or pd.isna(lng) or pd.isna(weight)):
                    population_data.append([lat, lng, weight])
            except (ValueError, TypeError):
                continue
        
        # Add heatmap layer with reduced opacity
        if population_data:
            folium.plugins.HeatMap(
                data=population_data,
                name='Population Density',
                min_opacity=0.3,  # Reduced opacity
                max_zoom=18,
                radius=20,
                blur=15,
                max_val=1.0
            ).add_to(population_map)
        
        # Add layer control
        folium.LayerControl().add_to(population_map)
        
        # Display the map
        folium_static(population_map, width=800, height=500)
        
        # Add cluster-specific population metrics below map
        st.markdown("##### Population by Cluster")
        cluster_pop_metrics = df_map.groupby('CLUSTERS').agg({
            'POPULATION_5MI': ['mean', 'max', 'count']
        }).round(0)
        
        cluster_pop_metrics.columns = ['Avg Population', 'Max Population', 'Store Count']
        # Convert to integers before displaying
        cluster_pop_metrics['Avg Population'] = cluster_pop_metrics['Avg Population'].astype(int)
        cluster_pop_metrics['Max Population'] = cluster_pop_metrics['Max Population'].astype(int)
        
        st.dataframe(
            cluster_pop_metrics,
            use_container_width=True,
            column_config={
                "Avg Population": st.column_config.NumberColumn(format="%d"),
                "Max Population": st.column_config.NumberColumn(format="%d"),
                "Store Count": st.column_config.NumberColumn(format="%d")
            }
        )
    
    with guide_col1:
        st.markdown("""
        **How to Read This Map:**
        - Heatmap colors show population density
        - Darker red = Higher population
        - Lighter blue = Lower population
        - Colored dots = Store locations
        - Dot colors indicate cluster membership
        
        **Store Markers:**
        """)
        # Add cluster color legend
        for cluster, color in cluster_colors.items():
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="width: 15px; height: 15px; background-color: {color}; '
                f'margin-right: 10px; border-radius: 50%; border: 1px solid black;"></div>'
                f'<span>Cluster {cluster}</span></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("""
        **Use This Map To:**
        - Compare population coverage across clusters
        - Identify potential expansion areas
        - Spot gaps in market coverage
        - Analyze cluster distribution
        """)

# Similar structure for Competition Analysis
with market_tab2:
    st.markdown("#### Competition Analysis")
    
    # Create three columns for the explanation
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        **How to Read This Map:**
        - Circle size = Number of competitors
        - Circle color = Cluster membership
        - Hover for detailed information
        """)
    
    with guide_col2:
        st.markdown("""
        **Competition Patterns:**
        - Larger circles = More competitors
        - Smaller circles = Fewer competitors
        - Clustering shows market saturation
        """)
    
    with guide_col3:
        st.markdown("""
        **Analysis by Cluster:**
        - Which clusters face more competition?
        - How does revenue correlate with competition?
        - Are certain clusters in more competitive markets?
        """)
    
    # Add a small space between guide and visualizations
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Adjust column widths - make them wider
    map_col2, metrics_col2 = st.columns([3, 1.5])  # Changed from [2, 1] to [3, 1.5]
    
    with map_col2:
        # Competition visualization
        competition_fig = px.scatter_mapbox(
            df_map,
            lat='LAT',
            lon='LNG',
            size='COMPETITORS_5MI',
            color='CLUSTERS_STR',
            hover_data={
                'CLUSTERS_STR': True,
                'CITY': True,
                'STATE': True,
                'COMPETITORS_5MI': True,
                'ANNUAL_REVENUE': ':$,.0f',
                'UNIT_ID': True
            },
            color_discrete_map=cluster_colors,
            size_max=25,
            zoom=3,
            title='Competition Analysis by Cluster',
            mapbox_style="carto-positron"
        )
        
        # Update layout with larger width
        competition_fig.update_layout(
            mapbox=dict(
                center=dict(lat=df_map["LAT"].mean(), lon=df_map["LNG"].mean()),
                zoom=3
            ),
            showlegend=True,
            legend_title="Clusters",
            legend=dict(
                itemsizing='constant',
                title="Clusters",
                font=dict(size=12)
            ),
            height=600,  # Keep the height
            width=800   # Add explicit width
        )
        
        # Update traces
        for cluster in sorted(df_map['CLUSTERS_STR'].unique()):
            competition_fig.for_each_trace(
                lambda t: t.update(
                    marker=dict(
                        color=cluster_colors[cluster]
                    )
                ) if t.name == f"Cluster {cluster}" else t
            )
        
        # Use full container width for the plot
        st.plotly_chart(competition_fig, use_container_width=True)
    
    with metrics_col2:
        # Add some padding around the metrics table
        st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
        
        st.markdown("##### Competition Metrics by Cluster")
        cluster_comp_metrics = df_map.groupby('CLUSTERS').agg({
            'COMPETITORS_5MI': ['mean', 'max'],
            'ANNUAL_REVENUE': 'mean'
        }).round(2)
        
        cluster_comp_metrics.columns = ['Avg Competitors', 'Max Competitors', 'Avg Revenue']
        st.dataframe(
            cluster_comp_metrics,
            use_container_width=True,
            column_config={
                "Avg Competitors": st.column_config.NumberColumn(format="%.1f"),
                "Max Competitors": st.column_config.NumberColumn(format="%.0f"),
                "Avg Revenue": st.column_config.NumberColumn(format="$%d")
            }
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Similar structure for Revenue Potential
with market_tab3:
    st.markdown("#### Revenue Potential by Cluster")
    
    map_col3, guide_col3 = st.columns([2, 1])
    
    with map_col3:
        # Revenue visualization
        cluster_revenue_fig = px.scatter(
            df_map,
            x='POPULATION_5MI',
            y='ANNUAL_REVENUE',
            color='CLUSTERS_STR',  # Use string version of clusters
            size='COMPETITORS_5MI',
            color_discrete_map=cluster_colors,
            title='Revenue vs Population by Cluster',
            labels={
                'POPULATION_5MI': 'Population within 5 miles',
                'ANNUAL_REVENUE': 'Annual Revenue ($)',
                'COMPETITORS_5MI': 'Number of Competitors',
                'CLUSTERS_STR': 'Cluster'
            },
            hover_data={
                'CLUSTERS_STR': True,
                'POPULATION_5MI': ':,',
                'ANNUAL_REVENUE': ':$,.0f',
                'COMPETITORS_5MI': True,
                'UNIT_ID': True
            }
        )
        
        # Update layout for better readability and consistent colors
        cluster_revenue_fig.update_layout(
            xaxis_title="Population within 5 miles",
            yaxis_title="Annual Revenue ($)",
            yaxis_tickformat='$,.0f',
            xaxis_tickformat=',',
            legend_title="Clusters",
            showlegend=True,
            # Update legend to show correct colors
            legend=dict(
                itemsizing='constant',
                title="Clusters",
                font=dict(size=12)
            )
        )
        
        # Remove the previous color forcing code and instead update traces individually
        for cluster in sorted(df_map['CLUSTERS_STR'].unique()):
            cluster_revenue_fig.for_each_trace(
                lambda t: t.update(
                    marker=dict(
                        color=cluster_colors[cluster]
                    )
                ) if t.name == f"Cluster {cluster}" else t
            )
        
        st.plotly_chart(cluster_revenue_fig, use_container_width=True)
        
        # Add cluster revenue metrics
        st.markdown("##### Revenue Metrics by Cluster")
        cluster_rev_metrics = df_map.groupby('CLUSTERS').agg({
            'ANNUAL_REVENUE': ['mean', 'median', 'std'],
            'REVENUE_PER_SQFT': 'mean'
        }).round(2)
        
        cluster_rev_metrics.columns = ['Avg Revenue', 'Median Revenue', 'Revenue Std Dev', 'Revenue/SqFt']
        st.dataframe(
            cluster_rev_metrics,
            use_container_width=True,
            column_config={
                "Avg Revenue": st.column_config.NumberColumn(format="$%d"),
                "Median Revenue": st.column_config.NumberColumn(format="$%d"),
                "Revenue Std Dev": st.column_config.NumberColumn(format="$%d"),
                "Revenue/SqFt": st.column_config.NumberColumn(format="$%.2f")
            }
        )
    
    with guide_col3:
        st.markdown("""
        **How to Read This Analysis:**
        - X-axis: Local population
        - Y-axis: Annual revenue
        - Point color: Cluster membership
        - Point size: Number of competitors
        
        **Cluster Performance:**
        Compare clusters based on:
        - Revenue generation
        - Population served
        - Competition levels
        - Revenue per square foot
        
        **Key Insights:**
        - Revenue spread within clusters
        - Population-revenue relationship
        - Competitive dynamics
        - Market optimization opportunities
        """)

# Now let's add the Predictive Analytics section
st.markdown("### Predictive Analytics")

pred_tab1, pred_tab2 = st.tabs(["Revenue Predictor", "Performance Insights"])

with pred_tab1:
    st.markdown("#### Store Revenue Predictor")
    
    # Prepare the data for modeling with named features
    feature_names = ['STORE_SIZE_SQFT', 'POPULATION_5MI', 'COMPETITORS_5MI', 'MEDIAN_HOUSEHOLD_INCOME']
    X = pd.DataFrame(df_map[feature_names])  # Create DataFrame with named features
    y = df_map['ANNUAL_REVENUE']
    
    # Scale the features while preserving feature names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names,
        index=X.index
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Create prediction interface
    col1, col2 = st.columns(2)
    
    with col1:
        store_size = st.number_input(
            "Store Size (sq ft)",
            min_value=2500,
            max_value=10000,
            value=5000,
            step=500
        )
        
        population = st.number_input(
            "Population within 5mi",
            min_value=0,
            value=100000,
            step=10000
        )
    
    with col2:
        competitors = st.number_input(
            "Number of Competitors",
            min_value=0,
            value=3,
            step=1
        )
        
        income = st.number_input(
            "Median Household Income",
            min_value=0,
            value=65000,
            step=5000
        )
    
    if st.button("Predict Revenue"):
        # Prepare input for prediction
        input_data = np.array([[store_size, population, competitors, income]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction with confidence interval
        st.success(f"Predicted Annual Revenue: ${prediction:,.2f}")
        
        # Show feature importance
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
        
        st.markdown("#### Feature Importance")
        fig = px.bar(
            importance,
            x='Feature',
            y='Importance',
            title='Relative Importance of Factors'
        )
        st.plotly_chart(fig, use_container_width=True)

with pred_tab2:
    st.markdown("#### Performance Analysis")
    
    # Calculate actual vs predicted values using named features
    y_pred = model.predict(X_scaled)  # X_scaled now has feature names
    df_map['Predicted_Revenue'] = y_pred
    df_map['Revenue_Difference'] = df_map['ANNUAL_REVENUE'] - df_map['Predicted_Revenue']
    
    # Create performance visualization
    fig = px.scatter(
        df_map,
        x='ANNUAL_REVENUE',
        y='Predicted_Revenue',
        color='CLUSTERS',
        title='Actual vs Predicted Revenue',
        labels={
            'ANNUAL_REVENUE': 'Actual Revenue',
            'Predicted_Revenue': 'Predicted Revenue'
        },
        color_discrete_map=cluster_colors
    )
    
    # Add 45-degree line
    fig.add_trace(
        go.Scatter(
            x=[df_map['ANNUAL_REVENUE'].min(), df_map['ANNUAL_REVENUE'].max()],
            y=[df_map['ANNUAL_REVENUE'].min(), df_map['ANNUAL_REVENUE'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='gray')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show stores performing above/below expectations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Overperforming Stores")
        overperformers = df_map.nlargest(5, 'Revenue_Difference')[
            ['UNIT_ID', 'CITY', 'STATE', 'Revenue_Difference']
        ]
        st.dataframe(
            overperformers,
            column_config={
                "Revenue_Difference": st.column_config.NumberColumn(
                    "Revenue Above Expected",
                    format="$%.0f"
                )
            }
        )
    
    with col2:
        st.markdown("#### Top Underperforming Stores")
        underperformers = df_map.nsmallest(5, 'Revenue_Difference')[
            ['UNIT_ID', 'CITY', 'STATE', 'Revenue_Difference']
        ]
        st.dataframe(
            underperformers,
            column_config={
                "Revenue_Difference": st.column_config.NumberColumn(
                    "Revenue Below Expected",
                    format="$%.0f"
                )
            }
        )

