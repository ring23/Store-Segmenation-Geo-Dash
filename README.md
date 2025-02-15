# Store Location Analysis Dashboard

An interactive Streamlit dashboard for analyzing store locations, across clusters, and performance metrics.

See it live here: https://storegeodash.streamlit.app/ :)

## Setup

1. Clone the repository:
git clone https://github.com/ring23/Store-Segmenation-Geo-Dash.git

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the application:
streamlit run mapViz.py

## Files
- `mapViz.py`: Main application file containing the Streamlit dashboard
- `updated_store_location_v3.csv`: Store location dataset
- `requirements.txt`: Project dependencies

## Features
- Interactive store location map with cluster visualization
- Operational efficiency analysis by cluster
- Market opportunity analysis including:
  - Population density heatmap
  - Competition analysis
  - Revenue potential assessment
- Store performance metrics and predictive analytics
- Detailed cluster comparison tools
- Geographic analysis with isolation metrics
- State-level breakdowns and demographics

## Dashboard Sections
1. **Store Location Map**: Interactive map showing store locations color-coded by cluster
2. **Cluster Statistics**: Overview of store distribution and geographic spread
3. **Operational Efficiency**: Analysis of revenue, store operations, and customer satisfaction
4. **Market Analysis**: Demographics and competition analysis
5. **Predictive Analytics**: Revenue prediction and performance insights

## Data
The dashboard uses store location data including:
- Geographic coordinates
- Cluster assignments
- Revenue metrics
- Store characteristics
- Market demographics
- Competition data

## Notes
- The map supports multiple visualization layers (individual stores, clusters, heatmap)
- All visualizations are interactive with hover details
- Data can be filtered by cluster or store ID
- Export functionality available for filtered data and analysis results 
