# 🏪 Store Location Analysis Dashboard

An interactive Streamlit dashboard for analyzing store locations, across clusters, and performance metrics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://storegeodash.streamlit.app/)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🚀 Setup

1. 📥 Clone the repository: git clone [repository-url]
2. 🔧 Create and activate a virtual environment: python -m venv venv
3. 📦 Install dependencies: pip install -r requirements.txt
4. 📂 Run the Streamlit app: streamlit run mapViz.py


## 📁 Files
- 📊 `mapViz.py`: Main application file containing the Streamlit dashboard
- 📈 `updated_store_location_v3.csv`: Store location dataset
- 📝 `requirements.txt`: Project dependencies

## ✨ Features
- 🗺️ Interactive store location map with cluster visualization
- 📊 Operational efficiency analysis by cluster
- 📈 Market opportunity analysis including:
  - 🌡️ Population density heatmap
  - 🎯 Competition analysis
  - 💰 Revenue potential assessment
- 📉 Store performance metrics and predictive analytics
- 🔍 Detailed cluster comparison tools
- 🌎 Geographic analysis with isolation metrics
- 🏢 State-level breakdowns and demographics

## 📋 Dashboard Sections
1. 🗺️ **Store Location Map**: Interactive map showing store locations color-coded by cluster
2. 📊 **Cluster Statistics**: Overview of store distribution and geographic spread
3. 💼 **Operational Efficiency**: Analysis of revenue, store operations, and customer satisfaction
4. 📈 **Market Analysis**: Demographics and competition analysis
5. 🤖 **Predictive Analytics**: Revenue prediction and performance insights

## 💾 Data
The dashboard uses store location data including:
- 📍 Geographic coordinates
- 🎯 Cluster assignments
- 💰 Revenue metrics
- 🏪 Store characteristics
- 👥 Market demographics
- 🔄 Competition data

## 📝 Notes
- 🔄 The map supports multiple visualization layers (individual stores, clusters, heatmap)
- 🖱️ All visualizations are interactive with hover details
- 🔍 Data can be filtered by cluster or store ID
- 📤 Export functionality available for filtered data and analysis results

## 🛠️ Built With
- [Streamlit](https://streamlit.io/) - The web framework used
- [Folium](https://python-visualization.github.io/folium/) - For interactive maps
- [Plotly](https://plotly.com/) - For interactive visualizations
- [Pandas](https://pandas.pydata.org/) - For data manipulation
- [Scikit-learn](https://scikit-learn.org/) - For predictive analytics

## 📫 Support
For support, please open an issue in the repository.
