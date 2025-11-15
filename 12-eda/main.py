# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go 
# from datetime import datetime
# import os

# # Set page configuration
# st.set_page_config(
#     page_title="Pakistan Property Market Analysis",
#     page_icon="🏠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .section-header {
#         font-size: 1.5rem;
#         color: #1f77b4;
#         margin-top: 2rem;
#         margin-bottom: 1rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Title and description
# st.markdown('<h1 class="main-header">🏠 Pakistan Property Market Analysis</h1>', unsafe_allow_html=True)
# st.markdown("""
# This dashboard provides insights into the Pakistani property market based on data from Zameen.com.
# Explore property trends, prices, and distributions across different cities and property types.
# """)

# # Detect the directory where the script is running
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Correct filename (no folder prefix)
# data_file_path = os.path.join(current_dir, "zameen-property.csv")


# if not os.path.exists(data_file_path):
#     st.error(f"File NOT found: {data_file_path}")
#     st.write("📂 Files in this directory:", os.listdir(current_dir))
#     st.stop()

# # Load data with all transformations from the notebook
# @st.cache_data
# def load_data(file_path=data_file_path):
#     # Load the actual dataset
#     try:
#         df = pd.read_csv(file_path, sep=';')
#     except FileNotFoundError:
#         st.error(f"Error: The file '{file_path}' was not found. Please ensure the file is present.")
#         return pd.DataFrame()
    
#     # Data cleaning and selection steps from the notebook
#     df['agency'] = df['agency'].fillna('Unknown')
#     df['agent'] = df['agent'].fillna('Unknown')
    
#     # Select relevant columns (as in the notebook)
#     relevant_columns = [
#         'property_type', 'price', 'location', 'city', 'province_name', 
#         'latitude', 'longitude', 'baths', 'bedrooms', 'date_added', 
#         'area', 'purpose', 'agency', 'agent'
#     ]
#     df_cleaned = df[relevant_columns].copy()
    
#     # Area transformation (as in the notebook)
#     df_cleaned['area_value'] = df_cleaned['area'].str.extract(r'(\d+(\.\d+)?)', expand=False)[0].astype(float)
#     df_cleaned['area_unit'] = df_cleaned['area'].str.extract(r'([Kk]anal|[Mm]arla)', expand=False)
#     df_cleaned['area_unit'] = df_cleaned['area_unit'].str.lower()
    
#     df_cleaned['area_in_marla'] = df_cleaned.apply(
#         lambda row: row['area_value'] * 20 if row['area_unit'] == 'kanal' else row['area_value'],
#         axis=1
#     )
    
#     # Select final columns (as in the notebook)
#     cleaned_columns = [
#         'property_type', 'price', 'location', 'city', 'province_name',
#         'latitude', 'longitude', 'baths', 'bedrooms', 'date_added',
#         'purpose', 'agency', 'agent', 'area_in_marla'
#     ]
    
#     final_df = df_cleaned[cleaned_columns].copy()
    
#     # Convert date_added to datetime and extract year
#     final_df['date_added'] = pd.to_datetime(final_df['date_added'], errors='coerce', format='%m-%d-%Y')
#     final_df['year_added'] = final_df['date_added'].dt.year
#     final_df['month_added'] = final_df['date_added'].dt.month
    
#     # Basic outlier removal for visual cleanliness (mimicking notebook analysis)
#     final_df = final_df[
#         (final_df['baths'] < 100) & 
#         (final_df['bedrooms'] < 20) & 
#         (final_df['price'] > 1000) & 
#         (final_df['year_added'].isin([2018, 2019]))
#     ]
    
#     return final_df

# # Load the data
# try:
#     df = load_data()
#     st.sidebar.success("✅ Data loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading data: {e}")
#     st.stop()

# # --- 3. Sidebar and Filtering ---

# st.sidebar.header("Filters")

# # --- FIXED Q1/Q10 THRESHOLD (Scalar) ---
# # This value is used for Q1 and Q10 analysis ONLY.
# Q1_HIGH_VALUE_THRESHOLD = 50000000 
# st.sidebar.markdown(f"**Q1/Q10 High-Value Threshold:** PKR {Q1_HIGH_VALUE_THRESHOLD:,.0f}")

# # Price range filter (Tuple)
# min_price, max_price = int(df['price'].min()), int(df['price'].max())
# price_range_tuple = st.sidebar.slider(
#     "Price Range (Overall Filter)",
#     min_value=min_price,
#     max_value=max_price,
#     value=(min_price, max_price)
# )

# # City filter
# cities = sorted(list(df['city'].unique()))
# selected_cities = st.sidebar.multiselect(
#     'Select Cities for Comparison',
#     options=cities,
#     default=cities
# )

# # Property type filter
# property_types = sorted(list(df['property_type'].unique()))
# selected_property_type = st.sidebar.selectbox("Select Property Type", ['All'] + property_types)

# # Purpose filter
# purposes = sorted(list(df['purpose'].unique()))
# selected_purpose = st.sidebar.selectbox("Select Purpose", ['All'] + purposes)

# # Area range filter (Marla)
# min_area, max_area = int(df['area_in_marla'].min()), int(df['area_in_marla'].max())
# area_range = st.sidebar.slider(
#     "Area Range (Marla)",
#     min_value=min_area,
#     max_value=max_area,
#     value=(min_area, max_area)
# )

# # --- APPLY FILTERS TO CREATE filtered_df ---
# filtered_df = df.copy()

# if selected_cities:
#     filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

# if selected_property_type != 'All':
#     filtered_df = filtered_df[filtered_df['property_type'] == selected_property_type]

# if selected_purpose != 'All':
#     filtered_df = filtered_df[filtered_df['purpose'] == selected_purpose]

# # Apply the price and area range tuples
# filtered_df = filtered_df[
#     (filtered_df['price'] >= price_range_tuple[0]) &  
#     (filtered_df['price'] <= price_range_tuple[1]) &
#     (filtered_df['area_in_marla'] >= area_range[0]) & 
#     (filtered_df['area_in_marla'] <= area_range[1])
# ]

# # --- 4. Dashboard Content ---

# # Display key metrics
# st.markdown('<h2 class="section-header">📊 Key Metrics</h2>', unsafe_allow_html=True)

# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric("Total Properties", len(filtered_df))

# with col2:
#     avg_price = filtered_df['price'].mean() if not filtered_df.empty else 0
#     st.metric("Average Price", f"₹{avg_price:,.0f}")

# with col3:
#     avg_area = filtered_df['area_in_marla'].mean() if not filtered_df.empty else 0
#     st.metric("Average Area (Marla)", f"{avg_area:.1f}")

# with col4:
#     most_common_city = filtered_df['city'].mode()[0] if len(filtered_df) > 0 else "N/A"
#     st.metric("Most Common City", most_common_city)

# st.markdown("---")


# # Data Table
# st.markdown('<h2 class="section-header">📋 Property Data</h2>', unsafe_allow_html=True)

# # Show a sample of the data
# if not filtered_df.empty:
#     st.dataframe(filtered_df.head(100), use_container_width=True)

#     # Download button for filtered data
#     csv = filtered_df.to_csv(index=False)
#     st.download_button(
#         label="Download Filtered Data as CSV",
#         data=csv,
#         file_name="filtered_property_data.csv",
#         mime="text/csv"
#     )
    
#     # Show data summary
#     with st.expander("📊 Data Summary"):
#         st.write("**Dataset Overview:**")
#         st.write(f"- Total properties in dataset: {len(df)}")
#         st.write(f"- Date range: {df['date_added'].min().strftime('%Y-%m-%d')} to {df['date_added'].max().strftime('%Y-%m-%d')}")
#         st.write(f"- Cities covered: {', '.join(sorted(df['city'].unique()))}")
#         st.write(f"- Property types: {', '.join(sorted(df['property_type'].unique()))}")
        
#         st.write("**Price Statistics:**")
#         st.write(f"- Minimum price: {df['price'].min():,}")
#         st.write(f"- Maximum price: {df['price'].max():,}")
#         st.write(f"- Average price: {df['price'].mean():,.0f}")
        
# else:
#     st.info("No data available for the selected filters")


# # --- Q9: Property Distribution (Sale vs Rent) ---
# st.markdown('<h2 class="section-header">Distribution of Property Purposes</h2>', unsafe_allow_html=True)

# if not filtered_df.empty:
#     purpose_distribution = filtered_df.groupby(['city', 'purpose']).size().reset_index(name='count')
    
#     fig_q9 = px.bar(
#         purpose_distribution, 
#         x='city', 
#         y='count', 
#         color='purpose',
#         title='Sale vs. Rent Listings by City'
#     )
#     st.plotly_chart(fig_q9, use_container_width=True)
#     st.caption("Most listings are typically 'For Sale'.")
# else:
#     st.info("No data available for the selected filters.")

# st.markdown("---")


# # --- Price Analysis Section ---
# st.markdown('<h2 class="section-header">💰 Price Analysis & Distribution</h2>', unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     # Average price by city
#     if not filtered_df.empty:
#         avg_price_by_city = filtered_df.groupby('city')['price'].mean().sort_values(ascending=False)
#         fig3 = px.bar(
#             x=avg_price_by_city.index,
#             y=avg_price_by_city.values,
#             title="Average Price by City",
#             labels={'x': 'City', 'y': 'Average Price'},
#             color=avg_price_by_city.index
#         )
#         st.plotly_chart(fig3, use_container_width=True)
#     else:
#         st.info("No data available for the selected filters")

# with col2:
#     # Average price by property type
#     if not filtered_df.empty:
#         avg_price_by_type = filtered_df.groupby('property_type')['price'].mean().sort_values(ascending=False)
#         fig4 = px.bar(
#             x=avg_price_by_type.index,
#             y=avg_price_by_type.values,
#             title="Average Price by Property Type",
#             labels={'x': 'Property Type', 'y': 'Average Price'},
#             color=avg_price_by_type.index
#         )
#         st.plotly_chart(fig4, use_container_width=True)
#     else:
#         st.info("No data available for the selected filters")


# # --- 5. Detailed Business Insights (Q1, Q3-Q8, Q10) ---

# st.markdown('<h2 class="section-header">📈 Detailed Business Insights</h2>', unsafe_allow_html=True)

# # Helper function to compute Top N listings (for Q4, Q10)
# def get_top_n_listings(df, group_col, n=10):
#     listing_counts = df.groupby([group_col, 'city']).size().reset_index(name='listing_count')
#     listing_counts = listing_counts[listing_counts[group_col] != 'Unknown']
#     top_n_groups = listing_counts.groupby(group_col)['listing_count'].sum().nlargest(n).index
#     return listing_counts[listing_counts[group_col].isin(top_n_groups)]

# # Row 1: Q1 and Q5
# col_q1, col_q5 = st.columns(2)

# with col_q1:
#     st.subheader("High-Value Property Growth (YoY)")
#     # Use the static Q1_HIGH_VALUE_THRESHOLD here
#     high_value_data = df[df['price'] > Q1_HIGH_VALUE_THRESHOLD]
#     high_value_growth = high_value_data.groupby(['city', 'year_added']).size().reset_index(name='property_count')
    
#     if not high_value_growth.empty and len(high_value_growth['year_added'].unique()) > 1:
#         high_value_growth['growth'] = high_value_growth.groupby('city')['property_count'].diff()
        
#         fig_q1 = px.line(
#             high_value_growth.dropna(), 
#             x='year_added', 
#             y='property_count', 
#             color='city',
#             markers=True,
#             title=f'Q1: High-Value Listings (> {Q1_HIGH_VALUE_THRESHOLD:,.0f} PKR) Growth'
#         )
#         fig_q1.update_layout(xaxis={'tickmode': 'linear', 'dtick': 1})
#         st.plotly_chart(fig_q1, use_container_width=True)
#         st.caption("Tracks the change in luxury property listings year-over-year.")
#     else:
#         st.info("Q1: Insufficient data for YoY high-value growth analysis.")

# with col_q5:
#     st.subheader("Seasonal Trend in Property Listings")
#     seasonal_trend = filtered_df.groupby(['year_added', 'month_added']).size().reset_index(name='listing_count')
    
#     fig_q5 = px.line(
#         seasonal_trend, 
#         x='month_added', 
#         y='listing_count', 
#         color='year_added',
#         markers=True,
#         title='Q5: Seasonal Trend by Month (2018-2019)'
#     )
#     fig_q5.update_layout(xaxis={'tickmode': 'linear', 'dtick': 1})
#     st.plotly_chart(fig_q5, use_container_width=True)
#     st.caption("Listing volume usually dips in mid-year (June/July).")

# # Row 2: Q7 and Q6
# col_q7, col_q6 = st.columns(2)

# with col_q7:
#     st.subheader("Property Type Prevalence by Province")
#     province_property_distribution = df.groupby(['province_name', 'property_type']).size().reset_index(name='count')
    
#     fig_q7 = px.bar(
#         province_property_distribution, 
#         x='province_name', 
#         y='count', 
#         color='property_type',
#         title='Q7: Property Type Prevalence by Province'
#     )
#     st.plotly_chart(fig_q7, use_container_width=True)
#     st.caption("Houses dominate Punjab, while Flats are more prevalent in Sindh.")

# with col_q6:
#     st.subheader("Area-Price Correlation Table")
    
#     correlation_df = filtered_df.groupby(['city', 'property_type']).apply(
#         lambda df: df['area_in_marla'].corr(df['price'])
#     ).reset_index(name='correlation')

#     correlation_df['correlation_strength'] = correlation_df['correlation'].apply(
#         lambda x: 'Highly Correlated' if abs(x) >= 0.5 else 'Less Correlated'
#     )
    
#     st.dataframe(correlation_df.sort_values('correlation', ascending=False).fillna('N/A'))
#     st.caption("""
#         **Insight:** Area is **highly correlated** (>= 0.5) with price for larger properties, but generally weak for Flats/Portions.
#     """)

# # Row 3: Q8 and Q3
# col_q8, col_q3 = st.columns(2)

# with col_q8:
#     st.subheader("Influence of Number of Bathrooms on Price")
#     df_baths_filtered_q8 = filtered_df[filtered_df['baths'] < 16]
#     bathroom_price_influence = df_baths_filtered_q8.groupby(['city', 'baths'])['price'].mean().reset_index()

#     fig_q8 = px.line(
#         bathroom_price_influence, 
#         x='baths', 
#         y='price', 
#         color='city',
#         markers=True,
#         title='Q8: Average Price vs. Number of Bathrooms'
#     )
#     st.plotly_chart(fig_q8, use_container_width=True)
#     st.caption("Price generally increases with the number of bathrooms.")

# with col_q3:
#     st.subheader("Price vs. Bedrooms across Types")
#     price_bedroom_variation = filtered_df[filtered_df['bedrooms'] < 15].groupby(['city', 'property_type', 'bedrooms'])['price'].mean().reset_index()
    
#     fig_q3 = px.line(
#         price_bedroom_variation,
#         x='bedrooms',
#         y='price',
#         color='property_type',
#         line_dash='city',
#         title='Q3: Avg Price vs. Bedrooms'
#     )
#     st.plotly_chart(fig_q3, use_container_width=True)
#     st.caption("Larger property types see higher prices with more bedrooms; others plateau.")

# # Row 4: Q4 and Q10 
# col_q4, col_q10 = st.columns(2)

# with col_q4:
#     st.subheader("Top Agencies (High-Demand Areas)")
#     high_demand_areas = df[df['city'].isin(['Karachi', 'Lahore', 'Rawalpindi', 'Islamabad', 'Faisalabad'])]
#     agency_listings_top_n = get_top_n_listings(high_demand_areas, 'agency', 10)
    
#     if not agency_listings_top_n.empty:
#         fig_q4 = px.bar(
#             agency_listings_top_n.sort_values(by=['listing_count'], ascending=False), 
#             x='listing_count', 
#             y='agency', 
#             color='city',
#             orientation='h',
#             title='Q4: Top 10 Agencies by Total Listings',
#             height=400
#         )
#         st.plotly_chart(fig_q4, use_container_width=True)
#         st.caption("Activity is heavily concentrated in major metropolitan centers.")
#     else:
#         st.info("Q4: No agency data available for the analysis.")


# with col_q10:
#     st.subheader("Top Agents in High-Value Markets")
#     # Use the static Q1_HIGH_VALUE_THRESHOLD here
#     high_value_data_q10 = filtered_df[filtered_df['price'] > Q1_HIGH_VALUE_THRESHOLD]
#     active_agents_top_n = get_top_n_listings(high_value_data_q10, 'agent', 10)
    
#     if not active_agents_top_n.empty:
#         fig_q10 = px.bar(
#             active_agents_top_n.sort_values(by=['listing_count'], ascending=False), 
#             x='listing_count', 
#             y='agent', 
#             color='city',
#             orientation='h',
#             title=f'Q10: Top 10 Agents in High-Value Markets (> {Q1_HIGH_VALUE_THRESHOLD:,.0f} PKR)',
#             height=400
#         )
#         st.plotly_chart(fig_q10, use_container_width=True)
#         st.caption("Agent activity is sensitive to the selected price threshold.")
#     else:
#         st.warning(f"Q10: No high-value agent listings found above the current threshold.")


# # Location Analysis
# st.markdown('<h2 class="section-header">🗺️ Location Analysis</h2>', unsafe_allow_html=True)

# # Create a map visualization
# if not filtered_df.empty:
#     center_lat = filtered_df['latitude'].mean() if not filtered_df['latitude'].empty else 30.3753
#     center_lon = filtered_df['longitude'].mean() if not filtered_df['longitude'].empty else 69.3451
    
#     fig_map = px.scatter_mapbox(
#         filtered_df,
#         lat="latitude",
#         lon="longitude",
#         color="price",
#         size="area_in_marla",
#         hover_name="property_type",
#         hover_data=["city", "location", "price", "area_in_marla", "baths", "bedrooms"],
#         title="Property Locations in Pakistan",
#         zoom=5,
#         height=500,
#         color_continuous_scale=px.colors.cyclical.IceFire
#     )
    
#     fig_map.update_layout(
#         mapbox_style="open-street-map",
#         mapbox=dict(
#             center=dict(lat=center_lat, lon=center_lon),
#             zoom=5
#         ),
#         margin={"r":0,"t":30,"l":0,"b":0}
#     )
    
#     st.plotly_chart(fig_map, use_container_width=True)
    
#     # Show city statistics
#     st.subheader("City-wise Property Distribution")
#     city_stats = filtered_df.groupby('city').agg({
#         'price': ['count', 'mean', 'median'],
#         'area_in_marla': 'mean'
#     }).round(2)
    
#     city_stats.columns = ['Property Count', 'Avg Price', 'Median Price', 'Avg Area (Marla)']
#     st.dataframe(city_stats.sort_values('Property Count', ascending=False))
    
# else:
#     st.info("No data available for the selected filters to display on the map.")



# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center'>
#         <p>Property Market Analysis Dashboard | Created with Streamlit</p>
#         <p><small>Data Source: Zameen.com Property Listings</small></p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# -------------------------------------------------------------------
# 1. Page configuration & basic styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistan Property Market Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 Pakistan Property Market Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides insights into the Pakistani property market based on data from Zameen.com.
Explore property trends, prices, and distributions across different cities and property types.
""")

# -------------------------------------------------------------------
# 2. Data loading helpers
# -------------------------------------------------------------------

def resolve_data_path():
    possible_paths = [
        "zameen-property.csv",                          # if running inside 12-eda
        "./zameen-property.csv",
        "12-eda/zameen-property.csv",                   # correct path on Streamlit Cloud
        os.path.join(os.getcwd(), "zameen-property.csv"),
        os.path.join(os.getcwd(), "12-eda", "zameen-property.csv"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    st.error("❌ Could not find 'zameen-property.csv'.")
    st.write("📂 Working directory:", os.getcwd())
    st.write("📄 Files:", os.listdir(os.getcwd()))
    st.stop()



@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and clean the property dataset.
    Implements the same logic as your notebook:
    - select relevant columns
    - handle missing values
    - transform area into marla
    - add year/month
    - light outlier filtering for visuals
    """
    file_path = resolve_data_path()

    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Ensure required columns exist
    required_cols = [
        'property_type', 'price', 'location', 'city', 'province_name',
        'latitude', 'longitude', 'baths', 'bedrooms', 'date_added',
        'area', 'purpose', 'agency', 'agent'
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns in CSV: {missing_cols}")
        st.write("Available columns:", list(df.columns))
        st.stop()

    # Fill agency/agent like notebook
    df['agency'] = df['agency'].fillna('Unknown')
    df['agent'] = df['agent'].fillna('Unknown')

    # Work on a copy with relevant columns
    df_cleaned = df[required_cols].copy()

    # Area transformation
    df_cleaned['area_value'] = (
        df_cleaned['area']
        .astype(str)
        .str.extract(r'(\d+(\.\d+)?)', expand=False)[0]
    )
    df_cleaned['area_value'] = pd.to_numeric(df_cleaned['area_value'], errors='coerce')

    df_cleaned['area_unit'] = (
        df_cleaned['area']
        .astype(str)
        .str.extract(r'([Kk]anal|[Mm]arla)', expand=False)
        .str.lower()
    )

    df_cleaned['area_in_marla'] = df_cleaned.apply(
        lambda row: row['area_value'] * 20
        if row['area_unit'] == 'kanal'
        else row['area_value'],
        axis=1
    )

    # Final columns as in notebook
    cleaned_columns = [
        'property_type', 'price', 'location', 'city', 'province_name',
        'latitude', 'longitude', 'baths', 'bedrooms', 'date_added',
        'purpose', 'agency', 'agent', 'area_in_marla'
    ]
    final_df = df_cleaned[cleaned_columns].copy()

    # Convert date to datetime and extract year & month
    final_df['date_added'] = pd.to_datetime(
        final_df['date_added'],
        errors='coerce',
        format='%m-%d-%Y'
    )
    final_df['year_added'] = final_df['date_added'].dt.year
    final_df['month_added'] = final_df['date_added'].dt.month

    # Basic outlier removal (matching your notebook intent)
    final_df = final_df[
        (final_df['baths'] < 100) &
        (final_df['bedrooms'] < 20) &
        (final_df['price'] > 1000)
    ]

    # If you specifically only want 2018/2019 as in your code:
    # final_df = final_df[final_df['year_added'].isin([2018, 2019])]

    return final_df


df = load_data()

if df.empty:
    st.error("Loaded dataframe is empty after cleaning. Please verify the CSV content.")
    st.stop()

st.sidebar.success("✅ Data loaded successfully!")

# -------------------------------------------------------------------
# 3. Sidebar filters
# -------------------------------------------------------------------

st.sidebar.header("Filters")

# Static Q1/Q10 threshold for high-value market
Q1_HIGH_VALUE_THRESHOLD = 50_000_000
st.sidebar.markdown(f"**Q1/Q10 High-Value Threshold:** PKR {Q1_HIGH_VALUE_THRESHOLD:,.0f}")

# Guard for missing key columns
for col in ['price', 'city', 'property_type', 'purpose', 'area_in_marla']:
    if col not in df.columns:
        st.error(f"Required column '{col}' is missing from the dataset.")
        st.stop()

# Price range slider
min_price, max_price = int(df['price'].min()), int(df['price'].max())
price_range_tuple = st.sidebar.slider(
    "Price Range (Overall Filter)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# City filter
cities = sorted(df['city'].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect(
    "Select Cities for Comparison",
    options=cities,
    default=cities
)

# Property type filter
property_types = sorted(df['property_type'].dropna().unique().tolist())
selected_property_type = st.sidebar.selectbox(
    "Select Property Type",
    options=['All'] + property_types
)

# Purpose filter
purposes = sorted(df['purpose'].dropna().unique().tolist())
selected_purpose = st.sidebar.selectbox(
    "Select Purpose",
    options=['All'] + purposes
)

# Area range slider
min_area = int(df['area_in_marla'].min() if df['area_in_marla'].notna().any() else 0)
max_area = int(df['area_in_marla'].max() if df['area_in_marla'].notna().any() else 1000)
area_range = st.sidebar.slider(
    "Area Range (Marla)",
    min_value=min_area,
    max_value=max_area,
    value=(min_area, max_area)
)

# Apply all filters
filtered_df = df.copy()

if selected_cities:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

if selected_property_type != 'All':
    filtered_df = filtered_df[filtered_df['property_type'] == selected_property_type]

if selected_purpose != 'All':
    filtered_df = filtered_df[filtered_df['purpose'] == selected_purpose]

filtered_df = filtered_df[
    (filtered_df['price'] >= price_range_tuple[0]) &
    (filtered_df['price'] <= price_range_tuple[1]) &
    (filtered_df['area_in_marla'] >= area_range[0]) &
    (filtered_df['area_in_marla'] <= area_range[1])
]

# -------------------------------------------------------------------
# 4. Key Metrics & Data Table
# -------------------------------------------------------------------

st.markdown('<h2 class="section-header">📊 Key Metrics</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Properties", len(filtered_df))

with col2:
    avg_price = filtered_df['price'].mean() if not filtered_df.empty else 0
    st.metric("Average Price (PKR)", f"{avg_price:,.0f}")

with col3:
    avg_area = filtered_df['area_in_marla'].mean() if not filtered_df.empty else 0
    st.metric("Average Area (Marla)", f"{avg_area:.1f}")

with col4:
    most_common_city = filtered_df['city'].mode()[0] if not filtered_df.empty else "N/A"
    st.metric("Most Common City", most_common_city)

st.markdown("---")

st.markdown('<h2 class="section-header">📋 Property Data</h2>', unsafe_allow_html=True)

if not filtered_df.empty:
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_property_data.csv",
        mime="text/csv"
    )

    with st.expander("📊 Data Summary"):
        st.write("**Dataset Overview:**")
        st.write(f"- Total properties in original dataset: {len(df)}")

        if df['date_added'].notna().any():
            st.write(f"- Date range: {df['date_added'].min().strftime('%Y-%m-%d')} "
                     f"to {df['date_added'].max().strftime('%Y-%m-%d')}")
        else:
            st.write("- Date range: N/A (invalid or missing dates)")

        st.write(f"- Cities covered: {', '.join(sorted(df['city'].dropna().unique()))}")
        st.write(f"- Property types: {', '.join(sorted(df['property_type'].dropna().unique()))}")

        st.write("**Price Statistics (original dataset):**")
        st.write(f"- Minimum price: {df['price'].min():,}")
        st.write(f"- Maximum price: {df['price'].max():,}")
        st.write(f"- Average price: {df['price'].mean():,.0f}")
else:
    st.info("No data available for the selected filters.")

# -------------------------------------------------------------------
# 5. Q9: Purpose Distribution (Sale vs Rent)
# -------------------------------------------------------------------

st.markdown('<h2 class="section-header">Distribution of Property Purposes</h2>', unsafe_allow_html=True)

if not filtered_df.empty:
    purpose_distribution = (
        filtered_df
        .groupby(['city', 'purpose'])
        .size()
        .reset_index(name='count')
    )

    if not purpose_distribution.empty:
        fig_q9 = px.bar(
            purpose_distribution,
            x='city',
            y='count',
            color='purpose',
            title='Sale vs. Rent Listings by City'
        )
        st.plotly_chart(fig_q9, use_container_width=True)
        st.caption("Most listings are typically 'For Sale'.")
    else:
        st.info("No purpose distribution data for the selected filters.")
else:
    st.info("No data available for the selected filters.")

st.markdown("---")

# -------------------------------------------------------------------
# 6. Price Analysis Section
# -------------------------------------------------------------------

st.markdown('<h2 class="section-header">💰 Price Analysis & Distribution</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if not filtered_df.empty:
        avg_price_by_city = (
            filtered_df
            .groupby('city')['price']
            .mean()
            .sort_values(ascending=False)
        )
        fig3 = px.bar(
            x=avg_price_by_city.index,
            y=avg_price_by_city.values,
            title="Average Price by City",
            labels={'x': 'City', 'y': 'Average Price (PKR)'},
            color=avg_price_by_city.index
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data available for the selected filters")

with col2:
    if not filtered_df.empty:
        avg_price_by_type = (
            filtered_df
            .groupby('property_type')['price']
            .mean()
            .sort_values(ascending=False)
        )
        fig4 = px.bar(
            x=avg_price_by_type.index,
            y=avg_price_by_type.values,
            title="Average Price by Property Type",
            labels={'x': 'Property Type', 'y': 'Average Price (PKR)'},
            color=avg_price_by_type.index
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data available for the selected filters")

# -------------------------------------------------------------------
# 7. Detailed Business Insights (Q1, Q3–Q8, Q10)
# -------------------------------------------------------------------

st.markdown('<h2 class="section-header">📈 Detailed Business Insights</h2>', unsafe_allow_html=True)

def get_top_n_listings(df_in: pd.DataFrame, group_col: str, n: int = 10) -> pd.DataFrame:
    """Helper to get top N agencies/agents by listing count."""
    if df_in.empty or group_col not in df_in.columns:
        return pd.DataFrame()

    listing_counts = (
        df_in
        .groupby([group_col, 'city'])
        .size()
        .reset_index(name='listing_count')
    )
    listing_counts = listing_counts[listing_counts[group_col] != 'Unknown']
    if listing_counts.empty:
        return listing_counts

    top_n_groups = (
        listing_counts
        .groupby(group_col)['listing_count']
        .sum()
        .nlargest(n)
        .index
    )
    return listing_counts[listing_counts[group_col].isin(top_n_groups)]

# Row 1: Q1 (high-value growth) & Q5 (seasonality)
col_q1, col_q5 = st.columns(2)

with col_q1:
    st.subheader("High-Value Property Growth (YoY)")
    high_value_data = df[df['price'] > Q1_HIGH_VALUE_THRESHOLD].copy()

    if not high_value_data.empty and high_value_data['year_added'].notna().any():
        high_value_growth = (
            high_value_data
            .groupby(['city', 'year_added'])
            .size()
            .reset_index(name='property_count')
        )
        if high_value_growth['year_added'].nunique() > 1:
            high_value_growth['growth'] = (
                high_value_growth
                .groupby('city')['property_count']
                .diff()
            )

            fig_q1 = px.line(
                high_value_growth.dropna(subset=['year_added', 'property_count']),
                x='year_added',
                y='property_count',
                color='city',
                markers=True,
                title=f"Q1: High-Value Listings (> {Q1_HIGH_VALUE_THRESHOLD:,.0f} PKR) Growth"
            )
            fig_q1.update_layout(xaxis={'tickmode': 'linear', 'dtick': 1})
            st.plotly_chart(fig_q1, use_container_width=True)
            st.caption("Tracks the change in luxury property listings year-over-year.")
        else:
            st.info("Q1: Need at least 2 distinct years for growth analysis.")
    else:
        st.info("Q1: Insufficient data for YoY high-value growth analysis.")

with col_q5:
    st.subheader("Seasonal Trend in Property Listings")
    if not filtered_df.empty and filtered_df['year_added'].notna().any():
        seasonal_trend = (
            filtered_df
            .groupby(['year_added', 'month_added'])
            .size()
            .reset_index(name='listing_count')
        )

        if not seasonal_trend.empty:
            fig_q5 = px.line(
                seasonal_trend,
                x='month_added',
                y='listing_count',
                color='year_added',
                markers=True,
                title='Q5: Seasonal Trend by Month'
            )
            fig_q5.update_layout(xaxis={'tickmode': 'linear', 'dtick': 1})
            st.plotly_chart(fig_q5, use_container_width=True)
            st.caption("Identifies months with higher/lower listing volumes.")
        else:
            st.info("Q5: No seasonal trend data for current filters.")
    else:
        st.info("Q5: Date information is not sufficient for seasonality analysis.")

# Row 2: Q7 (province/type) & Q6 (area-price correlation)
col_q7, col_q6 = st.columns(2)

with col_q7:
    st.subheader("Property Type Prevalence by Province")
    if not df.empty:
        province_property_distribution = (
            df
            .groupby(['province_name', 'property_type'])
            .size()
            .reset_index(name='count')
        )
        if not province_property_distribution.empty:
            fig_q7 = px.bar(
                province_property_distribution,
                x='province_name',
                y='count',
                color='property_type',
                title='Q7: Property Type Prevalence by Province'
            )
            st.plotly_chart(fig_q7, use_container_width=True)
            st.caption("Shows which property types dominate each province.")
        else:
            st.info("Q7: No province-level distribution data.")
    else:
        st.info("Q7: Dataset is empty.")

with col_q6:
    st.subheader("Area–Price Correlation Table")
    if not filtered_df.empty:
        correlation_df = (
            filtered_df
            .groupby(['city', 'property_type'])
            .apply(lambda d: d['area_in_marla'].corr(d['price']))
            .reset_index(name='correlation')
        )

        correlation_df['correlation_strength'] = correlation_df['correlation'].apply(
            lambda x: 'Highly Correlated'
            if pd.notnull(x) and abs(x) >= 0.5
            else 'Less Correlated'
        )

        st.dataframe(
            correlation_df
            .sort_values('correlation', ascending=False)
            .fillna('N/A')
        )
        st.caption("""
        **Insight:** Area is **highly correlated** with price (|corr| ≥ 0.5) mainly for larger or more uniform segments. 
        For smaller units or mixed segments, correlation weakens.
        """)
    else:
        st.info("Q6: No data for correlation analysis.")

# Row 3: Q8 (baths vs price) & Q3 (beds vs price)
col_q8, col_q3 = st.columns(2)

with col_q8:
    st.subheader("Influence of Number of Bathrooms on Price")
    if not filtered_df.empty:
        df_baths_filtered_q8 = filtered_df[filtered_df['baths'] < 16]
        if not df_baths_filtered_q8.empty:
            bathroom_price_influence = (
                df_baths_filtered_q8
                .groupby(['city', 'baths'])['price']
                .mean()
                .reset_index()
            )

            fig_q8 = px.line(
                bathroom_price_influence,
                x='baths',
                y='price',
                color='city',
                markers=True,
                title='Q8: Average Price vs. Number of Bathrooms'
            )
            st.plotly_chart(fig_q8, use_container_width=True)
            st.caption("Prices generally increase with more bathrooms, especially in premium markets.")
        else:
            st.info("Q8: Not enough data after filtering for baths < 16.")
    else:
        st.info("Q8: No data available for current filters.")

with col_q3:
    st.subheader("Price vs. Bedrooms across Property Types")
    if not filtered_df.empty:
        price_bedroom_variation = (
            filtered_df[filtered_df['bedrooms'] < 15]
            .groupby(['city', 'property_type', 'bedrooms'])['price']
            .mean()
            .reset_index()
        )

        if not price_bedroom_variation.empty:
            fig_q3 = px.line(
                price_bedroom_variation,
                x='bedrooms',
                y='price',
                color='property_type',
                line_dash='city',
                title='Q3: Avg Price vs. Bedrooms (by City & Type)'
            )
            st.plotly_chart(fig_q3, use_container_width=True)
            st.caption("Larger property types see higher price growth with bedroom count.")
        else:
            st.info("Q3: Not enough data for bedrooms/price variation.")
    else:
        st.info("Q3: No data available for current filters.")

# Row 4: Q4 (agencies) & Q10 (agents in high-value markets)
col_q4, col_q10 = st.columns(2)

with col_q4:
    st.subheader("Top Agencies (High-Demand Areas)")
    if not df.empty:
        high_demand_cities = ['Karachi', 'Lahore', 'Rawalpindi', 'Islamabad', 'Faisalabad']
        high_demand_areas = df[df['city'].isin(high_demand_cities)]

        agency_listings_top_n = get_top_n_listings(high_demand_areas, 'agency', 10)

        if not agency_listings_top_n.empty:
            fig_q4 = px.bar(
                agency_listings_top_n.sort_values('listing_count', ascending=False),
                x='listing_count',
                y='agency',
                color='city',
                orientation='h',
                title='Q4: Top 10 Agencies by Total Listings (High-Demand Cities)',
                height=400
            )
            st.plotly_chart(fig_q4, use_container_width=True)
            st.caption("Agency activity is heavily concentrated in major metropolitan centers.")
        else:
            st.info("Q4: No agency data available.")
    else:
        st.info("Q4: Dataset is empty.")

with col_q10:
    st.subheader("Top Agents in High-Value Markets")
    if not filtered_df.empty:
        high_value_data_q10 = filtered_df[filtered_df['price'] > Q1_HIGH_VALUE_THRESHOLD]
        active_agents_top_n = get_top_n_listings(high_value_data_q10, 'agent', 10)

        if not active_agents_top_n.empty:
            fig_q10 = px.bar(
                active_agents_top_n.sort_values('listing_count', ascending=False),
                x='listing_count',
                y='agent',
                color='city',
                orientation='h',
                title=f'Q10: Top 10 Agents in High-Value Markets (> {Q1_HIGH_VALUE_THRESHOLD:,.0f} PKR)',
                height=400
            )
            st.plotly_chart(fig_q10, use_container_width=True)
            st.caption("Shows which agents dominate the luxury property segment.")
        else:
            st.warning(
                f"Q10: No high-value agent listings found above the current threshold ({Q1_HIGH_VALUE_THRESHOLD:,.0f} PKR)."
            )
    else:
        st.info("Q10: No data for current filters.")

# -------------------------------------------------------------------
# 8. Location Analysis (Map + City stats)
# -------------------------------------------------------------------

st.markdown('<h2 class="section-header">🗺️ Location Analysis</h2>', unsafe_allow_html=True)

if not filtered_df.empty and filtered_df['latitude'].notna().any() and filtered_df['longitude'].notna().any():
    center_lat = filtered_df['latitude'].mean()
    center_lon = filtered_df['longitude'].mean()

    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="price",
        size="area_in_marla",
        hover_name="property_type",
        hover_data=["city", "location", "price", "area_in_marla", "baths", "bedrooms"],
        title="Property Locations in Pakistan",
        zoom=5,
        height=500,
        color_continuous_scale=px.colors.cyclical.IceFire
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=5
        ),
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("City-wise Property Distribution")
    city_stats = (
        filtered_df
        .groupby('city')
        .agg({
            'price': ['count', 'mean', 'median'],
            'area_in_marla': 'mean'
        })
        .round(2)
    )
    city_stats.columns = ['Property Count', 'Avg Price (PKR)', 'Median Price (PKR)', 'Avg Area (Marla)']
    st.dataframe(city_stats.sort_values('Property Count', ascending=False))
else:
    st.info("No valid latitude/longitude data available for the selected filters to display on the map.")

# -------------------------------------------------------------------
# 9. Footer
# -------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Property Market Analysis Dashboard | Created with Streamlit</p>
        <p><small>Data Source: Zameen.com Property Listings</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
