import streamlit as st
import pandas as pd
import yfinance as yf
 
# Set the page configuration with an icon
st.set_page_config(page_title="Upstok", page_icon="bull-market.png")

st.write("# UpStok")

# List of predefined ticker symbols for the selectbox
ticker_options = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'SPY']

# Let the user select a ticker symbol from the options
tickerSymbol = st.selectbox("Choose a Stock Ticker Symbol", ticker_options)

# Cache function to avoid fetching the data repeatedly
@st.cache_data
def get_stock_data(tickerSymbol, start_date, end_date):
    tickerData = yf.Ticker(tickerSymbol)
    # Fetch historical data with a smaller date range for faster loading
    tickerDf = tickerData.history(start=start_date, end=end_date)
    return tickerDf

# Select date range
start_date = st.date_input("Start Date", pd.to_datetime("2010-05-31"))
end_date = st.date_input("End Date", pd.to_datetime("2024-05-31"))

# Show a loading spinner while fetching data
with st.spinner(f"Fetching {tickerSymbol} stock data..."):
    tickerDf = get_stock_data(tickerSymbol, start_date, end_date)

# Display the fetched data
st.write(f"Displaying data for {tickerSymbol} from {start_date} to {end_date}")

# Display closing price chart
st.write("Closing Price")
st.line_chart(tickerDf['Close'])

# Display trading volume chart
st.write("Closing Volume")
st.line_chart(tickerDf['Volume'])

# Success message when data is successfully loaded
st.success('Data loaded successfully!')
