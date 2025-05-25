from dotenv import load_dotenv
import os

# Force reload environment variables
load_dotenv(override=True)

# Load API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# Clean up the API key if it has any extra characters
if news_api_key:
    news_api_key = news_api_key.split()[0]
    news_api_key = news_api_key.split('#')[0]
    news_api_key = news_api_key.strip()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import io
from typing import Dict, Any, Optional, Tuple, List
from textblob import TextBlob
from ratelimit import limits, sleep_and_retry
import time
from functools import wraps
import requests

# Add the project root to Python path to import query_engine
import sys
from pathlib import Path
app_dir = Path(__file__).parent.absolute()
sys.path.append(str(app_dir))

try:
    from query_engine import get_qa_response
except ImportError as e:
    def get_qa_response(question, stock_symbol):
        return f"Query engine not available. Question received: {question} for stock: {stock_symbol}"

# Configure Streamlit page - must be first Streamlit command
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'favorite_stocks' not in st.session_state:
    st.session_state.favorite_stocks = []

# Cached functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_info(symbol: str) -> Dict[str, Any]:
    """Get cached stock info."""
    try:
        stock = yf.Ticker(symbol)
        return stock.info
    except Exception as e:
        st.error(f"Error fetching stock info: {str(e)}")
        return {}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_news_headlines(symbol: str) -> list:
    """Get cached news headlines."""
    return load_news_headlines(symbol)

def calculate_performance_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance metrics."""
    try:
        returns = data['Close'].pct_change()
        metrics = {
            'daily_volatility': returns.std() * 100,
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0,
            'max_drawdown': (data['Close'] / data['Close'].cummax() - 1).min() * 100,
            'total_return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
        }
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis indicators to the data."""
    try:
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

# Rate limiting decorator
ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 30

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def rate_limited_api_call(func):
    """Rate limit API calls."""
    return func()

def rate_limit(func):
    """Decorator to rate limit any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return rate_limited_api_call(lambda: func(*args, **kwargs))
    return wrapper

# Load FMP API key
fmp_api_key = os.getenv("FMP_API_KEY")

# Map country to FMP exchange code
fmp_exchange_map = {
    "USA": "NASDAQ,NYSE,AMEX",
    "India (NSE)": "NSE",
    "UK (LSE)": "LSE",
    "Canada (TSX)": "TSX",
    "Germany (XETRA)": "XETRA",
    "Japan (TSE)": "TSE"
}

# Replace search_stocks with FMP-powered version
@st.cache_data(ttl=3600)
def search_stocks(query: str, country: str) -> List[Tuple[str, str]]:
    """Search for stocks globally using FMP API."""
    if not query or not fmp_api_key:
        return []
    try:
        exchange = fmp_exchange_map.get(country, "NASDAQ,NYSE,AMEX")
        url = f"https://financialmodelingprep.com/api/v3/search"
        params = {
            "query": query,
            "limit": 20,
            "exchange": exchange,
            "apikey": fmp_api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = [(item["symbol"], item.get("name", item["symbol"])) for item in data]
        return results
    except Exception as e:
        st.error(f"FMP search error: {str(e)}")
        return []

def analyze_news_sentiment(news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze sentiment of news headlines."""
    sentiments = []
    for news in news_data:
        try:
            text = news.get('title', '') + ' ' + news.get('summary', '')
            sentiment = TextBlob(text).sentiment
            
            # Categorize sentiment
            if sentiment.polarity > 0.1:
                category = "Positive"
                color = "green"
            elif sentiment.polarity < -0.1:
                category = "Negative"
                color = "red"
            else:
                category = "Neutral"
                color = "gray"
            
            sentiments.append({
                'title': news.get('title'),
                'sentiment_score': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'category': category,
                'color': color,
                'date': news.get('date'),
                'url': news.get('url')
            })
        except Exception as e:
            continue
    
    return sentiments

def export_stock_data(data: pd.DataFrame, info: Dict[str, Any], format: str = 'csv') -> bytes:
    """Export stock data in various formats."""
    try:
        if format == 'csv':
            # Combine price data and company info
            export_data = data.copy()
            export_data['Company'] = info.get('shortName', '')
            export_data['Sector'] = info.get('sector', '')
            export_data['Industry'] = info.get('industry', '')
            return export_data.to_csv(index=True).encode('utf-8')
        elif format == 'excel':
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name='Price Data')
                info_df = pd.DataFrame([
                    {'Metric': k, 'Value': v}
                    for k, v in info.items() if isinstance(v, (str, int, float))
                ])
                info_df.to_excel(writer, sheet_name='Company Info', index=False)
            buffer.seek(0)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return None

# Map country to currency code and symbol
country_currency_map = {
    "USA": ("USD", "$"),
    "India (NSE)": ("INR", "₹"),
    "UK (LSE)": ("GBP", "£"),
    "Canada (TSX)": ("CAD", "C$"),
    "Germany (XETRA)": ("EUR", "€"),
    "Japan (TSE)": ("JPY", "¥")
}

# Utility: Get exchange rate (USD to target)
@st.cache_data(ttl=3600)
def get_usd_fx_rate(target_currency: str) -> float:
    if target_currency == "USD":
        return 1.0
    try:
        url = f"https://api.exchangerate.host/latest?base=USD&symbols={target_currency}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data['rates'][target_currency]
    except Exception as e:
        st.warning(f"Could not fetch FX rate for {target_currency}, using USD.")
        return 1.0

def create_responsive_metrics(data: pd.DataFrame, info: Dict[str, Any], country: str):
    """Create a responsive metrics dashboard with currency conversion."""
    currency_code, currency_symbol = country_currency_map.get(country, ("USD", "$"))
    fx_rate = get_usd_fx_rate(currency_code)
    # Create a container for metrics
    metrics_container = st.container()
    with metrics_container:
        st.markdown(f"### Market Overview ({currency_symbol} {currency_code})")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = data['Close'].iloc[-1] * fx_rate
            prev_close = info.get('previousClose', current_price / fx_rate) * fx_rate
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0
            st.metric(
                "Current Price",
                f"{currency_symbol}{current_price:,.2f}",
                f"{change:+.2f} ({change_pct:+.2f}%)",
                delta_color="normal"
            )
        with col2:
            market_cap = info.get('marketCap', 0) * fx_rate
            st.metric(
                "Market Cap",
                f"{currency_symbol}{market_cap:,.0f}",
                f"Volume: {data['Volume'].iloc[-1]:,.0f}"
            )
        with col3:
            low = info.get('fiftyTwoWeekLow', 'N/A')
            high = info.get('fiftyTwoWeekHigh', 'N/A')
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                low = f"{currency_symbol}{low * fx_rate:,.2f}"
                high = f"{currency_symbol}{high * fx_rate:,.2f}"
            st.metric(
                "52W Range",
                f"{low} - {high}",
                f"P/E: {info.get('trailingPE', 'N/A')}"
            )
        with col4:
            st.metric(
                "Beta",
                f"{info.get('beta', 'N/A')}",
                f"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%"
            )
        st.markdown(f"### Performance Metrics ({currency_symbol} {currency_code})")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        metrics = calculate_performance_metrics(data)
        with perf_col1:
            st.metric("Daily Volatility", f"{metrics.get('daily_volatility', 0):.2f}%")
        with perf_col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with perf_col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        with perf_col4:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")

# Main app code starts here
st.title("📈 Stock Analysis Dashboard")
st.markdown("This will soon show stock insights and news in real time!")
st.markdown("---")

# --- Country selection logic ---
country_options = {
    "USA": "",
    "India (NSE)": ".NS",
    "UK (LSE)": ".L",
    "Canada (TSX)": ".TO",
    "Germany (XETRA)": ".DE",
    "Japan (TSE)": ".T"
}
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'USA'

country_flag_map = {
    "USA": "🇺🇸",
    "India (NSE)": "🇮🇳",
    "UK (LSE)": "🇬🇧",
    "Canada (TSX)": "🇨🇦",
    "Germany (XETRA)": "🇩🇪",
    "Japan (TSE)": "🇯🇵"
}
country = st.selectbox(
    "🌎 Select Country/Market:",
    list(country_options.keys()),
    key="country_select",
    format_func=lambda c: f"{country_flag_map.get(c, '')} {c}"
)
if country != st.session_state.selected_country:
    st.session_state.selected_country = country
    st.rerun()
country_suffix = country_options[st.session_state.selected_country]

# --- Stock selection logic ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🎯 Select Stock")
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'AAPL'
        st.session_state.selected_stock_name = 'Apple Inc.'

    # Popular stock options per country
    base_stock_options = {
        "USA": {
            "Apple Inc.": "AAPL",
            "Tesla Inc.": "TSLA",
            "Microsoft Corp.": "MSFT",
            "Amazon.com Inc.": "AMZN",
            "Alphabet Inc.": "GOOGL",
            "Meta Platforms": "META",
            "NVIDIA Corp.": "NVDA",
            "Netflix Inc.": "NFLX",
            "Adobe Inc.": "ADBE",
            "Salesforce Inc.": "CRM"
        },
        "India (NSE)": {
            "Reliance Industries": "RELIANCE",
            "Tata Consultancy": "TCS",
            "Infosys": "INFY",
            "HDFC Bank": "HDFCBANK",
            "ICICI Bank": "ICICIBANK",
            "Bharti Airtel": "BHARTIARTL",
            "State Bank of India": "SBIN",
            "Bajaj Finance": "BAJFINANCE",
            "Hindustan Unilever": "HINDUNILVR",
            "Asian Paints": "ASIANPAINT"
        },
        "UK (LSE)": {
            "HSBC Holdings": "HSBA",
            "BP plc": "BP",
            "GlaxoSmithKline": "GSK",
            "Barclays": "BARC",
            "AstraZeneca": "AZN",
            "Vodafone": "VOD",
            "Unilever": "ULVR",
            "Diageo": "DGE",
            "Tesco": "TSCO",
            "Rolls-Royce": "RR"
        },
        "Canada (TSX)": {
            "Royal Bank of Canada": "RY",
            "Toronto-Dominion Bank": "TD",
            "Enbridge": "ENB",
            "Canadian National Railway": "CNR",
            "Bank of Nova Scotia": "BNS",
            "Suncor Energy": "SU",
            "Brookfield": "BN",
            "Canadian Pacific": "CP",
            "Shopify": "SHOP",
            "Barrick Gold": "ABX"
        },
        "Germany (XETRA)": {
            "Siemens": "SIE",
            "SAP": "SAP",
            "Allianz": "ALV",
            "BASF": "BAS",
            "Deutsche Bank": "DBK",
            "Volkswagen": "VOW3",
            "BMW": "BMW",
            "Adidas": "ADS",
            "Deutsche Telekom": "DTE",
            "Infineon": "IFX"
        },
        "Japan (TSE)": {
            "Toyota": "7203",
            "Sony": "6758",
            "Mitsubishi UFJ": "8306",
            "SoftBank": "9984",
            "Keyence": "6861",
            "Recruit Holdings": "6098",
            "Nintendo": "7974",
            "Shin-Etsu": "4063",
            "Sumitomo Mitsui": "8316",
            "KDDI": "9433"
        }
    }
    stock_options = {k: v + country_suffix for k, v in base_stock_options[st.session_state.selected_country].items()}

    # Add favorites section
    if st.session_state.favorite_stocks:
        st.markdown("**⭐ Favorites:**")
        for fav in st.session_state.favorite_stocks:
            fav_col1, fav_col2 = st.columns([3, 1])
            with fav_col1:
                if st.button(f"⭐ {fav}", key=f"select_fav_{fav}"):
                    st.session_state.selected_stock = fav
                    st.session_state.selected_stock_name = fav
                    st.session_state.custom_selected = False
                    st.rerun()
            with fav_col2:
                if st.button("🗑️ Remove", key=f"remove_{fav}"):
                    st.session_state.favorite_stocks.remove(fav)
                    st.rerun()

    # Only set selectbox index if current stock is in the list
    stock_keys = list(stock_options.keys())
    stock_values = list(stock_options.values())
    if st.session_state.selected_stock in stock_values:
        selectbox_index = stock_values.index(st.session_state.selected_stock)
    else:
        selectbox_index = 0
    selected_stock_name_dropdown = st.selectbox(
        "📋 Choose a stock:",
        options=stock_keys,
        index=selectbox_index,
        key="dropdown_stock_name"
    )
    if st.button("✅ Select from List", key="select_from_list"):
        st.session_state.selected_stock = stock_options[selected_stock_name_dropdown]
        st.session_state.selected_stock_name = selected_stock_name_dropdown
        st.session_state.custom_selected = False
        st.rerun()

    # Custom stock input
    st.markdown("**Or enter custom symbol:**")
    custom_stock = st.text_input("🔤 Stock Symbol:", placeholder="e.g., TSLA", key="custom_stock_input").upper()
    if st.button("✅ Select Custom Symbol", key="select_custom_stock") and custom_stock:
        st.session_state.selected_stock = custom_stock + country_suffix if not custom_stock.endswith(country_suffix) else custom_stock
        st.session_state.selected_stock_name = custom_stock
        st.session_state.custom_selected = True
        st.rerun()

    # Search box for stocks (updated to use FMP API)
    search_query = st.text_input("🔍 Search stocks:", placeholder="Enter company name or symbol")
    search_results = []
    if search_query:
        search_results = search_stocks(search_query, st.session_state.selected_country)
        if search_results:
            st.markdown("**🔎 Search Results:**")
            stock_labels = [f"{symbol} - {name}" for symbol, name in search_results]
            selected_search_idx = st.selectbox(
                "🔽 Select a stock from search results:",
                options=list(range(len(stock_labels))),
                format_func=lambda i: stock_labels[i],
                key="search_result_selectbox"
            )
            if st.button("✅ Select Stock", key="select_search_stock"):
                st.session_state.selected_stock = search_results[selected_search_idx][0]
                st.session_state.selected_stock_name = search_results[selected_search_idx][1]
                st.success(f"Selected {st.session_state.selected_stock}")
                st.rerun()

    # Add to favorites button
    if st.button("➕ Add to Favorites"):
        if st.session_state.selected_stock not in st.session_state.favorite_stocks:
            st.session_state.favorite_stocks.append(st.session_state.selected_stock)
            st.success(f"Added {st.session_state.selected_stock} to favorites!")

with col2:
    st.subheader(f"📊 {country_flag_map.get(st.session_state.selected_country, '')} {st.session_state.selected_stock_name} ({st.session_state.selected_stock})")

# Function to fetch stock data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period="7d"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def load_news_headlines(stock_symbol):
    """Get news headlines using News API."""
    try:
        if not news_api_key:
            st.error("News API key not found in environment variables")
            return [{"title": "News API key not configured", 
                    "summary": "Please set NEWS_API_KEY in your .env file", 
                    "date": "N/A"}]

        # Get company name from yfinance for better search
        try:
            stock = yf.Ticker(stock_symbol)
            company_name = stock.info.get('shortName', stock_symbol)
        except Exception:
            company_name = stock_symbol

        # Prepare search query
        query = f"{company_name} OR {stock_symbol}"
        
        # News API endpoint
        url = "https://newsapi.org/v2/everything"
        
        # Parameters for the API request
        params = {
            'q': query,
            'apiKey': news_api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'searchIn': 'title,description',
            'excludeDomains': 'youtube.com,tiktok.com,instagram.com'
        }
        
        # Make the API request
        response = requests.get(url, params=params)
        if response.status_code != 200:
            response.raise_for_status()
        
        data = response.json()
        if data['status'] != 'ok':
            error_message = data.get('message', 'Unknown error')
            st.error(f"News API Error: {error_message}")
            return [{"title": "Error fetching news", 
                    "summary": f"News API returned status: {data['status']}. Message: {error_message}", 
                    "date": "N/A"}]
        
        # Transform the news data into our format
        news_data = []
        articles = data.get('articles', [])
        for article in articles:
            news_data.append({
                'title': article.get('title', ''),
                'summary': article.get('description', ''),
                'date': article.get('publishedAt', 'N/A'),
                'url': article.get('url', '')
            })
        
        if not news_data:
            st.warning(f"No news articles found for {stock_symbol}")
            return [{"title": f"No news found for {stock_symbol}", 
                    "summary": "Try searching with a different stock symbol", 
                    "date": "N/A"}]
        
        return news_data

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return [{"title": "Error fetching news", 
                "summary": f"Network error: {str(e)}", 
                "date": "N/A"}]
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return [{"title": "Error fetching news", 
                "summary": f"Unexpected error: {str(e)}", 
                "date": "N/A"}]

# Utility: Normalize ticker for yfinance

def normalize_ticker_for_yfinance(symbol: str, country: str) -> str:
    # Remove everything after first period or comma
    base = symbol.split('.')[0].split(',')[0]
    suffix_map = {
        "India (NSE)": ".NS",
        "UK (LSE)": ".L",
        "Canada (TSX)": ".TO",
        "Germany (XETRA)": ".DE",
        "Japan (TSE)": ".T",
        # USA: no suffix
    }
    suffix = suffix_map.get(country, "")
    return base + suffix

# Fetch stock data with loading state and error handling
with st.spinner("Fetching stock data..."):
    try:
        yf_symbol = normalize_ticker_for_yfinance(st.session_state.selected_stock, st.session_state.selected_country)
        data, info = fetch_stock_data(yf_symbol)
        if data is not None and not data.empty:
            # Add technical indicators
            data = add_technical_indicators(data)
            
            # Create responsive metrics dashboard
            create_responsive_metrics(data, info, st.session_state.selected_country)
            
            # Export options
            st.markdown("### 📥 Export Data")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📊 Export as CSV"):
                    csv_data = export_stock_data(data, info, 'csv')
                    if csv_data:
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            f"stock_data_{st.session_state.selected_stock}.csv",
                            "text/csv"
                        )
            
            with export_col2:
                if st.button("📊 Export as Excel"):
                    excel_data = export_stock_data(data, info, 'excel')
                    if excel_data:
                        st.download_button(
                            "📥 Download Excel",
                            excel_data,
                            f"stock_data_{st.session_state.selected_stock}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            # Create two columns for the main content
            left_col, right_col = st.columns([2, 1])
            
            with left_col:
                # Stock price chart with technical indicators
                st.subheader("📈 Price Chart & Technical Indicators")
                fig = go.Figure()
                # Add candlestick chart
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=st.session_state.selected_stock
                ))
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    name='20-day SMA',
                    line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    name='50-day SMA',
                    line=dict(color='blue', width=1)
                ))
                # Add volume as bar chart on secondary y-axis
                fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    yaxis='y2',
                    opacity=0.3,
                    marker_color='lightblue'
                ))
                # Update layout
                fig.update_layout(
                    title=f"{st.session_state.selected_stock} - Technical Analysis",
                    yaxis_title="Price ($)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying='y',
                        side='right'
                    ),
                    xaxis_title="Date",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add RSI and MACD charts
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("RSI (14)")
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=3),
                        mode='lines+markers',
                        marker=dict(size=6),
                        hovertemplate='Date: %{x}<br>RSI: %{y:.2f}'
                    ))
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="top left")
                    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom left")
                    rsi_fig.update_layout(
                        height=350,
                        margin=dict(l=40, r=20, t=40, b=40),
                        plot_bgcolor="#18191A",
                        paper_bgcolor="#18191A",
                        font=dict(size=16, color="#FAFAFA"),
                        xaxis=dict(title="Date", showgrid=True, gridcolor="#333"),
                        yaxis=dict(title="RSI Value", showgrid=True, gridcolor="#333", range=[0, 100]),
                        title=dict(text="RSI (14)", font=dict(size=24)),
                        hovermode="x unified"
                    )
                    st.plotly_chart(rsi_fig, use_container_width=True)
                
                with col2:
                    st.subheader("MACD")
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        name='MACD',
                        line=dict(color='blue', width=3),
                        mode='lines+markers',
                        marker=dict(size=6),
                        hovertemplate='Date: %{x}<br>MACD: %{y:.2f}'
                    ))
                    macd_fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Signal_Line'],
                        name='Signal Line',
                        line=dict(color='red', width=3, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=6),
                        hovertemplate='Date: %{x}<br>Signal: %{y:.2f}'
                    ))
                    macd_fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Zero Line", annotation_position="top left")
                    macd_fig.update_layout(
                        height=350,
                        margin=dict(l=40, r=20, t=40, b=40),
                        plot_bgcolor="#18191A",
                        paper_bgcolor="#18191A",
                        font=dict(size=16, color="#FAFAFA"),
                        xaxis=dict(title="Date", showgrid=True, gridcolor="#333"),
                        yaxis=dict(title="MACD Value", showgrid=True, gridcolor="#333"),
                        title=dict(text="MACD", font=dict(size=24)),
                        hovermode="x unified"
                    )
                    st.plotly_chart(macd_fig, use_container_width=True)
            
            with right_col:
                # Performance metrics
                st.subheader("📊 Performance Metrics")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("Daily Volatility", f"{calculate_performance_metrics(data).get('daily_volatility', 0):.2f}%")
                    st.metric("Sharpe Ratio", f"{calculate_performance_metrics(data).get('sharpe_ratio', 0):.2f}")
                
                with metric_col2:
                    st.metric("Max Drawdown", f"{calculate_performance_metrics(data).get('max_drawdown', 0):.2f}%")
                    st.metric("Total Return", f"{calculate_performance_metrics(data).get('total_return', 0):.2f}%")

            # Company info
            if info:
                st.subheader("🏦 Company Info")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
                
                # Company description
                business_summary = info.get('longBusinessSummary', '')
                if business_summary:
                    st.write("**Business Summary:**")
                    st.write(business_summary[:300] + "..." if len(business_summary) > 300 else business_summary)
            else:
                st.error(f"No data available for {st.session_state.selected_stock}")
        else:
            st.error(f"No data available for {st.session_state.selected_stock}")
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.info("Please try refreshing the data or selecting a different stock.")

# News section with sentiment analysis
st.markdown("---")
st.subheader(f"📰 Latest Headlines & Sentiment Analysis for {st.session_state.selected_stock}")

with st.spinner("Loading and analyzing news headlines..."):
    try:
        news_headlines = get_news_headlines(st.session_state.selected_stock)
        if news_headlines:
            # Analyze sentiment
            sentiments = analyze_news_sentiment(news_headlines)
            
            # Display sentiment summary
            sentiment_counts = {
                'Positive': len([s for s in sentiments if s['category'] == 'Positive']),
                'Neutral': len([s for s in sentiments if s['category'] == 'Neutral']),
                'Negative': len([s for s in sentiments if s['category'] == 'Negative'])
            }
            
            # Create sentiment summary chart
            sentiment_labels = list(sentiment_counts.keys())
            sentiment_values = list(sentiment_counts.values())
            sentiment_colors = ["#2ecc40", "#bdbdbd", "#ff4136"]  # green, gray, red
            sentiment_fig = go.Figure()
            sentiment_fig.add_trace(go.Bar(
                x=sentiment_labels,
                y=sentiment_values,
                marker_color=sentiment_colors,
                text=sentiment_values,
                textposition="auto",
                width=0.5
            ))
            sentiment_fig.update_layout(
                title="News Sentiment Distribution",
                height=250,
                showlegend=False,
                xaxis=dict(title="Sentiment", tickfont=dict(size=14)),
                yaxis=dict(title="Count", tickfont=dict(size=14), range=[0, max(sentiment_values + [1]) + 1]),
                plot_bgcolor="#fafafa",
                bargap=0.3,
                margin=dict(l=30, r=30, t=40, b=30)
            )
            st.plotly_chart(sentiment_fig, use_container_width=True)
            
            # Display news with sentiment
            for sentiment in sentiments:
                with st.expander(
                    f"📄 {sentiment['title']} ({sentiment['category']})",
                    expanded=False
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(sentiment['title'])
                        st.markdown(f"**Sentiment Score:** {sentiment['sentiment_score']:.2f}")
                        st.markdown(f"**Subjectivity:** {sentiment['subjectivity']:.2f}")
                        
                        if sentiment['url']:
                            st.markdown(f"[Read full article]({sentiment['url']})")
                    
                    with col2:
                        st.write(f"**Date:** {sentiment['date']}")
                        st.markdown(
                            f"<div style='color: {sentiment['color']};'>"
                            f"**Category:** {sentiment['category']}</div>",
                            unsafe_allow_html=True
                        )
        else:
            st.info(f"No news headlines available for {st.session_state.selected_stock}")
    
    except Exception as e:
        st.error(f"Error loading news: {str(e)}")

# Natural Language Query Section
st.markdown("---")
st.subheader("🤖 Ask Questions About This Stock")

query_col1, query_col2 = st.columns([3, 1])

with query_col1:
    user_question = st.text_input(
        "Ask anything about the stock:",
        placeholder=f"e.g., What are the key risks for {st.session_state.selected_stock}?",
        key="stock_question"
    )

with query_col2:
    ask_button = st.button("Ask Question", type="primary")

if ask_button and user_question.strip():
    with st.spinner("Analyzing your question..."):
        try:
            response = get_qa_response(user_question, st.session_state.selected_stock)
            st.success("**Answer:**")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.info("Make sure query_engine.py is properly configured and accessible.")

# Example questions
with st.expander("💡 Example Questions"):
    st.write("Try asking questions like:")
    st.write("- What are the main business segments?")
    st.write("- What are the key financial metrics?")
    st.write("- What are the major risks and opportunities?")
    st.write("- How has the stock performed recently?")
    st.write("- What do analysts think about this stock?")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>💡 This app provides stock data visualization, AI-powered Q&A, and news aggregation.<br>
        Data sources: Yahoo Finance, Custom Query Engine, Local News Feeds</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar with additional info
with st.sidebar:
    st.header("📊 Stock Copilot Dashboard")
    st.write("**✨ Features:**")
    st.write("• 🌎 Multi-country/market support")
    st.write("• 🔍 Global stock search (name/ticker)")
    st.write("• 📊 Real-time price charts & technicals")
    st.write("• 🏢 Company info & business summary")
    st.write("• 📰 News headlines & sentiment analysis")
    st.write("• 🤖 AI-powered Q&A (OpenAI)")
    st.write("• ⭐ Favorites (quick select)")
    st.write("• 📥 Export data (CSV, Excel)")
    st.markdown("---")
    st.write("**🔗 Data Sources:**")
    st.write("• 🟦 Yahoo Finance API (yfinance)")
    st.write("• 🟧 Financial Modeling Prep (FMP) API")
    st.write("• 📰 NewsAPI.org")
    st.write("• 🤖 OpenAI (Q&A)")
    st.markdown("---")
    st.write(f"**🕒 Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()