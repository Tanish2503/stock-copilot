# 📊 Stock Copilot Dashboard

A Streamlit web application for comprehensive stock analysis, news aggregation, and AI-powered natural language querying.

## ✨ Features

- **🌎 Multi-country/market support:** Analyze stocks from various global exchanges (USA, India, UK, Canada, Germany, Japan).
- **🔍 Global stock search:** Find stocks by company name or ticker symbol across supported markets using the Financial Modeling Prep (FMP) API.
- **📊 Real-time price charts & technicals:** View interactive candlestick charts with moving averages (SMA 20, SMA 50), RSI, MACD, and volume data powered by yfinance.
- **🏦 Company info & business summary:** Get key details like sector, industry, employee count, and a brief business description.
- **📰 News headlines & sentiment analysis:** Stay updated with the latest news articles and their sentiment distribution (Positive, Neutral, Negative) using NewsAPI.org and TextBlob.
- **🤖 AI-powered Q&A (OpenAI):** Ask natural language questions about a selected stock using a custom query engine powered by OpenAI and LangChain (placeholder implementation).
- **⭐ Favorites:** Quickly save and access your preferred stocks.
- **📥 Export data:** Download stock price data and company information in CSV or Excel format.
- **💵 Currency Conversion:** View market overview metrics converted to the local currency of the selected country using exchange rate data.

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tanish2503/stock-copilot.git
    cd stock-copilot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows use `venv\Scripts\activate`
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    -   Create a `.env` file in the root directory of the project.
    -   Get API keys for the following services:
        -   **OpenAI:** Required for the Q&A feature. Get one from [OpenAI Platform](https://platform.openai.com/).
        -   **NewsAPI.org:** Required for news headlines. Get one from [NewsAPI.org](https://newsapi.org/).
        -   **Financial Modeling Prep (FMP):** Required for global stock search. Get a free key from [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/).
    -   Add your keys to the `.env` file like this:
        ```env
        OPENAI_API_KEY=your_openai_key_here
        NEWS_API_KEY=your_newsapi_key_here
        FMP_API_KEY=your_fmp_key_here
        ```

## ▶️ How to Run

1.  **Ensure your virtual environment is active** and dependencies are installed (`pip install -r requirements.txt`).
2.  **Make sure your `.env` file** with the required API keys is in the project root.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app/main.py
    ```
4.  The app should open in your web browser.

## 🤝 Contributing

Feel free to fork the repository, make improvements, and submit pull requests.
