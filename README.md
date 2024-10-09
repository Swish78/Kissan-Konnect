# Kissan Konnect

Kissan Konnect is an innovative platform designed to assist farmers by providing real-time data and insights on various agricultural parameters such as crop prices, weather conditions, and summarized agricultural news. The platform uses machine learning for crop price prediction and news summarization for concise information delivery.

## Features

### 1. Price Trends
- Analyze and compare historical crop prices for multiple commodities.
- Visualize price changes over time to make informed decisions about selling or purchasing crops.

### 2. Crop Prediction
- Input crop and environmental parameters such as soil type, temperature, and rainfall.
  
### 3. Weather Forecast
- Access real-time weather updates and forecasts.
- Analyze the potential impact of weather on crop yield and plan farming activities accordingly.

### 4. Agricultural News Summarization
- Fetch the latest agricultural news related to farmers, fertilizers, and agriculture.
- Summarize the articles using **Sumy**, a custom text summarization solution, to provide key insights without the need to read full-length articles.
  
## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Flask/Streamlit)
- **Database**: MySQL (for historical price storage)
- **Data Visualization**: Plotly for interactive charts
- **News Summarization**: Sumy-based custom summarizer using `LsaSummarizer`
- **APIs**:
    - **NewsAPI**: For fetching agricultural news.
    - **Weather API**: For real-time weather updates.
- **Machine Learning**: Crop price forecasting model with environmental parameters.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/kissan-konnect.git
    cd kissan-konnect
    ```

2. **Set up a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK Data (for Summarization)**:
    ```bash
    python -m nltk.downloader punkt
    ```

5. **Set Up Environment Variables**:
    - You will need an API key from [NewsAPI](https://newsapi.org/).
    - Create a `.env` file in the root directory with your API key:
    ```bash
    NEWS_API_KEY=your_news_api_key_here
    ```

6. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

## Dependencies

Ensure that the following libraries are installed:
- **Streamlit**: For creating interactive web apps.
- **Pandas**: For data handling.
- **Plotly**: For interactive visualizations.
- **Joblib**: For loading machine learning models.
- **Requests**: For making API calls.
- **Sumy**: A text summarization tool using `LsaSummarizer`.
- **NLTK**: For natural language processing and tokenization.
- **Datetime and Time**: For handling time intervals and API request timestamps.

To install these libraries, ensure you use the `requirements.txt` file provided or run:

```bash
pip install streamlit pandas plotly joblib requests sumy nltk
```
