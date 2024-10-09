import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
from datetime import datetime, timedelta
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

model = joblib.load('models/crop_price_model.pkl')
crop_data = pd.read_csv('data/crop_data.csv')
calendar_today = pd.read_csv('data/india.csv')

# OpenWeatherMap API key
OPENWEATHERMAP_API_KEY = "a0d77a2c94c1137ceb3d4a8d17ede9f6"

NEWS_API_KEY = '72c08fc45c7348a39c4f84c249c2211d'


def get_weather(city):
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}"
    response = requests.get(base_url)
    if response.status_code != 200:
        st.error(f"Error in API call: {response.text}")
        return None
    return response.json()


def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]


def fetch_agricultural_news():
    date_from = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q=farmers-india&from={date_from}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles'][:100]  # Return top 100 articles
    else:
        st.error(f"Error fetching news: {response.status_code}")
        return []


def summarize_text(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")

    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)


def get_accumulated_summary(articles):
    all_content = " ".join([article['content'] or article['description'] for article in articles])
    return summarize_text(all_content, sentences_count=5)


# Streamlit page configuration
st.set_page_config(page_title="AgriForecast Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f9f4;
        color: #1e3d59;
    }
    .sidebar .sidebar-content {
        background-color: #1e3d59;
        color: #f4f9f4;
    }
    .sidebar .sidebar-content .element-container {
        padding: 1rem;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #e74c3c;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #1e3d59;
        border: 2px solid #4a7c59;
        border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        background-color: #ffffff;
        color: #1e3d59;
        border: 2px solid #4a7c59;
        border-radius: 5px;
    }
    .stSlider>div>div>div {
        background-color: #4a7c59;
    }
    h1, h2, h3 {
        color: #4a7c59;
    }
    .stAlert {
        background-color: #fef9e7;
        color: #1e3d59;
        border: 2px solid #f9bc60;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #4a7c59;
        color: #ffffff;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3d59;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .plot-container {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🌾 Kissan Konnect: Crop Price Prediction & Weather Insights")
st.markdown("Empowering farmers with data-driven decisions")

# tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price Trends", "🌱 Crop Prediction", "🌦️ Weather Forecast", "📰 Agricultural News Summary"])

with tab1:
    st.header("Historical Price Trends")
    col1, col2 = st.columns([3, 1])

    with col1:
        commodity = st.selectbox("Select Commodity", calendar_today.columns[1:])
        fig = px.line(calendar_today, x='date', y=commodity, title=f'{commodity} Prices Over Time',
                      labels={'date': 'Date', commodity: 'Price (₹)'},
                      color_discrete_sequence=['#4a7c59'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1e3d59',
            title_font_color='#4a7c59'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Quick Stats")
        latest_price = calendar_today[commodity].iloc[-1]
        avg_price = calendar_today[commodity].mean()
        max_price = calendar_today[commodity].max()
        min_price = calendar_today[commodity].min()

        st.metric("Latest Price", f"₹{latest_price:.2f}")
        st.metric("Average Price", f"₹{avg_price:.2f}")
        st.metric("Maximum Price", f"₹{max_price:.2f}")
        st.metric("Minimum Price", f"₹{min_price:.2f}")

    st.subheader("Interactive Data Exploration")
    selected_commodities = st.multiselect("Select Commodities", calendar_today.columns[1:], default=[commodity])
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(selected_commodities):
        fig.add_trace(go.Scatter(x=calendar_today['date'], y=calendar_today[col], mode='lines', name=col,
                                 line=dict(color=colors[i % len(colors)])))
    fig.update_layout(
        title='Commodity Price Comparison',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#1e3d59',
        title_font_color='#4a7c59'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Crop Price Prediction")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Parameters")
        crop = st.selectbox('Select Crop', crop_data['Crop'].unique())
        season = st.selectbox('Select Season', crop_data['Season'].unique())
        state = st.selectbox('Select State', crop_data['State'].unique())
        soil_type = st.selectbox('Select Soil Type', crop_data['Soil Type'].unique())

        col1a, col1b = st.columns(2)
        with col1a:
            pesticide_usage = st.select_slider('Pesticide Usage', options=['Low', 'Medium', 'High'])
            ph = st.slider('Soil pH', 4.0, 9.0, 6.5)
            temperature = st.slider('Temperature (°C)', 10.0, 50.0, 30.0)
        with col1b:
            fertilizer_usage = st.select_slider('Fertilizer Usage', options=['Low', 'Medium', 'High'])
            area = st.number_input('Area (in hectares)', min_value=0.1, value=1.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=100.0)

    with col2:
        st.subheader("Price Prediction")
        if st.button('Predict Price', key='predict_button'):
            input_data = {
                'Crop': crop, 'Season': season, 'State': state, 'Soil Type': soil_type,
                'Pesticide Usage': pesticide_usage, 'pH': ph, 'Temperature': temperature,
                'Fertilizer Usage': fertilizer_usage, 'Area': area, 'Rainfall': rainfall
            }
            predicted_price = predict_price(input_data)
            st.success(f"Predicted Price: ₹{predicted_price:.2f}")

            avg_price = crop_data[crop_data['Crop'] == crop]['Price'].mean()
            comparison_data = pd.DataFrame({
                'Category': ['Predicted Price', 'Average Price'],
                'Price': [predicted_price, avg_price]
            })
            fig = px.bar(comparison_data, x='Category', y='Price', title='Price Comparison',
                         color='Category',
                         color_discrete_map={'Predicted Price': '#ff6b6b', 'Average Price': '#4a7c59'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e3d59',
                title_font_color='#4a7c59'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Weather Forecast")
    city = st.text_input("Enter city for weather information", placeholder="e.g., Mumbai")
    if city:
        with st.spinner('Fetching weather data...'):
            weather_data = get_weather(city)
        if weather_data:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"Current Weather in {city}")
                temperature = weather_data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
                humidity = weather_data['main']['humidity']
                wind_speed = weather_data['wind']['speed']
                visibility = weather_data['visibility'] / 1000  # Convert to km
                description = weather_data['weather'][0]['description']

                weather_icon = weather_data['weather'][0]['icon']
                st.image(f"http://openweathermap.org/img/wn/{weather_icon}@2x.png", width=100)

                col1a, col1b, col1c = st.columns(3)
                col1a.metric("Temperature", f"{temperature:.1f}°C")
                col1b.metric("Humidity", f"{humidity}%")
                col1c.metric("Wind Speed", f"{wind_speed} m/s")

                st.write(f"**Visibility:** {visibility:.1f} km")
                st.write(f"**Description:** {description.capitalize()}")

            with col2:
                st.subheader("Agricultural Impact")
                if temperature > 30:
                    st.warning("High temperature may stress crops. Consider irrigation.")
                elif temperature < 10:
                    st.warning("Low temperature may slow growth. Protect sensitive crops.")

                if humidity > 80:
                    st.warning("High humidity may increase disease risk. Monitor crops closely.")
                elif humidity < 30:
                    st.warning("Low humidity may increase water requirements. Adjust irrigation.")

                if wind_speed > 10:
                    st.warning("Strong winds may damage crops. Consider wind breaks.")

                if "rain" in description.lower():
                    st.info("Rainfall expected. Adjust irrigation schedules accordingly.")
        else:
            st.error("Unable to fetch weather data. Please check the city name and try again.")

with tab4:
    st.header("Agricultural News Summary")
    if st.button("Fetch and Summarize News"):
        with st.spinner("Fetching and summarizing agricultural news..."):
            articles = fetch_agricultural_news()
            if articles:
                accumulated_summary = get_accumulated_summary(articles)
                st.subheader("Accumulated Summary of Agricultural News")
                st.write(accumulated_summary)

                st.subheader("Recent Agricultural News Articles")
                for article in articles[:5]:  # Display top 5 articles
                    st.markdown(f"**{article['title']}**")
                    st.write(f"Published At: {article['publishedAt']}")
                    st.write(f"Source: {article['source']['name']}")
                    st.write(article['description'])
                    st.markdown(f"[Read more]({article['url']})")
                    st.markdown("---")
            else:
                st.error("No articles found or there was an error fetching the news.")

st.sidebar.header("📊 Dashboard Overview")
st.sidebar.info(
    "Kissan Konnect integrates historical price analysis, predictive modeling, "
    "and real-time weather data to provide comprehensive insights for agricultural decision-making."
)

st.sidebar.header("🔍 How to Use")
st.sidebar.markdown(
    """
    1. **Price Trends**: Analyze historical crop prices and compare multiple commodities.
    2. **Crop Prediction**: Input your crop and environmental parameters to get a price forecast.
    3. **Weather Forecast**: Check current weather conditions and their potential impact on agriculture.
    4. **Agricultural News**: Stay updated with the latest agricultural news and summarized insights.
    """
)

st.sidebar.header("🚀 About the Hackathon")
st.sidebar.success(
    "This demo showcases the potential of data-driven solutions in agriculture. "
    "Our goal is to empower farmers with actionable insights for better decision-making and increased profitability."
)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ for the SIH 2024")
