import requests
from datetime import datetime, timedelta
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
nltk.download('punkt')

NEWS_API_KEY = '72c08fc45c7348a39c4f84c249c2211d'


def fetch_agricultural_news():
    date_from = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q=farmers-fertilizer&from={date_from}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles'][:100]  # Returns top 100 articles
    else:
        print(f"Error fetching news: {response.status_code}")
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


def main():
    while True:
        print("\nFetching and summarizing agricultural news...")
        articles = fetch_agricultural_news()

        if articles:
            accumulated_summary = get_accumulated_summary(articles)
            print("\nAccumulated Summary of Agricultural News:")
            print(accumulated_summary)
        else:
            print("No articles found or there was an error fetching the news.")

        print("\nWaiting for 3 hours before the next update...")
        time.sleep(3 * 60 * 60)  # Wait for 3 hours


if __name__ == "__main__":
    main()