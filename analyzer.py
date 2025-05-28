import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def analyze_sentiment(text):
    score = sid.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive', compound
    elif compound <= -0.05:
        return 'negative', compound
    return 'neutral', compound

def load_data(file_path, review_col, rating_col=None):
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return None
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return None
        
        if review_col not in df.columns:
            print(f"Warning: {review_col} not found in {file_path}")
            return None
        
        df['cleaned_review'] = df[review_col].apply(clean_text)
        sentiments = df['cleaned_review'].apply(analyze_sentiment).apply(pd.Series)
        df['sentiment'] = sentiments[0]
        df['sentiment_score'] = sentiments[1]
        
        summary = {
            'positive': (df['sentiment'] == 'positive').sum(),
            'negative': (df['sentiment'] == 'negative').sum(),
            'neutral': (df['sentiment'] == 'neutral').sum()
        }
        return {'df': df, 'summary': summary}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None