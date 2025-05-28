import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def generate_graphs():
    data_dir = 'data'
    graph_dir = 'static/graphs'
    os.makedirs(graph_dir, exist_ok=True)
    
    file_configs = {
        'amazon_review.csv': {'platform': 'amazon', 'review_col': 'reviewText', 'rating_col': 'overall'},
        'BestBut_Review.xlsx': {'platform': 'bestbuy', 'review_col': 'review_text', 'rating_col': 'N_stars'},
        'ebay_reviews.csv': {'platform': 'ebay', 'review_col': 'review content', 'rating_col': 'rating'},
        'flipkart_product.csv': {'platform': 'flipkart', 'review_col': 'Summary', 'rating_col': 'Rate'},  # Updated to Summary
        'wallmart_review.csv': {'platform': 'walmart', 'review_col': 'Review', 'rating_col': 'Rating'},  # Updated to Review
        'sixth_file.csv': {'platform': 'sixth', 'review_col': 'review', 'rating_col': 'rating'}  # Adjust as needed
    }
    
    all_data = {}
    for filename, config in file_configs.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
        
        try:
            if filename.endswith('.csv'):
                if filename == 'flipkart_product.csv':  # Updated file name
                    df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
                else:
                    df = pd.read_csv(file_path, encoding='utf-8')
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                continue
            
            print(f"Loaded {filename} with shape: {df.shape}")
            review_col = config['review_col']
            if review_col not in df.columns:
                print(f"Warning: {review_col} not found in {filename}")
                continue
            
            df['cleaned_review'] = df[review_col].apply(clean_text)
            sentiments = df['cleaned_review'].apply(analyze_sentiment).apply(pd.Series)
            df['sentiment'] = sentiments[0]
            df['sentiment_score'] = sentiments[1]
            
            summary = {
                'positive': (df['sentiment'] == 'positive').sum(),
                'negative': (df['sentiment'] == 'negative').sum(),
                'neutral': (df['sentiment'] == 'neutral').sum()
            }
            total_reviews = sum(summary.values())
            if total_reviews > 0:
                summary = {
                    'positive': summary['positive'] / total_reviews,
                    'negative': summary['negative'] / total_reviews,
                    'neutral': summary['neutral'] / total_reviews
                }
            all_data[config['platform']] = {'df': df, 'summary': summary}
            
            # Graph 1: Bar chart for sentiment (normalized)
            plt.figure(figsize=(6, 4))
            sns.barplot(x=['Positive', 'Negative', 'Neutral'], y=[summary['positive'], summary['negative'], summary['neutral']])
            plt.title(f'{config["platform"].capitalize()} Sentiment Distribution (Normalized)')
            plt.ylabel('Proportion')
            plt.savefig(os.path.join(graph_dir, f'{config["platform"]}_bar.png'))
            plt.close()
            
            # Graph 2: Pie chart for sentiment (normalized)
            plt.figure(figsize=(6, 4))
            plt.pie([summary['positive'], summary['negative'], summary['neutral']], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107'])
            plt.title(f'{config["platform"].capitalize()} Sentiment Pie (Normalized)')
            plt.savefig(os.path.join(graph_dir, f'{config["platform"]}_pie.png'))
            plt.close()
            
            # Graph 3: Line graph for hypothetical score distribution
            avg_scores = df.groupby('sentiment')['sentiment_score'].mean()
            plt.figure(figsize=(6, 4))
            plt.plot(avg_scores.index, avg_scores.values, marker='o')
            plt.title(f'{config["platform"].capitalize()} Average Sentiment Scores')
            plt.ylabel('Average Score')
            plt.savefig(os.path.join(graph_dir, f'{config["platform"]}_line.png'))
            plt.close()
            
            # Generate individual platform report
            platform_report = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Proportion': [summary['positive'], summary['negative'], summary['neutral']]
            })
            report_path = os.path.join(graph_dir, f'{config["platform"]}_report.csv')
            platform_report.to_csv(report_path, index=False)
            print(f"Generated individual report at {report_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Graph 4: Data points across all sites
    platforms = list(all_data.keys())
    counts = [len(all_data[p]['df']) for p in platforms]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=platforms, y=counts)
    plt.title('Number of Data Points per E-commerce Site')
    plt.xlabel('Platform')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(graph_dir, 'datapoints.png'))
    plt.close()
    
    # Graph 5: Comparative bar chart across all sites (side-by-side bars, normalized)
    platforms = list(all_data.keys())
    positive = [all_data[p]['summary']['positive'] for p in platforms]
    negative = [all_data[p]['summary']['negative'] for p in platforms]
    neutral = [all_data[p]['summary']['neutral'] for p in platforms]
    
    x = range(len(platforms))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - width for i in x], positive, width, label='Positive', color='#4CAF50')
    plt.bar([i for i in x], negative, width, label='Negative', color='#F44336')
    plt.bar([i + width for i in x], neutral, width, label='Neutral', color='#FFC107')
    
    plt.xlabel('Platform')
    plt.ylabel('Proportion')
    plt.title('Comparative Sentiment Analysis Across Sites (Normalized)')
    plt.xticks([i for i in x], platforms, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'comparison.png'))
    plt.close()
    
    # Generate overall comparison report
    report_data = {
        'Platform': platforms,
        'Positive Proportion': positive,
        'Negative Proportion': negative,
        'Neutral Proportion': neutral
    }
    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(graph_dir, 'sentiment_report.csv')
    report_df.to_csv(report_path, index=False)
    print(f"Generated overall report at {report_path}")

if __name__ == '__main__':
    generate_graphs()
    print("Graphs generated successfully in static/graphs/")