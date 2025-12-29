import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Load the cleaned sentiment data
script_dir = Path(__file__).resolve().parent
input_path = script_dir.parent / "2_Cleaned_Data" / "Cleaned_Sentiment_Final.csv"
df = pd.read_csv(input_path)
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded {len(df)} headlines for sentiment analysis")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique stocks: {df['stock'].nunique()}")

# Initialize sentiment analyzers
print("Initializing TextBlob and VADER analyzers")

# VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

print("Analyzers initialized successfully!")

def get_textblob_sentiment(text):
    try:
        blob = TextBlob(str(text))
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    except:
        return {
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0
        }

def get_vader_sentiment(text):
    try:
        scores = vader_analyzer.polarity_scores(str(text))
        return {
            'vader_compound': scores['compound'],
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu']
        }
    except:
        return {
            'vader_compound': 0.0,
            'vader_positive': 0.0,
            'vader_negative': 0.0,
            'vader_neutral': 1.0
        }

# Process sentiment analysis in batches 
batch_size = 1000
total_batches = len(df) // batch_size + 1

print(f"Processing {len(df)} headlines in {total_batches} batches...")

sentiment_results = []

for i in range(0, len(df), batch_size):
    batch_end = min(i + batch_size, len(df))
    batch_df = df.iloc[i:batch_end].copy()
    
    print(f"Processing batch {i//batch_size + 1}/{total_batches}")
    
    # Process TextBlob for entire batch
    print("  - Processing TextBlob...")
    textblob_results = []
    for idx, row in batch_df.iterrows():
        textblob_results.append(get_textblob_sentiment(row['headline']))
    print("  - TextBlob complete")
    
    # Process VADER for entire batch
    print("  - Processing VADER...")
    vader_results = []
    for idx, row in batch_df.iterrows():
        vader_results.append(get_vader_sentiment(row['headline']))
    print("  - VADER complete")
    
    # Combine TextBlob and VADER results for this batch
    batch_results = []
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        result = {
            'index': row['index'],
            'headline': row['headline'],
            'date': row['date'],
            'stock': row['stock']
        }
        result.update(textblob_results[idx])
        result.update(vader_results[idx])
        
        batch_results.append(result)
    
    sentiment_results.extend(batch_results)

# Create intermediate dataframe with TextBlob and VADER scores
sentiment_df = pd.DataFrame(sentiment_results)

print("\nTextBlob and VADER sentiment analysis completed!")
print(f"Generated sentiment scores for {len(sentiment_df)} headlines")

# Display sample results
print("\nSample results:")
sample_cols = ['headline', 'textblob_polarity', 'vader_compound']
print(sentiment_df[sample_cols].head())

# Save results
output_path = script_dir.parent/ "4_Sentiment_Scores" / "Sentiment_Scores_TextBlob_VADER.csv"
sentiment_df.to_csv(output_path, index=False)
print(f"\nResults saved to '{output_path}' with {len(sentiment_df)} records")
print("Ready for FinBERT processing")