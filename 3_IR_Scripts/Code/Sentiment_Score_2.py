import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import warnings
import os
import time
warnings.filterwarnings('ignore')
from pathlib import Path

# Set environment variables to reduce Hugging Face requests
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

def process_finbert_sequential(headlines_list, batch_size=50):
    """Process headlines with FinBERT sequentially to avoid rate limiting"""
    print("Loading FinBERT model once")
    device = 0 if torch.cuda.is_available() else -1
    
    # Load model once
    finbert_analyzer = pipeline("sentiment-analysis", 
                               model="ProsusAI/finbert", 
                               tokenizer="ProsusAI/finbert",
                               device=device)
    
    print(f"FinBERT loaded successfully on {'GPU' if device == 0 else 'CPU'}")
    
    all_results = []
    total_batches = len(headlines_list) // batch_size + (1 if len(headlines_list) % batch_size != 0 else 0)
    
    for i in range(0, len(headlines_list), batch_size):
        batch_headlines = headlines_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            # Add small delay to be respectful to the model
            if i > 0:
                time.sleep(0.1)
                
            finbert_batch_results = finbert_analyzer(batch_headlines)
            
            batch_results = []
            for result in finbert_batch_results:
                label = result['label'].lower()
                score = result['score']
                
                finbert_positive = score if label == 'positive' else 0.0
                finbert_negative = score if label == 'negative' else 0.0
                finbert_neutral = score if label == 'neutral' else 0.0
                
                if label == 'positive':
                    finbert_compound = score
                elif label == 'negative':
                    finbert_compound = -score
                else:
                    finbert_compound = 0.0
                    
                batch_results.append({
                    'finbert_compound': finbert_compound,
                    'finbert_positive': finbert_positive,
                    'finbert_negative': finbert_negative,
                    'finbert_neutral': finbert_neutral,
                    'finbert_label': label
                })
            
            all_results.extend(batch_results)
            
            if batch_num % 100 == 0:
                print(f"Processed {batch_num}/{total_batches} batches ({batch_num/total_batches*100:.1f}%) - {len(all_results)} headlines completed")
                
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            # Add neutral results for failed batch
            batch_results = [{
                'finbert_compound': 0.0,
                'finbert_positive': 0.0,
                'finbert_negative': 0.0,
                'finbert_neutral': 1.0,
                'finbert_label': 'neutral'
            } for _ in batch_headlines]
            all_results.extend(batch_results)
    
    return all_results

def main():
    # Load part 1 results
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir.parent / "4_Sentiment_Scores" / "Sentiment_Scores_TextBlob_VADER.csv"
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
   
    print(f"Loaded {len(df)} headlines with TextBlob and VADER scores")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique stocks: {df['stock'].nunique()}")

    print("Processing FinBERT sentiment analysis (sequential to avoid rate limiting)...")

    # Prepare headlines for FinBERT (truncate to 512 characters for BERT)
    headlines_list = [str(row['headline'])[:512] for _, row in df.iterrows()]

    print(f"Processing {len(headlines_list)} headlines...")

    # Process FinBERT sequentially to avoid rate limiting
    finbert_results = process_finbert_sequential(headlines_list, batch_size=50)

    print("FinBERT processing complete!")

    # Combine all results (add FinBERT scores to existing data)
    print("Combining results...")
    for i, finbert_result in enumerate(finbert_results):
        for key, value in finbert_result.items():
            df.loc[i, key] = value

    print("\nComplete sentiment analysis finished!")
    print(f"Generated all sentiment scores for {len(df)} headlines")

    # Display sample results
    print("\nSample results:")
    sample_cols = ['headline', 'textblob_polarity', 'vader_compound', 'finbert_compound', 'finbert_label']
    print(df[sample_cols].head())

    # Save complete results
    output_path = script_dir.parent / "4_Sentiment_Scores" / "Sentiment_Scores_Complete.csv"
    df.to_csv(output_path, index=False)
    print(f"\nComplete results saved to '{output_path}' with {len(df)} records")

    # Create aggregated daily sentiment scores by stock
    print("Creating daily aggregated sentiment scores...")
    daily_sentiment = df.groupby(['stock', 'date']).agg({
        'textblob_polarity': 'mean',
        'textblob_subjectivity': 'mean',
        'vader_compound': 'mean',
        'vader_positive': 'mean',
        'vader_negative': 'mean',
        'vader_neutral': 'mean',
        'finbert_compound': 'mean',
        'finbert_positive': 'mean',
        'finbert_negative': 'mean',
        'finbert_neutral': 'mean',
        'headline': 'count'
    }).reset_index()

    daily_sentiment.rename(columns={'headline': 'news_count'}, inplace=True)
    daily_sentiment.to_csv('Daily_Sentiment_Aggregated.csv', index=False)

    print(f"Daily aggregated sentiment saved to 'Daily_Sentiment_Aggregated.csv'")
    print(f"Aggregated to {len(daily_sentiment)} stock-date combinations")
    print("\nReady for predictive modeling!")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total headlines processed: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique stocks: {df['stock'].nunique()}")
    print(f"Daily aggregations: {len(daily_sentiment)}")

    print("\nSentiment score ranges:")
    print(f"TextBlob Polarity: {df['textblob_polarity'].min():.3f} to {df['textblob_polarity'].max():.3f}")
    print(f"VADER Compound: {df['vader_compound'].min():.3f} to {df['vader_compound'].max():.3f}")
    print(f"FinBERT Compound: {df['finbert_compound'].min():.3f} to {df['finbert_compound'].max():.3f}")

    print("\nFinBERT label distribution:")
    print(df['finbert_label'].value_counts())


if __name__ == '__main__':
    main()