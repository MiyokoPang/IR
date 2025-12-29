import pandas as pd
import numpy as np
from datetime import datetime
import os

def clean_and_merge_data():
    """
    Clean and merge sentiment and price data for modeling
    """
    print("Loading data files...")
    
    # Load sentiment data
    sentiment_df = pd.read_csv('Sentiment_Scores_Complete.csv')
    print(f"Loaded {len(sentiment_df)} sentiment records")
    
    # Load price data
    price_df = pd.read_csv('Stock_Price_20250727_150046.csv')
    print(f"Loaded {len(price_df)} price records")
    
    # Clean sentiment data
    print("Cleaning sentiment data...")
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date  # Convert to date only
    sentiment_df = sentiment_df.rename(columns={'stock': 'ticker'})  # Standardize column name
    
    # Clean price data
    print("Cleaning price data...")
    price_df['Date'] = pd.to_datetime(price_df['Date'], utc=True).dt.date  # Convert to date only
    price_df = price_df.rename(columns={
        'Date': 'date',
        'Close': 'adj_close'  # Your models expect 'adj_close'
    })
    
    # Create daily sentiment aggregations
    print("Aggregating sentiment by ticker and date...")
    daily_sentiment = sentiment_df.groupby(['ticker', 'date']).agg({
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
    
    # Merge price and sentiment data
    print("Merging price and sentiment data...")
    merged_df = pd.merge(
        price_df[['date', 'ticker', 'Open', 'High', 'Low', 'adj_close', 'Volume']], 
        daily_sentiment,
        on=['ticker', 'date'],
        how='left'  # Keep all price data, fill missing sentiment with neutral values
    )
    
    # Fill missing sentiment values with neutral/zero values
    sentiment_cols = ['textblob_polarity', 'textblob_subjectivity', 'vader_compound', 
                     'vader_positive', 'vader_negative', 'vader_neutral',
                     'finbert_compound', 'finbert_positive', 'finbert_negative', 'finbert_neutral']
    
    for col in sentiment_cols:
        if col in ['vader_neutral', 'finbert_neutral', 'textblob_subjectivity']:
            merged_df[col] = merged_df[col].fillna(0.5)  # Neutral values
        else:
            merged_df[col] = merged_df[col].fillna(0.0)  # Zero for sentiment scores
    
    merged_df['news_count'] = merged_df['news_count'].fillna(0)
    
    # Convert date back to datetime for processing
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Sort by ticker and date
    merged_df = merged_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"Merged dataset: {len(merged_df)} records")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Unique tickers: {merged_df['ticker'].nunique()}")
    
    # Save merged dataset
    merged_df.to_csv('Merged_Price_Sentiment_Data.csv', index=False)
    print("Saved merged data to 'Merged_Price_Sentiment_Data.csv'")
    
    # Create weekly aggregations (as expected by your models)
    print("Creating weekly aggregations...")
    weekly_df = merged_df.groupby(['ticker', pd.Grouper(key='date', freq='W')]).agg({
        'adj_close': 'last',  # Last price of the week
        'Volume': 'sum',
        'textblob_polarity': 'mean',
        'vader_compound': 'mean',
        'finbert_compound': 'mean',
        'news_count': 'sum'
    }).reset_index()
    
    # Remove weeks with no data
    weekly_df = weekly_df.dropna(subset=['adj_close'])
    weekly_df = weekly_df.rename(columns={'ticker': 'stock'})  # Match model expectation
    
    weekly_df.to_csv('Stock_Prices.csv', index=False)
    print("Saved weekly price data to 'Stock_Prices.csv'")
    
    # Create sentiment file expected by models
    sentiment_weekly = merged_df.groupby(['ticker', pd.Grouper(key='date', freq='W')]).agg({
        'textblob_polarity': 'mean',
        'vader_compound': 'mean', 
        'finbert_compound': 'mean'
    }).reset_index()
    
    sentiment_weekly = sentiment_weekly.rename(columns={'ticker': 'stock'})
    sentiment_weekly.to_csv('Merged_Scored_TextBlob.csv', index=False)
    print("Saved sentiment data to 'Merged_Scored_TextBlob.csv'")
    
    return merged_df, weekly_df

def create_individual_ticker_files(merged_df):
    """
    Create individual CSV files for each ticker (for CNN model)
    """
    print("Creating individual ticker files...")
    
    if not os.path.exists('Data'):
        os.makedirs('Data')
    if not os.path.exists('Data/Ticker_Price_Files'):
        os.makedirs('Data/Ticker_Price_Files')
    
    tickers = merged_df['ticker'].unique()
    
    for ticker in tickers:
        ticker_data = merged_df[merged_df['ticker'] == ticker].copy()
        
        # Create features expected by CNN model
        ticker_data = ticker_data.sort_values('date')
        
        # Add technical indicators
        ticker_data['return'] = ticker_data['adj_close'].pct_change()
        ticker_data['ma_5'] = ticker_data['adj_close'].rolling(5).mean()
        ticker_data['ma_10'] = ticker_data['adj_close'].rolling(10).mean()
        
        # Select columns for CNN model
        output_cols = ['date', 'adj_close', 'textblob_polarity', 'vader_compound', 'finbert_compound', 
                      'return', 'ma_5', 'ma_10']
        
        ticker_output = ticker_data[output_cols].dropna()
        
        if len(ticker_output) > 20:  # Only save if we have enough data
            filepath = f'Data/Ticker_Price_Files/{ticker}.csv'
            ticker_output.to_csv(filepath, index=False)
    
    print(f"Created individual files for {len(tickers)} tickers")

def main():
    print("=== DATA PREPARATION PIPELINE ===")
    print()
    
    # Step 1: Clean and merge data
    merged_df, weekly_df = clean_and_merge_data()
    
    print()
    
    # Step 2: Create individual ticker files for CNN
    create_individual_ticker_files(merged_df)
    
    print()
    print("=== DATA PREPARATION COMPLETE ===")
    print("Files created:")
    print("- Merged_Price_Sentiment_Data.csv (daily data)")
    print("- Stock_Prices.csv (weekly aggregated)")  
    print("- Merged_Scored_TextBlob.csv (weekly sentiment)")
    print("- Data/Ticker_Price_Files/*.csv (individual ticker files)")
    print()

if __name__ == "__main__":
    main()