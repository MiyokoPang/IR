import pandas as pd
from pathlib import Path

# Define file paths
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "2_Cleaned_Data"
sentiment_path = data_dir / "Cleaned_Sentiment_1.csv"
failed_path = data_dir / "Failed_Tickers_20250727_150046.csv"
output_path = data_dir / "Cleaned_Sentiment_Final.csv"

# Load the datasets
sentiment_df = pd.read_csv(sentiment_path)
failed_tickers_df = pd.read_csv(failed_path)

# Get the list of failed tickers
failed_tickers = failed_tickers_df.iloc[:, 0].tolist()
print(f"Original sentiment data shape: {sentiment_df.shape}")
print(f"Number of failed tickers to remove: {len(failed_tickers)}")

# Check how many rows will be affected
rows_to_remove = sentiment_df[sentiment_df['stock'].isin(failed_tickers)]
print(f"Rows to be removed: {len(rows_to_remove)}")

# Remove rows where stock ticker is in the failed tickers list
cleaned_sentiment_df = sentiment_df[~sentiment_df['stock'].isin(failed_tickers)]

print(f"Cleaned sentiment data shape: {cleaned_sentiment_df.shape}")
print(f"Rows removed: {sentiment_df.shape[0] - cleaned_sentiment_df.shape[0]}")

# Save the cleaned dataset
cleaned_sentiment_df.to_csv(output_path, index=False)
print(f"Cleaned sentiment data saved to '{output_path}'")

# Display summary statistics
remaining_stocks = cleaned_sentiment_df['stock'].nunique()
print(f"Remaining unique stocks: {remaining_stocks}")
print(f"Date range: {cleaned_sentiment_df['date'].min()} to {cleaned_sentiment_df['date'].max()}")
