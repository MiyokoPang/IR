import pandas as pd
import re
from pathlib import Path

# Load dataset
script_dir = Path(__file__).resolve().parent
raw_csv = script_dir.parent / "1_Raw_Data" / "raw_partner_headlines.csv"
df = pd.read_csv(raw_csv)

# Drop nulls and duplicates
df.dropna(subset=["headline", "date", "stock"], inplace=True)
df.drop_duplicates(subset=["headline", "date", "stock"], inplace=True)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Remove the 1969 outlier
print(f"Original dataset: {len(df):,} records")
print(f"Date range before filtering: {df['date'].min().date()} to {df['date'].max().date()}")

# Filter to 2010 onwards
df = df[df["date"] >= "2010-01-01"]

print(f"After removing outlier dates: {len(df):,} records")
print(f"Date range after filtering: {df['date'].min().date()} to {df['date'].max().date()}")

# Clean headline text (keep alphanumeric characters only, convert to lowercase)
df["clean_headline"] = df["headline"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).lower())

# Keep only required columns
df_cleaned = df[["index", "clean_headline", "date", "stock"]].rename(columns={"clean_headline": "headline"})

# Save cleaned file
output_path = script_dir.parent / "2_Cleaned_Data" / "Cleaned_Sentiment_1.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"Data cleaned and saved to {output_path}")
print(f"Final dataset: {len(df_cleaned):,} records from {len(df_cleaned['stock'].unique()):,} unique stocks")

