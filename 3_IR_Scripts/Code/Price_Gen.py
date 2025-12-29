import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timedelta
import time
import logging
import re
from typing import List, Dict, Optional
import warnings
import random
import glob
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

# Unified stock data processor for sentiment analysis datasets    
class StockDataProcessor:
    def __init__(self, max_workers: int = 10, delay_between_requests: float = 0.1):
        self.max_workers = max_workers
        self.delay_between_requests = delay_between_requests
        self.results = {
            'successful_tickers': [],
            'failed_tickers': [],
            'delisted_tickers': [],
            'invalid_tickers': []
        }
        
    def clean_ticker(self, ticker: str) -> Optional[str]:
        """Clean and validate ticker symbols"""
        if pd.isna(ticker) or ticker == '' or ticker is None:
            return None
            
        ticker = str(ticker).upper().strip()
        ticker = re.sub(r'[^\w.]', '', ticker)
        ticker = ticker.replace('_', '').replace('-', '.')
        ticker = re.sub(r'\.(TO|L|PA|F|DE|AS|MI|MC|SW|HK|SS|SZ|T|TSE|NYSE|NASDAQ)$', '', ticker)
        
        if len(ticker) == 0 or len(ticker) > 6 or not ticker[0].isalpha():
            return None
            
        return ticker
    
    # Fetch historical data for a single ticker with retry logic
    def fetch_ticker_data(self, ticker: str, start_date: str, end_date: str, 
                         max_retries: int = 1) -> Optional[pd.DataFrame]:
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = self.delay_between_requests * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
                else:
                    time.sleep(self.delay_between_requests)
                
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date, 
                    end=end_date, 
                    auto_adjust=True,
                    prepost=False,
                    actions=False,
                    timeout=30
                )
                
                if hist.empty or len(hist[hist['Volume'] > 0]) < 10:
                    self.results['delisted_tickers'].append(ticker)
                    return None
                
                hist = hist[hist['Volume'] > 0]
                hist['ticker'] = ticker
                hist.reset_index(inplace=True)
                hist['Date'] = pd.to_datetime(hist['Date'])
                
                numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numerical_cols:
                    if col in hist.columns:
                        if col == 'Volume':
                            hist[col] = hist[col].astype(int)
                        else:
                            hist[col] = hist[col].round(4)
                
                self.results['successful_tickers'].append(ticker)
                return hist
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.results['failed_tickers'].append(ticker)
                    return None
        
        return None
    
    # Process a batch of tickers concurrently
    def process_ticker_batch(self, tickers: List[str], start_date: str, 
                           end_date: str, retry_mode: bool = False) -> pd.DataFrame:
        all_stock_data = []
        max_retries = 3 if retry_mode else 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_ticker_data, ticker, start_date, end_date, max_retries): ticker 
                for ticker in tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                try:
                    data = future.result()
                    if data is not None:
                        all_stock_data.append(data)
                except Exception:
                    pass
        
        return pd.concat(all_stock_data, ignore_index=True) if all_stock_data else pd.DataFrame()
    
    # Extract tickers from sentiment dataset and clean them
    def extract_tickers_from_sentiment_data(self, csv_file: str) -> Dict:
        
        df = pd.read_csv(csv_file)
        unique_stocks = df['stock'].dropna().unique()
        cleaned_tickers = []
        
        for stock in unique_stocks:
            cleaned = self.clean_ticker(stock)
            if cleaned:
                cleaned_tickers.append(cleaned)
            else:
                self.results['invalid_tickers'].append(stock)
        
        cleaned_tickers = sorted(list(set(cleaned_tickers)))
        
        df['date'] = pd.to_datetime(df['date'])

        start_date = '2010-01-01'  # Match actual sentiment start date 
        end_date = '2020-06-04'    # Match sentiment dataset end date
        
        return {
            'tickers': cleaned_tickers,
            'start_date': start_date,
            'end_date': end_date,
            'total_records': len(df)
        }
    
    def load_failed_tickers(self) -> List[str]:
        
        failed_tickers = []
        
        # Try CSV files first
        csv_files = glob.glob('failed_tickers_*.csv')
        if csv_files:
            try:
                failed_df = pd.read_csv(max(csv_files))
                failed_tickers = failed_df['failed_ticker'].dropna().tolist()
                return failed_tickers
            except Exception:
                pass
        
        # Try report files
        report_files = glob.glob('fetch_report_full_dataset_*.txt')
        if report_files:
            try:
                with open(max(report_files), 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                in_failed_list = False
                
                for line in lines:
                    if 'failed_tickers = [' in line:
                        in_failed_list = True
                        continue
                    elif in_failed_list and line.strip() == ']':
                        break
                    elif in_failed_list and "'" in line:
                        ticker = line.strip().replace("'", "").replace(",", "").strip()
                        if ticker and len(ticker) <= 6:
                            failed_tickers.append(ticker)
            except Exception:
                pass
        
        return failed_tickers
    
    # Main function to generate stock dataset from sentiment data
    def generate_stock_dataset(self, sentiment_csv: str = 'Cleaned_Sentiment_1.csv', 
                             batch_size: int = 200) -> pd.DataFrame:
        print("Processing sentiment dataset for stock prices")
        
        # Extract tickers from sentiment data
        analysis = self.extract_tickers_from_sentiment_data(sentiment_csv)
        tickers = analysis['tickers']
        
        print(f"Found {len(tickers)} unique tickers")
        print(f"Date range: {analysis['start_date']} to {analysis['end_date']}")
        
        # Process initial batch
        all_stock_data = []
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            batch_data = self.process_ticker_batch(
                batch_tickers, analysis['start_date'], analysis['end_date']
            )
            
            if not batch_data.empty:
                all_stock_data.append(batch_data)
            
            if i + batch_size < len(tickers):
                time.sleep(5)
        
        initial_data = pd.concat(all_stock_data, ignore_index=True) if all_stock_data else pd.DataFrame()
        
        # Retry failed tickers if any
        if self.results['failed_tickers']:
            print(f"Retrying {len(self.results['failed_tickers'])} failed tickers")
            
            retry_processor = StockDataProcessor(max_workers=3, delay_between_requests=0.5)
            retry_data = retry_processor.process_ticker_batch(
                self.results['failed_tickers'], 
                analysis['start_date'], 
                analysis['end_date'],
                retry_mode=True
            )
            
            if not retry_data.empty:
                if not initial_data.empty:
                    combined_data = pd.concat([initial_data, retry_data], ignore_index=True)
                else:
                    combined_data = retry_data
            else:
                combined_data = initial_data
        else:
            combined_data = initial_data
        
        return combined_data
    
    # Save results and generate summary files
    def save_results(self, stock_data: pd.DataFrame, timestamp: str = None) -> Dict[str, str]:
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_created = {}
        
        if not stock_data.empty:
            script_dir = Path(__file__).resolve().parent
            output_dir = script_dir.parent / "3_Price_Data"
            output_dir.mkdir(parents=True, exist_ok=True)

            stock_file = output_dir / f"Stock_Price_{timestamp}.csv"
            stock_data.to_csv(stock_file, index=False)
            files_created['stock_data'] = str(stock_file)

            
            # Save failed tickers for future reference
            all_failed = (self.results['failed_tickers'] + 
                         self.results['delisted_tickers'] + 
                         [t for t in self.results['invalid_tickers'] if t])
            
            if all_failed:
                failed_df = pd.DataFrame({
                    'failed_ticker': sorted(set(all_failed)),
                    'reason': 'failed_or_delisted'
                })
                failed_file = output_dir / f"Failed_Tickers_{timestamp}.csv"
                failed_df.to_csv(failed_file, index=False)
                files_created['failed_tickers'] = str(failed_file)
        
        return files_created
    
    def get_summary(self) -> Dict:
        """Generate processing summary"""
        
        total_attempted = (len(self.results['successful_tickers']) + 
                          len(self.results['failed_tickers']) + 
                          len(self.results['delisted_tickers']) + 
                          len(self.results['invalid_tickers']))
        
        return {
            'total_attempted': total_attempted,
            'successful': len(self.results['successful_tickers']),
            'failed': len(self.results['failed_tickers']),
            'delisted': len(self.results['delisted_tickers']),
            'invalid': len(self.results['invalid_tickers']),
            'success_rate': len(self.results['successful_tickers']) / total_attempted * 100 if total_attempted > 0 else 0
        }

def main():
    processor = StockDataProcessor(max_workers=15, delay_between_requests=0.1)

    # Get script directory and parent dir
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    sentiment_path = parent_dir / "2_Cleaned_Data" / "Cleaned_Sentiment_1.csv"
        
    # Generate stock dataset
    stock_data = processor.generate_stock_dataset(sentiment_csv=str(sentiment_path))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files_created = processor.save_results(stock_data, timestamp)
    
    # Print summary
    summary = processor.get_summary()
    print(f"\nProcessing completed:")
    print(f"- Successful tickers: {summary['successful']} ({summary['success_rate']:.1f}%)")
    print(f"- Failed tickers: {summary['failed']}")
    print(f"- Delisted tickers: {summary['delisted']}")
    print(f"- Total stock records: {len(stock_data):,}")
    
    if files_created:
        print(f"\nFiles created:")
        for file_type, filename in files_created.items():
            print(f"- {filename}")
    
    return stock_data

if __name__ == "__main__":
    stock_data = main()
    
    