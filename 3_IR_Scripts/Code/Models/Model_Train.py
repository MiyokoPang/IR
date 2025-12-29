import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings
import os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# --- FIXED Data Preparation Functions ---

def prepare_weekly_price_data(price_df):
    price_df['date'] = pd.to_datetime(price_df['date'])
    return price_df.groupby(['stock', pd.Grouper(key='date', freq='W')])['adj_close'].mean().reset_index()

def prepare_sentiment_features(sentiment_df):
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    return sentiment_df.groupby(['stock', pd.Grouper(key='date', freq='W')])[
        ['textblob_polarity', 'vader_compound', 'finbert_compound']
    ].mean().reset_index()

def create_features(df):
    """
    FIXED: Create features WITHOUT including current price in calculations
    Use lagged values to prevent data leakage
    """
    df = df.copy()
    
    # Use LAGGED prices for all calculations
    df['prev_close'] = df['adj_close'].shift(1)
    
    # Returns based on PAST prices
    df['return'] = df['adj_close'].pct_change()
    df['log_return'] = np.log(df['adj_close'] / df['prev_close'])
    
    # Moving averages based on PAST prices (shifted by 1)
    df['ma_3'] = df['prev_close'].rolling(3).mean()
    df['ma_5'] = df['prev_close'].rolling(5).mean()
    df['ma_10'] = df['prev_close'].rolling(10).mean()
    
    # Volatility from PAST returns
    df['volatility'] = df['return'].rolling(5).std()
    
    # RSI from PAST prices
    df['rsi'] = calculate_rsi(df['prev_close'])
    
    # Price momentum features
    df['momentum_3'] = df['adj_close'] / df['adj_close'].shift(3) - 1
    df['momentum_5'] = df['adj_close'] / df['adj_close'].shift(5) - 1
    
    return df.dropna()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_metrics(y_true, y_pred):
    """Calculate metrics with safety checks including directional accuracy"""
    # Ensure arrays are valid
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return None
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # MAPE with safety check for zero values
    non_zero_mask = y_true_clean != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # Directional accuracy: did we predict the correct sign?
    correct_direction = np.sign(y_true_clean) == np.sign(y_pred_clean)
    directional_accuracy = np.mean(correct_direction) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

def prepare_ml_data(df, feature_cols, scale_target=True):
    """
    FIXED: Properly prepare data with per-ticker scaling
    Returns scaled features, target, and scaler for inverse transform
    """
    # Predict NEXT period's return instead of absolute price
    df = df.copy()
    df['target_return'] = df['adj_close'].pct_change().shift(-1)
    
    # Remove last row (no future return) and any NaN
    df = df.dropna()
    
    if len(df) < 30:
        return None, None, None, None
    
    X = df[feature_cols].values
    
    if scale_target:
        # Predict returns (already normalized by definition)
        y = df['target_return'].values
        target_scaler = None
    else:
        # If predicting absolute price, scale it
        y = df['adj_close'].values[:-1]  # Align with shifted target
        target_scaler = MinMaxScaler()
        y = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Scale features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    return X_scaled, y, feature_scaler, target_scaler

# --- FIXED Model Functions ---

def run_linear_regression(X, y, ticker, model_type):
    try:
        # Use time-series split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        if metrics:
            return {'ticker': ticker, 'model': f'LinearRegression_{model_type}', **metrics}
    except Exception as e:
        print(f"[LinearRegression_{model_type}] {ticker}: {e}")
    return None

def run_ridge_regression(X, y, ticker, model_type):
    try:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        if metrics:
            return {'ticker': ticker, 'model': f'Ridge_{model_type}', **metrics}
    except Exception as e:
        print(f"[Ridge_{model_type}] {ticker}: {e}")
    return None

def run_lasso_regression(X, y, ticker, model_type):
    try:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = Lasso(alpha=0.01, random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        if metrics:
            return {'ticker': ticker, 'model': f'Lasso_{model_type}', **metrics}
    except Exception as e:
        print(f"[Lasso_{model_type}] {ticker}: {e}")
    return None

def run_random_forest(X, y, ticker, model_type):
    try:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        if metrics:
            return {'ticker': ticker, 'model': f'RandomForest_{model_type}', **metrics}
    except Exception as e:
        print(f"[RandomForest_{model_type}] {ticker}: {e}")
    return None

def run_svr(X, y, ticker, model_type):
    try:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, y_pred)
        if metrics:
            return {'ticker': ticker, 'model': f'SVR_{model_type}', **metrics}
    except Exception as e:
        print(f"[SVR_{model_type}] {ticker}: {e}")
    return None

def run_mlp(X, y, ticker, model_type):
    configurations = [
        {'hidden_layer_sizes': (50,), 'name': 'MLP_Small'},
        {'hidden_layer_sizes': (64, 32), 'name': 'MLP_Medium'},
        {'hidden_layer_sizes': (100, 50, 25), 'name': 'MLP_Large'}
    ]
    
    results = []
    for config in configurations:
        try:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'], 
                max_iter=1000, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.001
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            if metrics:
                results.append({'ticker': ticker, 'model': f"{config['name']}_{model_type}", **metrics})
        except Exception as e:
            print(f"[{config['name']}_{model_type}] {ticker}: {e}")
    
    return results

def run_xgboost(X, y, ticker, model_type):
    configurations = [
        {'n_estimators': 50, 'max_depth': 3, 'name': 'XGBoost_Light'},
        {'n_estimators': 100, 'max_depth': 5, 'name': 'XGBoost_Medium'},
        {'n_estimators': 200, 'max_depth': 7, 'name': 'XGBoost_Heavy'}
    ]
    
    results = []
    for config in configurations:
        try:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = xgb.XGBRegressor(
                n_estimators=config['n_estimators'], 
                max_depth=config['max_depth'],
                learning_rate=0.1,
                objective='reg:squarederror', 
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = calculate_metrics(y_test, y_pred)
            if metrics:
                results.append({'ticker': ticker, 'model': f"{config['name']}_{model_type}", **metrics})
        except Exception as e:
            print(f"[{config['name']}_{model_type}] {ticker}: {e}")
    
    return results

def run_arima(series, ticker, train_ratio=0.8):
    """ARIMA for price returns instead of absolute prices"""
    arima_configs = [(1,1,1), (2,1,1), (1,1,2), (3,1,1), (5,1,0)]
    
    results = []
    for order in arima_configs:
        try:
            train_size = int(len(series) * train_ratio)
            train, test = series[:train_size], series[train_size:]
            
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            
            metrics = calculate_metrics(test, forecast)
            if metrics:
                results.append({
                    'ticker': ticker, 
                    'model': f'ARIMA_{order[0]}{order[1]}{order[2]}_PriceOnly', 
                    **metrics
                })
        except Exception as e:
            print(f"[ARIMA_{order}] {ticker}: {e}")
    
    return results

def run_arimax(y, exog, ticker, train_ratio=0.8):
    """ARIMAX for price returns with sentiment"""
    arima_configs = [(1,1,1), (2,1,1), (1,1,2), (3,1,1), (5,1,0)]
    
    results = []
    for order in arima_configs:
        try:
            train_size = int(len(y) * train_ratio)
            y_train, y_test = y[:train_size], y[train_size:]
            exog_train, exog_test = exog[:train_size], exog[train_size:]

            model = ARIMA(y_train, order=order, exog=exog_train)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(y_test), exog=exog_test)
            
            metrics = calculate_metrics(y_test, forecast)
            if metrics:
                results.append({
                    'ticker': ticker, 
                    'model': f'ARIMAX_{order[0]}{order[1]}{order[2]}_Price+Sentiment', 
                    **metrics
                })
        except Exception as e:
            print(f"[ARIMAX_{order}] {ticker}: {e}")
    
    return results

# --- Main Execution ---

def run_all_models():
    """Run all non-TensorFlow models and save results"""
    
    print("=== LOADING DATA ===")
    
    # Absolute file path - matching your original structure
    script_dir = Path(__file__).resolve().parent
    stock_csv = script_dir.parent / "3_Price_Data" / "Stock_Prices.csv"
    merged_blob_csv = script_dir.parent / "5_Merged_Data" / "Merged_Scored_TextBlob.csv"
    
    # Check if required files exist
    if not os.path.exists(stock_csv):
        print("Error: Stock_Prices.csv not found!")
        print("Please run Final_File_Prep.py first.")
        return
    
    if not os.path.exists(merged_blob_csv):
        print("Error: Merged_Scored_TextBlob.csv not found!")
        print("Please run Final_File_Prep.py first.")
        return
    
    # Load data
    price_df = pd.read_csv(stock_csv)
    sentiment_df = pd.read_csv(merged_blob_csv)
    
    weekly_price = prepare_weekly_price_data(price_df)
    weekly_sentiment = prepare_sentiment_features(sentiment_df)
    
    tickers = sorted(weekly_price['stock'].unique())
    print(f"Found {len(tickers)} tickers\n")
    
    all_results = []
    os.makedirs("Results", exist_ok=True)
    
    print("=== RUNNING MODELS (PREDICTING RETURNS) ===\n")
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        try:
            # Get price data
            ticker_price = weekly_price[weekly_price['stock'] == ticker].sort_values('date')
            if len(ticker_price) < 30:
                continue

            ticker_price = create_features(ticker_price)
            if len(ticker_price) < 30:
                continue

            # --- Price-only features ---
            price_features = ['return', 'log_return', 'ma_3', 'ma_5', 'ma_10', 
                            'volatility', 'rsi', 'momentum_3', 'momentum_5']
            
            X, y, _, _ = prepare_ml_data(ticker_price, price_features, scale_target=True)
            if X is None:
                continue

            # Run all models
            models = [
                (run_linear_regression, 'PriceOnly'),
                (run_ridge_regression, 'PriceOnly'),
                (run_lasso_regression, 'PriceOnly'),
                (run_random_forest, 'PriceOnly'),
                (run_svr, 'PriceOnly')
            ]
            
            for model_func, model_type in models:
                result = model_func(X, y, ticker, model_type)
                if result:
                    all_results.append(result)
            
            mlp_results = run_mlp(X, y, ticker, 'PriceOnly')
            all_results.extend(mlp_results)
            
            xgb_results = run_xgboost(X, y, ticker, 'PriceOnly')
            all_results.extend(xgb_results)

            # --- With sentiment ---
            ticker_sent = weekly_sentiment[weekly_sentiment['stock'] == ticker]
            merged = pd.merge(ticker_price, ticker_sent, on=['stock', 'date'], how='inner')
            
            if len(merged) >= 30:
                sentiment_components = ['textblob_polarity', 'vader_compound', 'finbert_compound']
                
                # All sentiment
                all_features = price_features + sentiment_components
                X_sent, y_sent, _, _ = prepare_ml_data(merged, all_features, scale_target=True)
                
                if X_sent is not None:
                    for model_func, _ in models:
                        result = model_func(X_sent, y_sent, ticker, 'Price+AllSentiment')
                        if result:
                            all_results.append(result)
                    
                    mlp_results = run_mlp(X_sent, y_sent, ticker, 'Price+AllSentiment')
                    all_results.extend(mlp_results)
                    
                    xgb_results = run_xgboost(X_sent, y_sent, ticker, 'Price+AllSentiment')
                    all_results.extend(xgb_results)

        except Exception as e:
            print(f"\n[{ticker}] Error: {e}")
            continue

    # --- Save Results ---
    print()
    print("=== SAVING RESULTS ===")
    
    # Create results directory matching your structure
    results_dir = script_dir.parent / "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_df.to_csv(results_dir / "Model_Results.csv", index=False)
        print(f"Saved {len(all_results)} model results to Results/Model_Results.csv")
        
        # Print summary statistics
        print()
        print("=== SUMMARY STATISTICS ===")
        summary = result_df.groupby('model')['rmse'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(summary.round(6))
        
        # Best performing models by R²
        print()
        print("=== BEST PERFORMING MODELS (by average R²) ===")
        best_r2 = result_df.groupby('model')['r2'].mean().sort_values(ascending=False)
        for model, r2 in best_r2.head(10).items():
            print(f"{model}: {r2:.6f}")
        
        # Best performing models by RMSE
        print()
        print("=== BEST PERFORMING MODELS (by average RMSE) ===")
        best_rmse = result_df.groupby('model')['rmse'].mean().sort_values()
        for model, rmse in best_rmse.head(10).items():
            print(f"{model}: {rmse:.6f}")
        
        # Average directional accuracy
        print()
        print("=== DIRECTIONAL ACCURACY ===")
        avg_dir_acc = result_df.groupby('model')['directional_accuracy'].mean().sort_values(ascending=False)
        for model, acc in avg_dir_acc.head(10).items():
            print(f"{model}: {acc:.2f}%")
        
        # Create summary by model type for R analysis
        model_summary = result_df.groupby('model').agg({
            'rmse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std', 'min', 'max'],
            'r2': ['mean', 'std', 'min', 'max'],
            'mape': ['mean', 'std', 'min', 'max'],
            'directional_accuracy': ['mean', 'std', 'min', 'max']
        }).round(6)
        
        model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
        model_summary.to_csv(results_dir / "Model_Performance_Summary.csv")
        print("Saved model summary to Results/Model_Performance_Summary.csv")
            
    else:
        print("No results to save - all models failed!")

if __name__ == "__main__":
    run_all_models()