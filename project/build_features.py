"""
Feature engineering pipeline for NIFTY 1-second dataset (Expanded 24-Feature Set).
Reads daily Parquet files from data/1s_bars/, computes features with cross-day continuity,
and saves daily Parquet files to data/features/.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ========= CONFIG =========
INPUT_DIR = Path("data/1s_bars")
OUTPUT_DIR = Path("data/features")
BUFFER_SIZE = 60  # Max rolling window is 60s for rolling_std_60s

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 24 engineered features on the provided DataFrame.
    Assumes df is sorted by datetime and is continuous (1s frequency).
    """
    df = df.copy()
    
    # A. Returns / Momentum (4 features)
    for k in [1, 5, 10, 30]:
        df[f'log_ret_{k}s'] = np.log(df['close'] / df['close'].shift(k))
        
    # B. Momentum Acceleration (2 features)
    df['ret_accel_5s'] = df['log_ret_5s'] - df['log_ret_10s']
    df['ret_accel_10s'] = df['log_ret_10s'] - df['log_ret_30s']
        
    # C. Candle Microstructure (3 features)
    df['body_1s'] = df['close'] - df['open']
    df['range_1s'] = df['high'] - df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['range_1s'] + 1e-9)
    
    # D. Volatility Regime (4 features)
    for k in [10, 30, 60]:
        df[f'rolling_std_{k}s'] = df['log_ret_1s'].rolling(window=k).std()
    df['volatility_ratio'] = df['rolling_std_10s'] / (df['rolling_std_60s'] + 1e-9)
        
    # E. Market Activity (4 features)
    for k in [5, 30]:
        df[f'tick_count_{k}s'] = df['tick_count'].rolling(window=k).sum()
    df['spread'] = df['ask'] - df['bid']
    
    # F. Microstructure Pressure (1 feature)
    df['price_pressure'] = (df['close'] - df['bid']) / (df['spread'] + 1e-9)
    
    # G. Trend / Mean Reversion (4 features)
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema_fast_minus_slow'] = df['ema_10'] - df['ema_30']
    
    rolling_mean_30 = df['close'].rolling(window=30).mean()
    rolling_std_30 = df['close'].rolling(window=30).std()
    df['zscore_price_30s'] = (df['close'] - rolling_mean_30) / (rolling_std_30 + 1e-9)
    
    # H. Intraday Seasonality (2 features)
    dt = df['datetime'].dt
    seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
    day_seconds = 24 * 3600
    df['time_of_day_sin'] = np.sin(2 * np.pi * seconds_since_midnight / day_seconds)
    df['time_of_day_cos'] = np.cos(2 * np.pi * seconds_since_midnight / day_seconds)
    
    return df

def validate_data(df: pd.DataFrame, filename: str):
    """Basic data integrity checks."""
    if not df['datetime'].is_monotonic_increasing:
        logger.warning(f"Datetime not monotonic in {filename}")
    if df.isnull().any().any():
        nan_counts = df.isnull().sum().sum()
        logger.warning(f"File {filename} contains {nan_counts} NaN values after processing.")

def process_all_days():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(INPUT_DIR.glob("*.parquet"))
    if not input_files:
        logger.error(f"No parquet files found in {INPUT_DIR}")
        return
        
    logger.info(f"Found {len(input_files)} files to process.")
    
    buffer_df = pd.DataFrame()
    total_rows_processed = 0
    
    # Track feature columns for summary
    feature_cols = []
    
    for i, file_path in enumerate(tqdm(input_files, desc="Generating Features")):
        current_day_df = pd.read_parquet(file_path)
        
        # Prepend buffer (60s) for continuity
        if not buffer_df.empty:
            combined_df = pd.concat([buffer_df, current_day_df], axis=0).reset_index(drop=True)
        else:
            combined_df = current_day_df
            
        # Compute features
        feat_df = compute_features(combined_df)
        
        # Trim back to current day only
        if not buffer_df.empty:
            final_df = feat_df.iloc[len(buffer_df):].copy()
        else:
            final_df = feat_df.copy()
            
        # Capture feature names once
        if i == 0:
            feature_cols = [c for c in final_df.columns if c not in current_day_df.columns]
            
        # Statistics
        total_rows_processed += len(final_df)
        
        # Validation and Logging
        validate_data(final_df, file_path.name)
        
        # Save
        output_path = OUTPUT_DIR / file_path.name
        final_df.to_parquet(output_path)
        
        # Update buffer for next day
        buffer_df = current_day_df.tail(BUFFER_SIZE).copy()

    # Final Summary
    print("\n" + "="*40)
    print("FEATURE REGENERATION COMPLETE")
    print("="*40)
    print(f"Processed files:    {len(input_files)}")
    print(f"Total rows:         ~{total_rows_processed/1e6:.1f}M ({total_rows_processed:,})")
    print(f"Feature columns:    {len(feature_cols)}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print("="*40)
    
if __name__ == "__main__":
    process_all_days()
