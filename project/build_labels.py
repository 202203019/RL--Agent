"""
Label generation pipeline for NIFTY 1-second dataset.
Implements triple-barrier labeling with configurable thresholds and timeout.
Output: one labeled Parquet file per day in data/labeled/.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ========= CONFIG =========
INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/labeled")

UP_POINTS = 10
DOWN_POINTS = 7
TIMEOUT_SECONDS = 120
TIE_POLICY = "DOWN_first"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def label_one_day(df: pd.DataFrame, 
                  up_pts: float = UP_POINTS, 
                  down_pts: float = DOWN_POINTS, 
                  timeout: int = TIMEOUT_SECONDS, 
                  tie_policy: str = TIE_POLICY) -> pd.DataFrame:
    """
    Computes triple-barrier labels for a single day.
    
    Logic:
    - Entry price = close_t
    - Scan forward from t+1 up to t+timeout
    - If high >= entry + up_pts -> UP
    - If low <= entry - down_pts -> DOWN
    - If both in same bar -> tie_policy (default DOWN_first)
    - Else -> NONE
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    N = len(df)
    
    # Pre-allocate results
    # 0 = DOWN, 1 = UP, 2 = NONE
    labels = np.full(N, 2, dtype=np.int32)
    label_strs = np.full(N, "NONE", dtype=object)
    event_types = np.full(N, "TIMEOUT", dtype=object)
    time_to_events = np.full(N, float(timeout), dtype=np.float32)
    
    # We use a nested loop here. Since timeout is small (120), 
    # it's roughly 2.7M iterations per day, which Python handles fine in ~0.5s.
    for i in range(N):
        entry = close[i]
        up_barrier = entry + up_pts
        down_barrier = entry - down_pts
        
        # Max look-forward is restricted to the current day (N)
        horizon = min(i + timeout + 1, N)
        
        for j in range(i + 1, horizon):
            h_j = high[j]
            l_j = low[j]
            
            hit_up = (h_j >= up_barrier)
            hit_down = (l_j <= down_barrier)
            
            if hit_up and hit_down:
                if tie_policy == "DOWN_first":
                    labels[i] = 0
                    label_strs[i] = "DOWN"
                    event_types[i] = "DOWN_HIT"
                else:
                    labels[i] = 1
                    label_strs[i] = "UP"
                    event_types[i] = "UP_HIT"
                time_to_events[i] = float(j - i)
                break
                
            elif hit_up:
                labels[i] = 1
                label_strs[i] = "UP"
                event_types[i] = "UP_HIT"
                time_to_events[i] = float(j - i)
                break
                
            elif hit_down:
                labels[i] = 0
                label_strs[i] = "DOWN"
                event_types[i] = "DOWN_HIT"
                time_to_events[i] = float(j - i)
                break
                
    df['label'] = labels
    df['label_str'] = label_strs
    df['event_type'] = event_types
    df['time_to_event_seconds'] = time_to_events
    
    return df

def process_all_days():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("*.parquet"))
    
    if not input_files:
        logger.error(f"No feature files found in {INPUT_DIR}")
        return
        
    logger.info(f"Processing {len(input_files)} files...")
    
    summary_stats = {
        'total_rows': 0,
        'UP': 0,
        'DOWN': 0,
        'NONE': 0,
        'time_to_event_sum_up': 0.0,
        'time_to_event_sum_down': 0.0
    }
    
    for i, file_path in enumerate(tqdm(input_files, desc="Labeling days")):
        try:
            df = pd.read_parquet(file_path)
            labeled_df = label_one_day(df)
            
            # Update stats
            summary_stats['total_rows'] += len(labeled_df)
            counts = labeled_df['label_str'].value_counts()
            summary_stats['UP'] += counts.get('UP', 0)
            summary_stats['DOWN'] += counts.get('DOWN', 0)
            summary_stats['NONE'] += counts.get('NONE', 0)
            
            summary_stats['time_to_event_sum_up'] += labeled_df[labeled_df['label_str'] == 'UP']['time_to_event_seconds'].sum()
            summary_stats['time_to_event_sum_down'] += labeled_df[labeled_df['label_str'] == 'DOWN']['time_to_event_seconds'].sum()
            
            # Save
            output_path = OUTPUT_DIR / file_path.name
            labeled_df.to_parquet(output_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Print Summary Report
    print_summary(summary_stats, len(input_files))

def print_summary(stats, file_count):
    total = stats['total_rows']
    if total == 0:
        print("\nNo data processed.")
        return
        
    up_pct = (stats['UP'] / total) * 100
    dn_pct = (stats['DOWN'] / total) * 100
    no_pct = (stats['NONE'] / total) * 100
    
    avg_time_up = stats['time_to_event_sum_up'] / stats['UP'] if stats['UP'] > 0 else 0
    avg_time_dn = stats['time_to_event_sum_down'] / stats['DOWN'] if stats['DOWN'] > 0 else 0
    
    print("\n" + "="*40)
    print("LABEL GENERATION SUMMARY")
    print("="*40)
    print(f"Total Files Processed:  {file_count}")
    print(f"Total Rows Labeled:     {total:,}")
    print("-"*40)
    print(f"UP Labels:    {stats['UP']:,} ({up_pct:.2f}%)")
    print(f"DOWN Labels:  {stats['DOWN']:,} ({dn_pct:.2f}%)")
    print(f"NONE Labels:  {stats['NONE']:,} ({no_pct:.2f}%)")
    print("-"*40)
    print(f"Avg Time to UP:   {avg_time_up:.2f}s")
    print(f"Avg Time to DOWN: {avg_time_dn:.2f}s")
    print("="*40)

if __name__ == "__main__":
    process_all_days()
