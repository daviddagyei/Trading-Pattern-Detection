#!/usr/bin/env python3
"""
Parameter Tuning Script for Wedge Pattern Detection
---------------------------------------------------

This script systematically tests different parameter combinations for the wedge pattern
detection algorithm to find the optimal settings that yield the most pattern detections.

The script will:
1. Download price data for various stocks across different timeframes
2. Test combinations of window size, extrema order, and slope tolerance
3. Count the number of wedge patterns detected with each parameter combination
4. Report the best parameter combinations that yield the most pattern detections
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add parent directory to path to import helper modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import wedges_helpers functions
from wedges_helpers import (
    fetch_price_data,
    compute_matrix_profile,
    extract_motif_candidates,
    detect_extrema,
    check_wedge_pattern
)

# Define stock symbols to test with
# Mix of large caps, mid caps, volatile stocks, etc.
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Large tech 
    'SPY', 'QQQ', 'IWM',                      # ETFs
    'NVDA', 'AMD', 'INTC',                    # Semiconductors
    'JPM', 'BAC', 'GS',                       # Banking
    'XOM', 'CVX', 'BP',                       # Energy
]

# Define time frames to test with
TIME_FRAMES = [
    # (interval, period)
    ('1h', '6mo'),   # 1-hour data for 6 months
    ('1d', '1y'),    # Daily data for 1 year
    ('1d', '2y'),    # Daily data for 2 years
    ('1wk', '5y'),   # Weekly data for 5 years
]

# Parameter ranges to test
WINDOW_SIZES = [20, 30, 40, 50, 60, 80, 100]
EXTREMA_ORDERS = [2, 3, 4, 5, 6]
SLOPE_TOLERANCES = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
TOP_K_VALUES = [3, 5, 8, 10]

# Results container
results = []

def detect_patterns_with_params(prices, window_size, extrema_order, slope_tol, top_k):
    """
    Detect wedge patterns using the specified parameters.
    
    Returns:
        List of detected patterns
    """
    # Skip if we don't have enough data points
    if len(prices) <= window_size:
        return []
    
    # Check for NaN values and handle them
    if prices.isna().any():
        prices = prices.dropna()
        if len(prices) <= window_size:
            return []
    
    # Convert to numpy array
    price_array = np.asarray(prices.values, dtype=float)
    
    # Compute matrix profile
    try:
        profile_distances, profile_indices = compute_matrix_profile(price_array, window_size)
        
        # Extract motif candidates
        motifs = extract_motif_candidates(profile_distances, profile_indices, window_size, k=top_k)
        
        # If no motifs found, return empty list
        if not motifs:
            return []
        
        # Analyze each motif pair for wedge patterns
        patterns = []
        used_indices = set()
        
        # Loop through each motif pair (each pair yields two subsequences)
        for (idx1, idx2, dist) in motifs:
            for idx in (idx1, idx2):
                if idx in used_indices:
                    continue
                
                # Ensure indices are within bounds
                if idx + window_size > len(prices):
                    continue
                
                subseq = price_array[idx: idx + window_size]
                
                # Detect extrema in this subsequence
                peaks_idx, troughs_idx = detect_extrema(subseq, order=min(extrema_order, window_size // 3))
                
                # Classify pattern type via trendline slope analysis
                pat_type, slope_peak, slope_trough = check_wedge_pattern(
                    subseq, peaks_idx, troughs_idx, slope_tolerance=slope_tol
                )
                
                if pat_type is not None:
                    patterns.append({
                        'type': pat_type,
                        'start_idx': idx,
                        'end_idx': idx + window_size - 1
                    })
                    used_indices.add(idx)
        
        return patterns
    except Exception as e:
        # If any error occurs, return empty list
        print(f"Error in pattern detection: {e}")
        return []

def test_parameters(symbol, interval, period):
    """Test all parameter combinations for a given stock and timeframe."""
    # Fetch price data
    print(f"Fetching data for {symbol} ({interval}, {period})...")
    prices = fetch_price_data(symbol=symbol, interval=interval, period=period)
    
    if prices is None or len(prices) == 0:
        print(f"No data returned for {symbol} with {interval} interval and {period} period.")
        return []
    
    # Store results for this stock and timeframe
    stock_results = []
    
    # Iterate through parameter combinations
    param_combinations = list(itertools.product(
        WINDOW_SIZES, EXTREMA_ORDERS, SLOPE_TOLERANCES, TOP_K_VALUES
    ))
    
    for window_size, extrema_order, slope_tol, top_k in tqdm(param_combinations, 
                                                             desc=f"{symbol} ({interval}, {period})"):
        # Skip if window size is too large for the data
        if window_size >= len(prices):
            continue
        
        # Detect patterns with current parameter combination
        patterns = detect_patterns_with_params(
            prices, window_size, extrema_order, slope_tol, top_k
        )
        
        # Store result
        rising_count = sum(1 for p in patterns if p['type'] == 'Rising Wedge')
        falling_count = sum(1 for p in patterns if p['type'] == 'Falling Wedge')
        total_count = rising_count + falling_count
        
        stock_results.append({
            'symbol': symbol,
            'interval': interval,
            'period': period,
            'window_size': window_size,
            'extrema_order': extrema_order,
            'slope_tolerance': slope_tol,
            'top_k': top_k,
            'rising_wedges': rising_count,
            'falling_wedges': falling_count,
            'total_patterns': total_count
        })
    
    return stock_results

def process_stock(args):
    """Process a single stock - for multiprocessing."""
    symbol, interval, period = args
    return test_parameters(symbol, interval, period)

def run_parameter_tuning():
    """Run the parameter tuning process and analyze results."""
    start_time = time.time()
    print("Starting parameter tuning for wedge pattern detection...")
    
    # Create all combinations of stocks and timeframes to test
    all_combinations = list(itertools.product(SYMBOLS, 
                                             [tf[0] for tf in TIME_FRAMES],
                                             [tf[1] for tf in TIME_FRAMES]))
    
    # Use multiprocessing to speed up testing
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        all_results = pool.map(process_stock, all_combinations)
    
    # Flatten results list
    flat_results = [item for sublist in all_results for item in sublist]
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(flat_results)
    
    print("\n--- Parameter Tuning Complete ---")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Tested {len(SYMBOLS)} stocks with {len(TIME_FRAMES)} timeframes")
    print(f"Tested {len(WINDOW_SIZES) * len(EXTREMA_ORDERS) * len(SLOPE_TOLERANCES) * len(TOP_K_VALUES)} parameter combinations")
    
    # Save results to CSV
    results_df.to_csv('wedge_parameter_tuning_results.csv', index=False)
    print("Results saved to wedge_parameter_tuning_results.csv")
    
    # Analyze results to find best parameter combinations
    analyze_results(results_df)

def analyze_results(results_df):
    """Analyze results to find the best parameter combinations."""
    print("\n--- Best Parameter Combinations ---")
    
    # Group by parameter combinations and calculate average pattern count
    param_groups = results_df.groupby(['window_size', 'extrema_order', 'slope_tolerance', 'top_k'])
    avg_patterns = param_groups['total_patterns'].mean().reset_index()
    avg_patterns = avg_patterns.sort_values('total_patterns', ascending=False)
    
    # Print top 10 parameter combinations
    print("\nTop 10 parameter combinations by average pattern count:")
    for i, row in avg_patterns.head(10).iterrows():
        print(f"Rank {i+1}: Window Size: {row['window_size']}, "
              f"Extrema Order: {row['extrema_order']}, "
              f"Slope Tolerance: {row['slope_tolerance']:.2f}, "
              f"Top K: {row['top_k']}, "
              f"Avg Patterns: {row['total_patterns']:.2f}")
    
    # Find best parameters for each stock
    print("\nBest parameters for each stock:")
    for symbol in SYMBOLS:
        stock_data = results_df[results_df['symbol'] == symbol]
        if not stock_data.empty:
            best_row = stock_data.loc[stock_data['total_patterns'].idxmax()]
            print(f"{symbol}: Window Size: {best_row['window_size']}, "
                  f"Extrema Order: {best_row['extrema_order']}, "
                  f"Slope Tolerance: {best_row['slope_tolerance']:.2f}, "
                  f"Top K: {best_row['top_k']}, "
                  f"Total Patterns: {best_row['total_patterns']}")
    
    # Find best parameters for each timeframe
    print("\nBest parameters for each timeframe:")
    for interval, period in TIME_FRAMES:
        timeframe_data = results_df[(results_df['interval'] == interval) & 
                                   (results_df['period'] == period)]
        if not timeframe_data.empty:
            best_row = timeframe_data.loc[timeframe_data['total_patterns'].idxmax()]
            print(f"{interval} {period}: Window Size: {best_row['window_size']}, "
                  f"Extrema Order: {best_row['extrema_order']}, "
                  f"Slope Tolerance: {best_row['slope_tolerance']:.2f}, "
                  f"Top K: {best_row['top_k']}, "
                  f"Total Patterns: {best_row['total_patterns']}")
    
    # Overall best parameters (average across all stocks and timeframes)
    best_overall = avg_patterns.iloc[0]
    print("\nBEST OVERALL PARAMETERS:")
    print(f"Window Size: {best_overall['window_size']}")
    print(f"Extrema Order: {best_overall['extrema_order']}")
    print(f"Slope Tolerance: {best_overall['slope_tolerance']:.2f}")
    print(f"Top K: {best_overall['top_k']}")
    print(f"Average Pattern Count: {best_overall['total_patterns']:.2f}")
    
    # Generate visualizations
    print("\nGenerating parameter impact visualizations...")
    visualize_parameter_impact(results_df)

def visualize_parameter_impact(results_df):
    """Create visualizations showing the impact of each parameter on pattern detection."""
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Impact on Wedge Pattern Detection', fontsize=16)
    
    # Impact of window size
    window_impact = results_df.groupby('window_size')['total_patterns'].mean()
    axs[0, 0].bar(window_impact.index.astype(str), window_impact.values)
    axs[0, 0].set_title('Impact of Window Size')
    axs[0, 0].set_xlabel('Window Size')
    axs[0, 0].set_ylabel('Average Pattern Count')
    
    # Impact of extrema order
    extrema_impact = results_df.groupby('extrema_order')['total_patterns'].mean()
    axs[0, 1].bar(extrema_impact.index.astype(str), extrema_impact.values)
    axs[0, 1].set_title('Impact of Extrema Order')
    axs[0, 1].set_xlabel('Extrema Order')
    axs[0, 1].set_ylabel('Average Pattern Count')
    
    # Impact of slope tolerance
    slope_impact = results_df.groupby('slope_tolerance')['total_patterns'].mean()
    axs[1, 0].bar([f"{x:.2f}" for x in slope_impact.index], slope_impact.values)
    axs[1, 0].set_title('Impact of Slope Tolerance')
    axs[1, 0].set_xlabel('Slope Tolerance')
    axs[1, 0].set_ylabel('Average Pattern Count')
    
    # Impact of top K
    topk_impact = results_df.groupby('top_k')['total_patterns'].mean()
    axs[1, 1].bar(topk_impact.index.astype(str), topk_impact.values)
    axs[1, 1].set_title('Impact of Top K Value')
    axs[1, 1].set_xlabel('Top K')
    axs[1, 1].set_ylabel('Average Pattern Count')
    
    plt.tight_layout()
    plt.savefig('parameter_impact_visualization.png')
    print("Visualization saved as parameter_impact_visualization.png")

if __name__ == "__main__":
    # Check if we want to run a simplified test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running quick test with reduced parameter sets...")
        # Use smaller sets for quick testing
        SYMBOLS = SYMBOLS[:3]
        TIME_FRAMES = TIME_FRAMES[:2]
        WINDOW_SIZES = [30, 50, 80]
        EXTREMA_ORDERS = [2, 4]
        SLOPE_TOLERANCES = [0.1, 0.15]
        TOP_K_VALUES = [5, 8]
    
    run_parameter_tuning()