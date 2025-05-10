#!/usr/bin/env python3
"""
Optimal Head and Shoulders Pattern Parameters Finder
---------------------------------------------------

This script systematically tests different parameter combinations for the 
head and shoulders pattern detection algorithm to find the optimal settings 
that yield the most pattern detections.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import itertools
from tqdm import tqdm

# Add parent directory to path to import helper modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import streamlit_helpers functions
from streamlit_helpers import (
    fetch_price_data,
    compute_matrix_profile,
    detect_extrema,
    check_head_and_shoulders,
    extract_motif_candidates
)

# Test stocks representing different market sectors
TEST_STOCKS = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'NVDA', 'JPM', 'XOM']

# Parameter sets to evaluate
PARAMETER_SETS = [
    # (window_length, order, shoulder_tol, trough_tol, top_k)
    (30, 3, 0.10, 0.20, 3),  # Default parameters
    (50, 3, 0.10, 0.20, 5),  # Increased window and motifs
    (40, 2, 0.05, 0.10, 3),  # Lower tolerance (stricter)
    (30, 4, 0.15, 0.25, 3),  # Higher tolerance (more lenient)
    (60, 5, 0.10, 0.20, 8),  # Larger window, more extrema points
    (20, 2, 0.10, 0.20, 3),  # Smaller window (shorter patterns)
    (40, 4, 0.12, 0.18, 5),  # Balanced approach
]

def detect_head_shoulders_patterns(prices, window_length, order, shoulder_tol, trough_tol, top_k):
    """
    Detect head and shoulders patterns using the given parameters.
    
    Parameters:
    - prices: Price series data
    - window_length: Window length for matrix profile
    - order: Extrema detection order
    - shoulder_tol: Shoulder height tolerance
    - trough_tol: Trough alignment tolerance
    - top_k: Number of motifs to extract
    
    Returns:
    - List of pattern indices
    """
    if prices is None or len(prices) <= window_length:
        return []
    
    # Handle NaN values
    if hasattr(prices.isna(), 'any'):
        if prices.isna().any().any():
            prices = prices.dropna()
            if len(prices) <= window_length:
                return []
    elif prices.isna().any():
        prices = prices.dropna()
        if len(prices) <= window_length:
            return []
    
    # Convert to numpy array
    price_array = np.asarray(prices.values, dtype=float)
    
    try:
        # Compute matrix profile
        mp = compute_matrix_profile(price_array, window_length)
        
        # Extract motif candidates
        motifs = extract_motif_candidates(mp, window_length, top_k=top_k)
        
        # If no motifs found, return empty list
        if not motifs:
            return []
        
        # Check each motif for head and shoulders pattern
        pattern_indices = []
        for idx1, idx2 in motifs:
            # Ensure indices are within bounds
            if idx1 >= len(prices) or idx2 >= len(prices) or idx1 + window_length > len(prices) or idx2 + window_length > len(prices):
                continue
            
            # Check first subsequence
            segment1 = prices.iloc[idx1:idx1+window_length]
            peaks1, troughs1 = detect_extrema(segment1, order=min(order, window_length // 3))
            is_pattern1 = check_head_and_shoulders(segment1, peaks1, troughs1, 
                                                 shoulder_tol=shoulder_tol, 
                                                 trough_tol=trough_tol)
            
            # Check second subsequence
            segment2 = prices.iloc[idx2:idx2+window_length]
            peaks2, troughs2 = detect_extrema(segment2, order=min(order, window_length // 3))
            is_pattern2 = check_head_and_shoulders(segment2, peaks2, troughs2,
                                                 shoulder_tol=shoulder_tol,
                                                 trough_tol=trough_tol)
            
            if is_pattern1:
                pattern_indices.append(idx1)
            if is_pattern2:
                pattern_indices.append(idx2)
                
        return pattern_indices
        
    except Exception as e:
        print(f"Error in pattern detection: {e}")
        return []

def evaluate_parameter_set(prices, params):
    """Evaluate a parameter set against price data."""
    window_length, order, shoulder_tol, trough_tol, top_k = params
    patterns = detect_head_shoulders_patterns(
        prices, window_length, order, shoulder_tol, trough_tol, top_k)
    return len(patterns)

def find_optimal_parameters():
    """Find the optimal parameters across test stocks."""
    print("Determining optimal head and shoulders pattern parameters...")
    
    results = {}
    
    # Test each parameter set on all stocks
    for symbol in TEST_STOCKS:
        print(f"Testing {symbol}...")
        
        # Try to get data for multiple timeframes
        timeframes = [
            ("1d", "2y"),  # Daily data for 2 years
            ("1d", "1y"),  # Daily data for 1 year
            ("1h", "60d")  # Hourly data for 60 days
        ]
        
        prices = None
        for interval, period in timeframes:
            prices = fetch_price_data(symbol, interval=interval, period=period)
            if prices is not None and len(prices) > 60:  # Ensure enough data points
                print(f"  Using {interval} data for {period}")
                break
        
        if prices is None:
            print(f"Could not fetch data for {symbol}, skipping.")
            continue
        
        # Test each parameter set
        param_scores = {}
        for params in tqdm(PARAMETER_SETS, desc=f"Parameters for {symbol}"):
            pattern_count = evaluate_parameter_set(prices, params)
            param_scores[params] = pattern_count
            print(f"  Parameters {params}: {pattern_count} patterns")
        
        # Store best parameters for this stock
        best_params = max(param_scores.items(), key=lambda x: x[1])
        results[symbol] = best_params
    
    # Find overall best parameters
    param_counts = {params: 0 for params in PARAMETER_SETS}
    for symbol, (params, count) in results.items():
        param_counts[params] += count
    
    overall_best_params = max(param_counts.items(), key=lambda x: x[1])
    
    # Print results
    print("\n=== OPTIMAL PARAMETER RESULTS ===")
    
    print("\nBest parameters by stock:")
    for symbol, (params, count) in results.items():
        window_length, order, shoulder_tol, trough_tol, top_k = params
        print(f"  {symbol}: Window Length={window_length}, Order={order}, "
              f"Shoulder Tol={shoulder_tol}, Trough Tol={trough_tol}, "
              f"Top K={top_k} ({count} patterns)")
    
    print("\nOVERALL OPTIMAL PARAMETERS:")
    window_length, order, shoulder_tol, trough_tol, top_k = overall_best_params[0]
    print(f"Window Length: {window_length}")
    print(f"Extrema Detection Order: {order}")
    print(f"Shoulder Height Tolerance: {shoulder_tol}")
    print(f"Trough Alignment Tolerance: {trough_tol}")
    print(f"Top K: {top_k}")
    print(f"Total patterns detected: {overall_best_params[1]}")

if __name__ == "__main__":
    start_time = time.time()
    find_optimal_parameters()
    print(f"\nParameter optimization completed in {time.time() - start_time:.2f} seconds")