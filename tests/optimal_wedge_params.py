#!/usr/bin/env python3
"""
Optimal Wedge Parameters Finder
------------------------------

This script quickly determines the optimal parameters for wedge pattern detection
and outputs just the best settings.

Based on extensive testing, this script provides the best parameter combinations
for maximizing pattern detection across different securities and timeframes.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

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

# Test stocks that represent different market segments
TEST_STOCKS = ['AAPL', 'SPY', 'NVDA', 'JPM', 'XOM']

# Parameter sets to evaluate (these are pre-selected combinations that generally perform well)
PARAMETER_SETS = [
    # (window_size, extrema_order, slope_tolerance, top_k)
    (50, 3, 0.10, 8),   # balanced general-purpose settings
    (60, 4, 0.12, 10),  # more aggressive pattern detection
    (40, 3, 0.10, 5),   # more conservative pattern detection
    (80, 5, 0.15, 10),  # for longer-term patterns
    (30, 2, 0.08, 5),   # for shorter-term patterns
]

def detect_wedge_patterns(prices, window_size, extrema_order, slope_tol, top_k):
    """Detect wedge patterns with the given parameters."""
    if prices is None or len(prices) <= window_size:
        return []
    
    # Handle NaN values - fix the .isna().any() issue
    if prices.isna().any().any() if hasattr(prices.isna(), 'any') else prices.isna().any():
        prices = prices.dropna()
        if len(prices) <= window_size:
            return []
    
    # Process price data
    price_array = np.asarray(prices.values, dtype=float)
    
    try:
        # Compute matrix profile
        profile_distances, profile_indices = compute_matrix_profile(price_array, window_size)
        
        # Extract motif candidates
        motifs = extract_motif_candidates(profile_distances, profile_indices, window_size, k=top_k)
        if not motifs:
            return []
        
        # Detect patterns from motifs
        patterns = []
        used_indices = set()
        
        for (idx1, idx2, dist) in motifs:
            for idx in (idx1, idx2):
                if idx in used_indices or idx + window_size > len(prices):
                    continue
                
                subseq = price_array[idx: idx + window_size]
                peaks_idx, troughs_idx = detect_extrema(
                    subseq, order=min(extrema_order, window_size // 3))
                pat_type, _, _ = check_wedge_pattern(
                    subseq, peaks_idx, troughs_idx, slope_tolerance=slope_tol)
                
                if pat_type is not None:
                    patterns.append({'type': pat_type})
                    used_indices.add(idx)
        
        return patterns
    except Exception as e:
        print(f"Error: {e}")
        return []

def evaluate_parameter_set(prices, params):
    """Evaluate a parameter set against price data."""
    window_size, extrema_order, slope_tol, top_k = params
    patterns = detect_wedge_patterns(prices, window_size, extrema_order, slope_tol, top_k)
    return len(patterns)

def find_optimal_parameters():
    """Find the optimal parameters across test stocks."""
    print("Determining optimal wedge pattern parameters...")
    
    results = {}
    
    # Test each parameter set on different stocks
    for symbol in TEST_STOCKS:
        print(f"Testing {symbol}...")
        # Get price data - use 1 year of daily data as baseline
        prices = fetch_price_data(symbol, period="1y", interval="1d")
        if prices is None:
            print(f"Could not fetch data for {symbol}, skipping.")
            continue
        
        # Test each parameter set
        param_scores = {}
        for params in PARAMETER_SETS:
            pattern_count = evaluate_parameter_set(prices, params)
            param_scores[params] = pattern_count
            print(f"  Parameters {params}: {pattern_count} patterns")
        
        # Store best parameters for this stock
        best_params = max(param_scores.items(), key=lambda x: x[1])
        results[symbol] = best_params
    
    # Find overall best parameters (most consistent across stocks)
    param_counts = {params: 0 for params in PARAMETER_SETS}
    for symbol, (params, count) in results.items():
        param_counts[params] += count
    
    overall_best_params = max(param_counts.items(), key=lambda x: x[1])
    
    # Print results
    print("\n=== OPTIMAL PARAMETER RESULTS ===")
    
    print("\nBest parameters by stock:")
    for symbol, (params, count) in results.items():
        window_size, extrema_order, slope_tol, top_k = params
        print(f"  {symbol}: Window Size={window_size}, Extrema Order={extrema_order}, " 
              f"Slope Tolerance={slope_tol}, Top K={top_k} ({count} patterns)")
    
    print("\nOVERALL OPTIMAL PARAMETERS:")
    window_size, extrema_order, slope_tol, top_k = overall_best_params[0]
    print(f"Window Size: {window_size}")
    print(f"Extrema Order: {extrema_order}")
    print(f"Slope Tolerance: {slope_tol}")
    print(f"Top K: {top_k}")
    print(f"Total patterns detected: {overall_best_params[1]}")

if __name__ == "__main__":
    start_time = time.time()
    find_optimal_parameters()
    print(f"\nParameter optimization completed in {time.time() - start_time:.2f} seconds")