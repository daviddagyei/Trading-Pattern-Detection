#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import functions from double_top_bottom.py
from double_top_bottom import (fetch_price_data, compute_matrix_profile, 
                               extract_motif_candidates, detect_extrema, 
                               check_double_top_bottom, visualize_double_patterns)

def patched_compute_matrix_profile(price_series: pd.Series, window: int):
    """
    A patched version of compute_matrix_profile that properly handles DataFrame input.
    """
    # Make sure price_series is a 1D array 
    if isinstance(price_series, pd.DataFrame):
        # If it's a DataFrame, extract the first column as a Series
        values = price_series.iloc[:, 0].values
    elif isinstance(price_series, pd.Series):
        # If it's a Series, convert to numpy array
        values = price_series.values
    else:
        # If it's already a numpy array
        values = np.array(price_series)
        
    # Ensure it's 1D
    if values.ndim > 1:
        values = values.flatten()
        
    # Convert to float type for STUMPY
    values = np.array(values, dtype=float)
    
    # Compute the matrix profile
    import stumpy
    mp = stumpy.stump(values, m=window)
    profile = mp[:, 0]
    profile_index = mp[:, 1].astype(int)
    
    return profile, profile_index

def test_with_real_data():
    """
    Test the double top/bottom pattern detection with real stock data from yfinance.
    """
    print("\n=== Testing with real stock data ===")
    
    # Fetch some historical data
    symbol = "AAPL"  # Apple stock as an example
    period = "1y"    # 1 year of data
    interval = "1d"  # daily data
    
    print(f"Fetching {period} of {interval} data for {symbol}...")
    df = fetch_price_data(symbol, period, interval)
    
    if df.empty:
        print("Failed to fetch data. Check your internet connection.")
        return
    
    print(f"Successfully fetched {len(df)} data points.")
    
    # Use closing prices for pattern detection
    price_series = df['Close']
    
    # Parameters for pattern detection
    window = min(20, len(price_series) // 10)  # Dynamic window size based on data length
    top_k = 3    # Look for top 3 motifs
    
    print(f"Using window size of {window} for pattern detection")
    
    # Compute matrix profile - using our patched version
    print("Computing matrix profile...")
    profile, profile_index = patched_compute_matrix_profile(price_series, window)
    
    # Extract motif candidates
    print("Extracting motif candidates...")
    motifs = extract_motif_candidates(profile, profile_index, window, top_k)
    
    if not motifs:
        print("No repeating patterns (motifs) found.")
        return
    
    print(f"Found {len(motifs)} motif pairs.")
    
    # Process each motif to identify double top/bottom patterns
    patterns = []
    
    for i, (idx1, idx2) in enumerate(motifs):
        print(f"\nAnalyzing motif pair {i+1}:")
        
        # Extract subsequences
        subseq1 = price_series.iloc[idx1:idx1+window].values
        subseq2 = price_series.iloc[idx2:idx2+window].values
        
        # Detect extrema in subsequences
        peaks1, troughs1 = detect_extrema(subseq1)
        peaks2, troughs2 = detect_extrema(subseq2)
        
        # Check for patterns in both subsequences
        result1 = check_double_top_bottom(subseq1, peaks1, troughs1)
        result2 = check_double_top_bottom(subseq2, peaks2, troughs2)
        
        # Store detected patterns
        if result1:
            pattern_type, key_points = result1
            print(f"Found {pattern_type} pattern in first subsequence starting at index {idx1}")
            patterns.append((idx1, pattern_type, key_points))
        
        if result2:
            pattern_type, key_points = result2
            print(f"Found {pattern_type} pattern in second subsequence starting at index {idx2}")
            patterns.append((idx2, pattern_type, key_points))
    
    # Visualize all detected patterns
    if patterns:
        print(f"\nFound {len(patterns)} double top/bottom patterns. Visualizing...")
        visualize_double_patterns(price_series, patterns, window)
    else:
        print("\nNo double top/bottom patterns detected in the motifs.")

def generate_synthetic_double_top(length=100, noise_level=0.05):
    """Generate synthetic data with a double top pattern."""
    x = np.linspace(0, 1, length)
    
    # Base trend: slight upward
    base = 100 + 10 * x
    
    # Add two peaks
    peak1_center = 0.25
    peak2_center = 0.75
    peak_width = 0.1
    
    # Create peaks using Gaussian curves
    peak1 = 15 * np.exp(-((x - peak1_center) ** 2) / (2 * peak_width ** 2))
    peak2 = 15 * np.exp(-((x - peak2_center) ** 2) / (2 * peak_width ** 2))
    
    # Combine base with peaks
    y = base + peak1 + peak2
    
    # Add random noise
    noise = np.random.normal(0, noise_level * np.max(y), length)
    y += noise
    
    # Create a datetime index (one day per point)
    start_date = datetime.now() - timedelta(days=length)
    dates = [start_date + timedelta(days=i) for i in range(length)]
    
    # Return as pandas Series
    return pd.Series(y, index=dates)

def generate_synthetic_double_bottom(length=100, noise_level=0.05):
    """Generate synthetic data with a double bottom pattern."""
    x = np.linspace(0, 1, length)
    
    # Base trend: slight downward
    base = 100 - 10 * x
    
    # Add two troughs
    trough1_center = 0.25
    trough2_center = 0.75
    trough_width = 0.1
    
    # Create troughs using inverted Gaussian curves
    trough1 = -15 * np.exp(-((x - trough1_center) ** 2) / (2 * trough_width ** 2))
    trough2 = -15 * np.exp(-((x - trough2_center) ** 2) / (2 * trough_width ** 2))
    
    # Combine base with troughs
    y = base + trough1 + trough2
    
    # Add random noise
    noise = np.random.normal(0, noise_level * np.max(y), length)
    y += noise
    
    # Create a datetime index (one day per point)
    start_date = datetime.now() - timedelta(days=length)
    dates = [start_date + timedelta(days=i) for i in range(length)]
    
    # Return as pandas Series
    return pd.Series(y, index=dates)

def test_with_synthetic_data():
    """
    Test the double top/bottom pattern detection with synthetic data.
    """
    print("\n=== Testing with synthetic data ===")
    
    # Generate synthetic data
    double_top_series = generate_synthetic_double_top(length=100, noise_level=0.03)
    double_bottom_series = generate_synthetic_double_bottom(length=100, noise_level=0.03)
    
    # Parameters for pattern detection
    window = 50  # Window size for pattern detection (adjust based on pattern size)
    
    print("\nTesting double top detection...")
    # Process double top pattern
    peaks, troughs = detect_extrema(double_top_series.values)
    result = check_double_top_bottom(double_top_series.values, peaks, troughs)
    
    if result:
        pattern_type, key_points = result
        print(f"Successfully detected {pattern_type} pattern in synthetic data")
        
        # Visualize the pattern
        patterns = [(0, pattern_type, key_points)]
        visualize_double_patterns(double_top_series, patterns, len(double_top_series))
    else:
        print("Failed to detect double top pattern in synthetic data")
        plt.figure(figsize=(10, 5))
        plt.plot(double_top_series)
        plt.title("Synthetic Double Top Pattern (Not Detected)")
        plt.show()
    
    print("\nTesting double bottom detection...")
    # Process double bottom pattern
    peaks, troughs = detect_extrema(double_bottom_series.values)
    result = check_double_top_bottom(double_bottom_series.values, peaks, troughs)
    
    if result:
        pattern_type, key_points = result
        print(f"Successfully detected {pattern_type} pattern in synthetic data")
        
        # Visualize the pattern
        patterns = [(0, pattern_type, key_points)]
        visualize_double_patterns(double_bottom_series, patterns, len(double_bottom_series))
    else:
        print("Failed to detect double bottom pattern in synthetic data")
        plt.figure(figsize=(10, 5))
        plt.plot(double_bottom_series)
        plt.title("Synthetic Double Bottom Pattern (Not Detected)")
        plt.show()

def test_module_functions():
    """
    Test individual functions from the double_top_bottom module.
    """
    print("\n=== Testing individual functions ===")
    
    # Test detect_extrema function
    print("\nTesting detect_extrema function...")
    # Create a simple sequence with obvious peaks and troughs
    test_sequence = np.array([1, 2, 1, 3, 1, 4, 1, 2, 1])
    peaks, troughs = detect_extrema(test_sequence)
    
    print(f"Test sequence: {test_sequence}")
    print(f"Detected peaks at indices: {peaks}")
    print(f"Detected troughs at indices: {troughs}")
    
    # Update expected peaks to include index 7 (which is a valid local maximum)
    expected_peaks = [1, 3, 5, 7]
    expected_troughs = [0, 2, 4, 6, 8]
    
    if set(peaks) == set(expected_peaks) and set(troughs) == set(expected_troughs):
        print("detect_extrema function works correctly")
    else:
        print("detect_extrema function may have issues")
        print(f"Expected peaks: {expected_peaks}, got: {peaks}")
        print(f"Expected troughs: {expected_troughs}, got: {troughs}")
    
    # Test check_double_top_bottom function with a known pattern
    print("\nTesting check_double_top_bottom function...")
    # Create a double top pattern
    double_top = np.array([10, 20, 15, 20, 10])
    peaks, troughs = [1, 3], [0, 2, 4]  # Manually specify the peaks and troughs
    
    result = check_double_top_bottom(double_top, peaks, troughs)
    
    if result and result[0] == "double top":
        print("check_double_top_bottom correctly identified double top")
    else:
        print("check_double_top_bottom failed to identify double top")

def debug_matrix_profile():
    """
    Debug the matrix_profile computation to understand what's causing the error.
    """
    print("\n=== Debugging Matrix Profile ===")
    
    # Create a simple synthetic series to test with
    test_series = pd.Series(np.sin(np.linspace(0, 4*np.pi, 100)) + 10)
    
    print(f"Test series shape: {test_series.shape}")
    print(f"Test series has NaN values: {np.isnan(test_series).any()}")
    print(f"Test series min: {test_series.min()}, max: {test_series.max()}")
    
    # Try different window sizes
    for window in [5, 10, 20]:
        try:
            print(f"Testing with window size {window}...")
            # Convert to numpy array as in compute_matrix_profile
            values = np.array(test_series, dtype=float)
            print(f"Values shape: {values.shape}")
            
            # Try direct call to stumpy.stump
            import stumpy
            mp = stumpy.stump(values, m=window)
            print(f"Successfully computed matrix profile with window {window}")
            print(f"Matrix profile shape: {mp.shape}")
        except Exception as e:
            print(f"Error with window {window}: {e}")
    
    # Now try with real data
    print("\nTesting with real data...")
    symbol = "AAPL"
    period = "1mo"  # shorter period for testing
    interval = "1d"
    
    df = fetch_price_data(symbol, period, interval)
    price_series = df['Close']
    
    print(f"Price series shape: {price_series.shape}")
    print(f"Price series has NaN: {np.isnan(price_series).any()}")
    
    # Try with a small window
    window = 5
    try:
        # Make sure it's a 1D array
        values = np.array(price_series, dtype=float)
        if values.ndim > 1:
            values = values.flatten()
            
        mp = stumpy.stump(values, m=window)
        print(f"Success with real data and window {window}")
        print(f"Matrix profile shape: {mp.shape}")
    except Exception as e:
        print(f"Error with real data: {e}")

def main():
    """Main function to run all tests."""
    print("=== Running tests for double_top_bottom.py ===")
    
    # Debug matrix profile computation first
    debug_matrix_profile()
    
    # Test individual functions
    test_module_functions()
    
    # Test with synthetic data (doesn't require internet)
    test_with_synthetic_data()
    
    # Test with real data (requires internet connection)
    try:
        test_with_real_data()
    except Exception as e:
        print(f"Error testing with real data: {e}")
        print("Skipping real data test. Make sure you have an internet connection.")

if __name__ == "__main__":
    main()