import pandas as pd
import numpy as np
# Set matplotlib to use a non-interactive backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI
import matplotlib.pyplot as plt
from head_shoulder import (
    fetch_price_data, 
    compute_matrix_profile,
    detect_extrema,
    check_head_and_shoulders,
    extract_motif_candidates,
    visualize_patterns
)

def test_pattern_detection_pipeline():
    """
    Test the complete head and shoulders pattern detection pipeline using real market data.
    This function tests all components of the pattern detection process:
    1. Data fetching
    2. Matrix profile computation
    3. Motif extraction
    4. Pattern detection
    5. Visualization
    """
    print("\n=== Testing Head and Shoulders Pattern Detection Pipeline ===\n")
    
    # Step 1: Fetch price data
    print("Fetching price data...")
    symbol = "AAPL"
    interval = "1d"
    period = "180d"  # Use a longer period to have more data to work with
    
    prices = fetch_price_data(symbol=symbol, interval=interval, period=period)
    print(f"Retrieved {len(prices)} price points for {symbol}")
    print(f"Data type: {type(prices)}")
    
    # Convert prices to 1D array if needed
    if isinstance(prices, pd.DataFrame):
        print("Detected DataFrame, converting to Series")
        prices = prices.iloc[:, 0]  # Get the first column as a Series
        
    # For debugging:
    print(f"Data structure: {type(prices)}")
    print(f"Data is a Series: {isinstance(prices, pd.Series)}")
    print(f"Data shape: {prices.shape if hasattr(prices, 'shape') else len(prices)}")
    print(f"First few values: {prices.head() if hasattr(prices, 'head') else prices[:5]}")
    
    # Make sure we have a 1D numpy array for matrix profile computation
    flat_array = np.asarray(prices.values.flatten() if hasattr(prices, 'values') else prices.flatten() if hasattr(prices, 'flatten') else prices)
    print(f"Flattened array shape: {flat_array.shape}")
    print(f"Flattened array dimensionality: {flat_array.ndim}")
    
    # Step 2: Compute the matrix profile
    print("\nComputing matrix profile...")
    m = 30  # Window size (approx. 1.5 months of trading days)
    try:
        mp = compute_matrix_profile(flat_array, m)
        print(f"Matrix profile shape: {mp.shape}")
        
        # Step 3: Extract motif candidates
        print("\nExtracting motif candidates...")
        motifs = extract_motif_candidates(mp, m, top_k=5)
        print(f"Found {len(motifs)} potential motif pairs")
        for i, (idx1, idx2) in enumerate(motifs):
            print(f"  Motif {i+1}: ({idx1}, {idx2})")
        
        # Step 4: Check each motif for head and shoulders pattern
        print("\nAnalyzing motifs for head and shoulders patterns...")
        pattern_indices = []
        
        for motif_idx, (idx1, idx2) in enumerate(motifs):
            print(f"\nAnalyzing Motif {motif_idx+1}:")
            
            # Check first subsequence
            segment1 = prices.iloc[idx1:idx1+m] if hasattr(prices, 'iloc') else pd.Series(prices[idx1:idx1+m])
            peaks1, troughs1 = detect_extrema(segment1, order=3)
            is_pattern1 = check_head_and_shoulders(segment1, peaks1, troughs1)
            
            # Check second subsequence
            segment2 = prices.iloc[idx2:idx2+m] if hasattr(prices, 'iloc') else pd.Series(prices[idx2:idx2+m])
            peaks2, troughs2 = detect_extrema(segment2, order=3)
            is_pattern2 = check_head_and_shoulders(segment2, peaks2, troughs2)
            
            print(f"  Subsequence 1: Found {len(peaks1)} peaks, {len(troughs1)} troughs")
            print(f"  Subsequence 1 is head and shoulders pattern: {is_pattern1}")
            print(f"  Subsequence 2: Found {len(peaks2)} peaks, {len(troughs2)} troughs")
            print(f"  Subsequence 2 is head and shoulders pattern: {is_pattern2}")
            
            if is_pattern1:
                pattern_indices.append(idx1)
            if is_pattern2:
                pattern_indices.append(idx2)
        
        # Step 5: Visualize the results
        print(f"\nFound {len(pattern_indices)} head and shoulders patterns")
        
        if len(pattern_indices) > 0:
            print("Visualizing detected patterns...")
            visualize_patterns(prices, pattern_indices, m)
            plt.savefig(f"{symbol}_head_shoulders_patterns.png")
            print(f"Visualization saved as {symbol}_head_shoulders_patterns.png")
        else:
            print("No patterns found to visualize.")
            
            # Let's try a different value of 'order' for extrema detection
            print("\nTrying with different extrema detection parameters...")
            for order in [1, 2]:
                print(f"Using order={order} for extrema detection")
                test_indices = []
                
                # Sample a few random windows from the price series
                for i in range(3):
                    start_idx = np.random.randint(0, len(prices) - m)
                    segment = prices.iloc[start_idx:start_idx+m] if hasattr(prices, 'iloc') else pd.Series(prices[start_idx:start_idx+m])
                    peaks, troughs = detect_extrema(segment, order=order)
                    is_pattern = check_head_and_shoulders(segment, peaks, troughs)
                    
                    print(f"  Random window {i+1}: Found {len(peaks)} peaks, {len(troughs)} troughs")
                    print(f"  Is head and shoulders pattern: {is_pattern}")
                    
                    if is_pattern:
                        test_indices.append(start_idx)
                
                if test_indices:
                    print(f"Found {len(test_indices)} patterns with order={order}")
                    visualize_patterns(prices, test_indices, m)
                    plt.savefig(f"{symbol}_head_shoulders_alt_order{order}.png")
                    print(f"Visualization saved as {symbol}_head_shoulders_alt_order{order}.png")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_synthetic_data():
    """
    Generate synthetic data with a clear head and shoulders pattern 
    to test the detection algorithm.
    """
    print("\n=== Testing with Synthetic Data ===\n")
    
    # Generate synthetic price series with a head and shoulders pattern
    n = 50
    x = np.linspace(0, 10, n)
    
    # Create a clear head and shoulders pattern with more pronounced shoulders and troughs
    y = 100
    y = y + 10*np.exp(-0.5*((x-2)/0.5)**2)  # left shoulder
    y = y + 20*np.exp(-0.5*((x-5)/0.5)**2)  # head
    y = y + 10*np.exp(-0.5*((x-8)/0.5)**2)  # right shoulder
    
    # Add a slight downward trend to the whole series
    y = y - x * 1.0
    
    # Add some small noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, n)
    y += noise
    
    # Create troughs between the shoulders
    # Left trough
    trough_mask1 = (x > 3) & (x < 4)
    y[trough_mask1] = y[trough_mask1] - 8
    
    # Right trough
    trough_mask2 = (x > 6) & (x < 7)
    y[trough_mask2] = y[trough_mask2] - 8
    
    # Create a pandas Series
    dates = pd.date_range(start='2025-01-01', periods=n)
    synthetic_series = pd.Series(y, index=dates)
    
    # Detect extrema with different order values to find the best one for this pattern
    for order in [1, 2, 3]:
        print(f"\nTesting with detect_extrema order={order}")
        peaks, troughs = detect_extrema(synthetic_series, order=order)
        print(f"Found {len(peaks)} peaks and {len(troughs)} troughs")
        
        # Check for head and shoulders pattern
        is_pattern = check_head_and_shoulders(synthetic_series, peaks, troughs)
        print(f"Synthetic data is head and shoulders pattern: {is_pattern}")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(synthetic_series.index, synthetic_series, label="Synthetic Price")
        
        # Mark peaks and troughs
        for p in peaks:
            ax.plot(synthetic_series.index[p], synthetic_series.iloc[p], 'r^', markersize=10)
        for t in troughs:
            ax.plot(synthetic_series.index[t], synthetic_series.iloc[t], 'gv', markersize=10)
        
        ax.set_title(f"Synthetic Head and Shoulders Pattern (order={order})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        
        plt.savefig(f"synthetic_head_shoulders_order{order}.png")
        print(f"Synthetic data visualization saved as synthetic_head_shoulders_order{order}.png")
        plt.close()


def run_tests():
    """Run all test functions."""
    try:
        test_synthetic_data()
    except Exception as e:
        print(f"Error in synthetic data test: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_pattern_detection_pipeline()
    except Exception as e:
        print(f"Error in pattern detection pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_tests()