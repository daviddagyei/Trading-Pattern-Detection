import pandas as pd
import yfinance as yf

def fetch_price_data(symbol: str = "AAPL", period: str = "1y", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch historical price data for the given stock symbol from Yahoo Finance.
    By default, downloads 1 year of hourly data for AAPL.
    - symbol: Ticker symbol to download (e.g., "AAPL").
    - period: Lookback period (e.g., "1y" for one year, "6mo" for six months).
    - interval: Data interval (e.g., "1h" for hourly, "1d" for daily).
    Returns:
    - Pandas DataFrame with datetime index and OHLCV columns (Open, High, Low, Close, Volume).
    """
    df = yf.download(symbol, period=period, interval=interval)
    # Ensure the data is sorted by time (ascending chronological order)
    df.sort_index(inplace=True)
    return df

import numpy as np
import stumpy

def compute_matrix_profile(price_series, window: int):
    """
    Compute the matrix profile for the given price series using a specified window length.
    - price_series: Pandas Series of price data (e.g., closing prices indexed by time).
    - window: Length of the subsequence (number of data points) to use for motif search.
              This should roughly match the expected length of the pattern to detect.
    Returns:
    - profile: Numpy array of matrix profile distances (lowest neighbor distance for each subsequence).
    - profile_index: Numpy array of indices of the nearest neighbor for each subsequence.
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
    
    # Handle any NaN values
    if np.isnan(values).any():
        # Replace NaNs with interpolated values or simply drop them
        nan_indices = np.isnan(values)
        valid_indices = ~nan_indices
        if np.any(valid_indices):  # If there's at least one valid value
            # Simple linear interpolation - replace NaNs with nearest valid values
            values[nan_indices] = np.interp(
                np.flatnonzero(nan_indices), 
                np.flatnonzero(valid_indices), 
                values[valid_indices]
            )
        else:
            # If all values are NaN (unlikely but possible), replace with zeros
            values = np.zeros_like(values)
    
    # Compute the matrix profile
    mp = stumpy.stump(values, m=window)
    profile = mp[:, 0]   # matrix profile distances
    profile_index = mp[:, 1].astype(int)  # indices of the nearest neighbor for each subsequence
    return profile, profile_index

def extract_motif_candidates(profile: np.ndarray, profile_index: np.ndarray, window: int, top_k: int = 3):
    """
    Identify the top-k motif pairs (subsequence indices of repeating patterns) from the matrix profile.
    Ensures that once a motif is selected, no overlapping subsequences are selected again.
    - profile: Numpy array of matrix profile distances.
    - profile_index: Numpy array of nearest-neighbor indices corresponding to 'profile'.
    - window: The subsequence window length (used to enforce non-overlap exclusion zones).
    - top_k: Number of motif pairs to retrieve.
    Returns:
    - motifs: List of tuples [(i, j), ...] of starting indices for each motif pair (i < j).
    """
    motifs = []
    # Make a copy of the profile distances so we can modify it without affecting original
    distance = profile.copy()
    n = len(distance)
    for _ in range(top_k):
        # Find the smallest distance in the profile
        idx = np.nanargmin(distance)  # use nanargmin to ignore any NaN (if present)
        min_dist = distance[idx]
        if not np.isfinite(min_dist):
            break  # no more valid motifs (we've exhausted all non-overlapping regions)
        # Get the nearest neighbor index for this subsequence
        j = int(profile_index[idx])
        i = int(idx)
        if i == j:
            # This would indicate a self-match (shouldn't happen with proper matrix profile)
            distance[idx] = np.inf
            continue
        # Ensure i < j for consistency in ordering
        if j < i:
            i, j = j, i
        motifs.append((i, j))
        # Exclude a window-length region around each of these indices from further consideration
        start_i = max(0, i - window + 1)
        end_i   = min(n, i + window - 1)
        start_j = max(0, j - window + 1)
        end_j   = min(n, j + window - 1)
        distance[start_i:end_i+1] = np.inf
        distance[start_j:end_j+1] = np.inf
    return motifs

from scipy.signal import argrelextrema

def detect_extrema(subsequence: np.ndarray):
    """
    Find local extrema (peaks and troughs) in the given subsequence.
    - subsequence: numpy array of price values representing a segment of the time series.
    Returns:
    - peaks: list of indices (int) of local maxima in the subsequence.
    - troughs: list of indices of local minima in the subsequence.
    """
    # Use argrelextrema to find indices of local maxima and minima
    # np.greater for peaks (value greater than its immediate neighbors)
    # np.less for troughs (value less than its immediate neighbors)
    peaks = list(argrelextrema(subsequence, np.greater)[0])
    troughs = list(argrelextrema(subsequence, np.less)[0])
    # Check the edges manually: include index 0 or last index if they qualify as extremum
    n = len(subsequence)
    if n < 2:
        return peaks, troughs  # trivial case, no interior points
    # Edge at start
    if n > 1:
        if subsequence[0] > subsequence[1]:
            peaks.insert(0, 0)
        if subsequence[0] < subsequence[1]:
            troughs.insert(0, 0)
    # Edge at end
    if subsequence[-1] > subsequence[-2]:
        peaks.append(n-1)
    if subsequence[-1] < subsequence[-2]:
        troughs.append(n-1)
    # Sort the lists just in case (not strictly necessary, argrelextrema returns sorted by default)
    peaks.sort()
    troughs.sort()
    return peaks, troughs

def check_double_top_bottom(subsequence: np.ndarray, peaks: list, troughs: list, similarity_tol: float = 0.05):
    """
    Determine if a given subsequence (with identified peaks/troughs) forms a double top or double bottom pattern.
    - subsequence: numpy array of price values for the segment.
    - peaks: list of indices of local maxima in the segment.
    - troughs: list of indices of local minima in the segment.
    - similarity_tol: tolerance for peak/trough height similarity (fraction of value).
                      For example, 0.05 means peaks can differ by at most 5% in height.
    Returns:
    - A tuple (pattern_type, key_indices) where:
         pattern_type is "double top" or "double bottom" (or None if no pattern detected),
         key_indices is a list of the key points [idx1, idx2, idx3] that define the pattern:
            For double top: [first_peak_idx, trough_idx, second_peak_idx] (chronological order).
            For double bottom: [first_trough_idx, peak_idx, second_trough_idx].
    """
    n = len(subsequence)
    pattern_type = None
    key_points = []

    # Check for double top pattern
    if len(peaks) >= 2:
        # Identify the two highest peaks by value
        peak_values = [(subsequence[i], i) for i in peaks]
        peak_values.sort(key=lambda x: x[0], reverse=True)  # sort by height descending
        top1_val, top1_idx = peak_values[0]
        top2_val, top2_idx = peak_values[1]
        # Ensure the two highest peaks are not the same point and appear in time order
        if top1_idx == top2_idx:
            # (This would only happen if the highest peak was listed twice, which shouldn't occur)
            pass
        else:
            # Sort the two peak indices chronologically
            p1, p2 = sorted([top1_idx, top2_idx])
            # Check there is a trough between these two peaks
            if p2 - p1 > 1:  # there is at least one point between p1 and p2
                # Find the lowest value between these two peaks (the valley)
                mid_segment = subsequence[p1:p2+1]
                valley_idx_rel = np.argmin(mid_segment)  # index relative to p1
                valley_idx = p1 + valley_idx_rel
                valley_val = subsequence[valley_idx]
                # Check that valley is indeed between the two peaks
                if valley_idx > p1 and valley_idx < p2:
                    # Check similarity of peak heights
                    height_diff = abs(top1_val - top2_val)
                    max_height = max(top1_val, top2_val)
                    if max_height == 0:
                        height_ratio = 0.0
                    else:
                        height_ratio = height_diff / max_height
                    # Both peaks should be of similar height (difference within tolerance)
                    if height_ratio <= similarity_tol:
                        # (Optional) You could also enforce that the valley is significantly lower than the peaks,
                        # e.g., ensure valley_val is some percentage below the peak values if desired.
                        pattern_type = "double top"
                        key_points = [p1, valley_idx, p2]

    # Check for double bottom pattern (if not already classified as double top)
    if pattern_type is None and len(troughs) >= 2:
        # Identify two lowest troughs by value
        trough_values = [(subsequence[i], i) for i in troughs]
        trough_values.sort(key=lambda x: x[0])  # sort by value ascending (lower = more extreme trough)
        bot1_val, bot1_idx = trough_values[0]
        bot2_val, bot2_idx = trough_values[1]
        if bot1_idx != bot2_idx:
            t1, t2 = sorted([bot1_idx, bot2_idx])
            if t2 - t1 > 1:  # at least one point between troughs
                mid_segment = subsequence[t1:t2+1]
                peak_idx_rel = np.argmax(mid_segment)
                peak_idx = t1 + peak_idx_rel
                peak_val = subsequence[peak_idx]
                if peak_idx > t1 and peak_idx < t2:
                    depth_diff = abs(bot1_val - bot2_val)
                    # Use absolute values for depth comparison, since trough values are lower
                    # (If prices are positive, bot1_val and bot2_val will be positive numbers or equal zero.)
                    max_depth = max(abs(bot1_val), abs(bot2_val))
                    depth_ratio = depth_diff / max_depth if max_depth != 0 else 0.0
                    if depth_ratio <= similarity_tol:
                        # (Optional) Enforce peak between troughs is significantly higher than trough values.
                        pattern_type = "double bottom"
                        key_points = [t1, peak_idx, t2]

    if pattern_type:
        return pattern_type, key_points
    else:
        return None

import matplotlib.pyplot as plt

def visualize_double_patterns(price_series: pd.Series, patterns: list, window: int):
    """
    Plot the price series with highlighted double top/bottom patterns.
    - price_series: Pandas Series of the price data (indexed by time).
    - patterns: List of detected patterns, where each element is 
                (start_index, pattern_type, key_points_indices).
                start_index is the index in the price_series where the pattern subsequence begins,
                pattern_type is "double top" or "double bottom",
                key_points_indices is a list of three relative indices [i1, i2, i3] within the subsequence 
                (as returned by check_double_top_bottom).
    - window: The length of each pattern subsequence (same as used for motif search).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(price_series.index, price_series.values, label="Price", color='blue')
    ax = plt.gca()
    # Define markers and colors for peaks/troughs
    peak_marker = '^'   # triangle up for peaks
    trough_marker = 'v'  # triangle down for troughs
    
    for (start_idx, pattern_type, key_idxs) in patterns:
        end_idx = start_idx + window - 1  # end index of the subsequence
        # Shade the region of the pattern
        ax.axvspan(price_series.index[start_idx], price_series.index[end_idx], 
                   color='orange' if pattern_type == "double top" else 'green', 
                   alpha=0.3)
        # Plot markers for the key points
        # Convert relative indices to absolute indices in the original series
        abs_key_idxs = [start_idx + ki for ki in key_idxs]
        # If it's a double top, key_idxs = [peak1, trough, peak2] in time order
        # If double bottom, key_idxs = [trough1, peak, trough2]
        if pattern_type == "double top":
            # Mark peaks (first and last in key_idxs) with triangle-up, trough (middle) with triangle-down
            ax.plot(price_series.index[abs_key_idxs[0]], price_series.iloc[abs_key_idxs[0]], 
                    peak_marker, color='red', markersize=8)
            ax.plot(price_series.index[abs_key_idxs[2]], price_series.iloc[abs_key_idxs[2]], 
                    peak_marker, color='red', markersize=8)
            ax.plot(price_series.index[abs_key_idxs[1]], price_series.iloc[abs_key_idxs[1]], 
                    trough_marker, color='red', markersize=8)
        else:  # double bottom
            # Mark troughs (first and last in key_idxs) with triangle-down, peak (middle) with triangle-up
            ax.plot(price_series.index[abs_key_idxs[0]], price_series.iloc[abs_key_idxs[0]], 
                    trough_marker, color='green', markersize=8)
            ax.plot(price_series.index[abs_key_idxs[2]], price_series.iloc[abs_key_idxs[2]], 
                    trough_marker, color='green', markersize=8)
            ax.plot(price_series.index[abs_key_idxs[1]], price_series.iloc[abs_key_idxs[1]], 
                    peak_marker, color='green', markersize=8)
        # Annotate the pattern type text near the middle of the pattern region
        mid_idx = start_idx + (window // 2)
        ax.text(price_series.index[mid_idx], price_series.iloc[mid_idx], 
                "Double Top" if pattern_type == "double top" else "Double Bottom", 
                color='black', fontsize=9, fontweight='bold',
                ha='center', va='bottom' if pattern_type == "double top" else 'top')
    
    plt.title("Detected Double Top/Bottom Patterns in Price Series")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
