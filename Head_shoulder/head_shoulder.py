import yfinance as yf
import pandas as pd

def fetch_price_data(symbol: str = "AAPL", interval: str = "1h", period: str = "60d") -> pd.Series:
    """
    Fetch historical price data for a given symbol and interval using Yahoo Finance.
    Returns a Pandas Series of adjusted closing prices indexed by datetime.
    
    Parameters:
    - symbol: Ticker symbol to download (default "AAPL").
    - interval: Data frequency (e.g. "1h" for hourly data).
    - period: Lookback period (e.g. "60d" for 60 days of data).
    """
    # Download data from Yahoo Finance
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    # Use Adjusted Close if available, otherwise fall back to Close
    price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    # Drop any rows with missing prices (if any)
    price_series = price_series.dropna()
    return price_series

import numpy as np
import stumpy

def compute_matrix_profile(price_series: pd.Series, m: int):
    """
    Compute the matrix profile for the given price series and window length m.
    
    Parameters:
    - price_series: Pandas Series of prices.
    - m: Window length for subsequences (number of points in each subsequence).
    
    Returns:
    - mp: NumPy array of shape (len(price_series) - m + 1, 4) containing the matrix profile.
          Column 0 is the distance (minimum z-normalized Euclidean distance to nearest neighbor).
          Column 1 is the index of the nearest neighbor subsequence.
          Column 2 and 3 are indices for the left and right nearest neighbors (for time series chains).
    """
    # Convert series to numpy array (float type) for STUMPY
    T = np.asarray(price_series, dtype=float)
    # Compute the matrix profile using STUMPY (STOMP algorithm)
    mp = stumpy.stump(T, m)
    return mp

def extract_motif_candidates(mp: np.ndarray, m: int, top_k: int = 3):
    """
    Identify top motif candidates from the matrix profile.
    
    Parameters:
    - mp: Matrix profile array (as returned by compute_matrix_profile).
    - m: Window length used for the matrix profile (subsequence length).
    - top_k: Number of top motif pairs to retrieve.
    
    Returns:
    - motifs: List of tuples (i, j) for each motif, where i and j are the start indices of the motif pair.
             These indices represent subsequences that are motifs (lowest distance pairs) in the series.
    """
    motifs = []
    if mp is None or len(mp) == 0:
        return motifs
    # Extract the matrix profile values (distances) and sort indices by distance (ascending)
    distances = mp[:, 0]
    sorted_indices = np.argsort(distances)
    used = np.zeros(len(mp), dtype=bool)  # marker for indices already part of a chosen motif
    
    for idx in sorted_indices:
        if len(motifs) >= top_k:
            break  # found the required number of motifs
        if np.isinf(distances[idx]):
            # Skip if distance is infinite (could happen if no valid neighbor)
            continue
        j = int(mp[idx, 1])  # nearest neighbor index for idx
        if j == -1:
            continue  # no neighbor (edge case)
        # Ensure neither idx nor its neighbor j overlap with any already selected motif
        if used[idx] or used[j]:
            continue
        if abs(idx - j) < m:
            # If the two motif subsequences themselves overlap (distance < m apart), skip
            # (Normally, nearest neighbor won't be an overlapping subsequence due to how matrix profile is computed)
            continue
        # Accept this motif pair
        motifs.append((idx, j))
        # Mark an exclusion zone around this motif pair to avoid overlapping selections
        start = min(idx, j)
        end = max(idx, j)
        used[max(0, start - m): min(len(mp), start + m)] = True
        used[max(0, end - m): min(len(mp), end + m)] = True
    return motifs

from scipy.signal import argrelextrema

def detect_extrema(series_window: pd.Series, order: int = 3):
    """
    Identify local peaks and troughs in the given series window.
    
    Parameters:
    - series_window: Pandas Series (or 1D array-like) containing a segment of the time series.
    - order: How many points on each side to use for comparing to find extrema (tuning parameter).
             For example, order=3 means each candidate point is compared to 3 points on left and right.
    
    Returns:
    - peaks: List of indices (relative to the start of series_window) that are local maxima.
    - troughs: List of indices that are local minima.
    """
    arr = np.asarray(series_window, dtype=float)
    # Find indices of local maxima (peaks)
    peak_indices = argrelextrema(arr, np.greater, order=order)[0]
    # Find indices of local minima (troughs)
    trough_indices = argrelextrema(arr, np.less, order=order)[0]
    # Convert to Python lists for easier handling
    peaks = peak_indices.tolist()
    troughs = trough_indices.tolist()
    return peaks, troughs

def check_head_and_shoulders(series_window: pd.Series, peaks: list, troughs: list, 
                             shoulder_tol: float = 0.1, trough_tol: float = 0.2):
    """
    Determine if the given series window exhibits a head-and-shoulders pattern.
    
    Parameters:
    - series_window: Pandas Series for the subsequence being analyzed.
    - peaks: List of indices of local maxima in the window (from detect_extrema).
    - troughs: List of indices of local minima in the window.
    - shoulder_tol: Tolerance (fraction) for shoulder height similarity. For example, 0.1 means
                    shoulders' heights should differ by no more than 10%.
    - trough_tol: Tolerance for trough alignment. This controls how much the two neckline troughs
                  can differ in depth (e.g., 0.2 = 20% max difference).
    
    Returns:
    - bool indicating whether a head-and-shoulders pattern is present.
    """
    if len(peaks) < 3 or len(troughs) < 2:
        # Need at least 3 peaks and 2 troughs for a head-and-shoulders
        return False
    
    arr = np.asarray(series_window, dtype=float)
    # Identify the highest peak (head)
    head_idx = max(peaks, key=lambda i: arr[i])
    head_val = arr[head_idx]
    # Find the tallest peak to the left of the head
    left_peaks = [i for i in peaks if i < head_idx]
    # Find the tallest peak to the right of the head
    right_peaks = [i for i in peaks if i > head_idx]
    if not left_peaks or not right_peaks:
        return False  # No shoulder on one side
    # Define left shoulder as the peak immediately before the head (rightmost peak on the left side)
    left_shoulder_idx = max(left_peaks)
    # Define right shoulder as the peak immediately after the head (leftmost peak on the right side)
    right_shoulder_idx = min(right_peaks)
    
    # Ensure the order: left_shoulder < head < right_shoulder in time
    if not (left_shoulder_idx < head_idx < right_shoulder_idx):
        return False
    
    left_shoulder_val = arr[left_shoulder_idx]
    right_shoulder_val = arr[right_shoulder_idx]
    
    # Identify troughs (valleys) between the peaks:
    # trough1 between left shoulder and head (min value in that interval)
    if left_shoulder_idx < head_idx:
        segment = arr[left_shoulder_idx: head_idx]
        if len(segment) == 0:
            return False
        trough1_rel = int(np.argmin(segment))  # relative index within the segment
        trough1_idx = left_shoulder_idx + trough1_rel
    else:
        return False
    # trough2 between head and right shoulder
    if head_idx < right_shoulder_idx:
        segment = arr[head_idx: right_shoulder_idx]
        if len(segment) == 0:
            return False
        trough2_rel = int(np.argmin(segment))
        trough2_idx = head_idx + trough2_rel
    else:
        return False
    
    trough1_val = arr[trough1_idx]
    trough2_val = arr[trough2_idx]
    
    # Rule 1: Head is higher than both shoulders
    if not (head_val > left_shoulder_val and head_val > right_shoulder_val):
        return False
    # Rule 2: Shoulders are lower than head but relatively close to each other in height
    shoulder_height_diff = abs(left_shoulder_val - right_shoulder_val)
    # Shoulders difference relative to head (or to the lower shoulder) should be within tolerance
    if shoulder_height_diff > shoulder_tol * head_val:
        # If shoulders differ by more than, say, 10% of head height, not a clear pattern
        return False
    # Rule 3: Neckline troughs alignment (within tolerance)
    # Calculate difference relative to higher trough
    trough_height_diff = abs(trough1_val - trough2_val)
    higher_trough_val = max(trough1_val, trough2_val)
    if higher_trough_val > 0:  # to avoid division by zero
        if trough_height_diff > trough_tol * higher_trough_val:
            # If troughs differ by more than (e.g.) 20% of the higher trough value, pattern is not well-formed
            return False
    
    # Optional (Rule 4): The pattern should resemble: peak-trough-peak-trough-peak sequence
    # We ensure that the indices are in alternating order, which they are by construction here.
    # Additional checks for symmetry (e.g., distance between shoulders and head) can be added if needed.
    
    return True

import matplotlib.pyplot as plt

def visualize_patterns(price_series: pd.Series, patterns: list, m: int):
    """
    Visualize the price series with detected head-and-shoulders patterns highlighted.
    Also plot each detected pattern separately with annotations.
    
    Parameters:
    - price_series: Pandas Series of the full price data.
    - patterns: List of start indices for detected head-and-shoulders patterns.
    - m: Window length (number of points) corresponding to the pattern length.
    """
    if not patterns:
        print("No patterns to display.")
        return
    
    # Plot full price series
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_series.index, price_series.values, label="Price")
    for idx in patterns:
        # Shade the window corresponding to each pattern
        start_time = price_series.index[idx]
        end_time = price_series.index[min(idx + m - 1, len(price_series) - 1)]
        ax.axvspan(start_time, end_time, color='orange', alpha=0.3)
    ax.set_title("Price Series with Detected Head-and-Shoulders Patterns")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    
    # Plot each pattern in a separate subplot with annotations
    num_patterns = len(patterns)
    fig2, axs = plt.subplots(num_patterns, 1, figsize=(8, 4 * num_patterns))
    if num_patterns == 1:
        axs = [axs]  # ensure axs is iterable
    for i, idx in enumerate(patterns):
        segment = price_series.iloc[idx: idx + m]
        axs[i].plot(segment.index, segment.values, color='C1')
        axs[i].set_title(f"Pattern starting at index {idx}")
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Price")
        # Detect extrema in this segment for annotation
        peaks, troughs = detect_extrema(segment, order=1)  # use a smaller order for fine detail if needed
        # Identify pattern points (if any) to annotate
        if check_head_and_shoulders(segment, peaks, troughs):
            arr = np.asarray(segment, dtype=float)
            head_idx = max(peaks, key=lambda j: arr[j])
            left_sh_idx = max([j for j in peaks if j < head_idx]) if any(j < head_idx for j in peaks) else None
            right_sh_idx = min([j for j in peaks if j > head_idx]) if any(j > head_idx for j in peaks) else None
            if left_sh_idx is not None:
                axs[i].annotate("Left Shoulder", xy=(segment.index[left_sh_idx], segment.iloc[left_sh_idx]),
                                xytext=(segment.index[left_sh_idx], segment.iloc[left_sh_idx] * 1.01),
                                arrowprops=dict(arrowstyle="->"))
            axs[i].annotate("Head", xy=(segment.index[head_idx], segment.iloc[head_idx]),
                            xytext=(segment.index[head_idx], segment.iloc[head_idx] * 1.05),
                            arrowprops=dict(arrowstyle="->"))
            if right_sh_idx is not None:
                axs[i].annotate("Right Shoulder", xy=(segment.index[right_sh_idx], segment.iloc[right_sh_idx]),
                                xytext=(segment.index[right_sh_idx], segment.iloc[right_sh_idx] * 1.01),
                                arrowprops=dict(arrowstyle="->"))
            # Troughs (neckline points)
            # Find troughs between peaks for annotation
            if left_sh_idx is not None:
                trough1_idx = left_sh_idx + int(np.argmin(arr[left_sh_idx: head_idx]))
                axs[i].plot(segment.index[trough1_idx], segment.iloc[trough1_idx], 'v', color='green')
                axs[i].annotate("Trough 1", xy=(segment.index[trough1_idx], segment.iloc[trough1_idx]),
                                xytext=(segment.index[trough1_idx], segment.iloc[trough1_idx] * 0.95),
                                arrowprops=dict(arrowstyle="->"))
            if right_sh_idx is not None:
                trough2_idx = head_idx + int(np.argmin(arr[head_idx: right_sh_idx]))
                axs[i].plot(segment.index[trough2_idx], segment.iloc[trough2_idx], 'v', color='green')
                axs[i].annotate("Trough 2", xy=(segment.index[trough2_idx], segment.iloc[trough2_idx]),
                                xytext=(segment.index[trough2_idx], segment.iloc[trough2_idx] * 0.95),
                                arrowprops=dict(arrowstyle="->"))
        else:
            axs[i].text(0.5, 0.5, "Pattern criteria not met", transform=axs[i].transAxes,
                        horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
    plt.tight_layout()
    plt.show()
