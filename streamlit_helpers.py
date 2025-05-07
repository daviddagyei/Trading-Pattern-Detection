import yfinance as yf
import pandas as pd
import numpy as np
import stumpy
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

def fetch_price_data(symbol: str = "AAPL", interval: str = "1h", period: str = "60d",
                    start_date: str = None, end_date: str = None) -> pd.Series:
    """
    Fetch historical price data for a given symbol and interval using Yahoo Finance.
    Returns a Pandas Series of adjusted closing prices indexed by datetime.
    
    Parameters:
    - symbol: Ticker symbol to download (default "AAPL").
    - interval: Data frequency (e.g. "1h" for hourly data).
    - period: Lookback period (e.g. "60d" for 60 days of data). Used if start_date and end_date are None.
    - start_date: Start date for data in YYYY-MM-DD format. If provided with end_date, period is ignored.
    - end_date: End date for data in YYYY-MM-DD format. If provided with start_date, period is ignored.
    """
    try:
        # Download data from Yahoo Finance based on either period or date range
        if start_date and end_date:
            df = yf.download(symbol, interval=interval, start=start_date, end=end_date, progress=False)
        else:
            df = yf.download(symbol, interval=interval, period=period, progress=False)
        
        # Check if we got data
        if df.empty:
            return None
            
        # Use Adjusted Close if available, otherwise fall back to Close
        price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        
        # Drop any rows with missing prices (if any)
        price_series = price_series.dropna()
        
        # Ensure we're returning a Series, not a DataFrame
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
            
        return price_series
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def compute_matrix_profile(price_array: np.ndarray, m: int):
    """
    Compute the matrix profile for the given price array and window length m.
    
    Parameters:
    - price_array: NumPy array of prices.
    - m: Window length for subsequences (number of points in each subsequence).
    
    Returns:
    - mp: NumPy array containing the matrix profile.
    """
    # Validate inputs
    if len(price_array) <= m:
        raise ValueError(f"Price array length ({len(price_array)}) must be greater than window length ({m})")
    
    # Compute the matrix profile using STUMPY (STOMP algorithm)
    mp = stumpy.stump(price_array, m)
    return mp

def extract_motif_candidates(mp: np.ndarray, m: int, top_k: int = 3):
    """
    Identify top motif candidates from the matrix profile.
    
    Parameters:
    - mp: Matrix profile array.
    - m: Window length used for the matrix profile (subsequence length).
    - top_k: Number of top motif pairs to retrieve.
    
    Returns:
    - motifs: List of tuples (i, j) for each motif, where i and j are the start indices of the motif pair.
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
            continue
        # Accept this motif pair
        motifs.append((idx, j))
        # Mark an exclusion zone around this motif pair to avoid overlapping selections
        start = min(idx, j)
        end = max(idx, j)
        used[max(0, start - m): min(len(mp), start + m)] = True
        used[max(0, end - m): min(len(mp), end + m)] = True
    return motifs

def detect_extrema(series_window: pd.Series, order: int = 3):
    """
    Identify local peaks and troughs in the given series window.
    
    Parameters:
    - series_window: Pandas Series containing a segment of the time series.
    - order: How many points on each side to use for comparing to find extrema.
    
    Returns:
    - peaks: List of indices that are local maxima.
    - troughs: List of indices that are local minima.
    """
    # Convert to numpy array and ensure float type
    arr = np.asarray(series_window.values, dtype=float)
    
    # Ensure order is valid for the array size
    valid_order = min(order, (len(arr) - 1) // 2)
    if valid_order < 1:
        valid_order = 1
    
    # Find indices of local maxima (peaks)
    peak_indices = argrelextrema(arr, np.greater, order=valid_order)[0]
    # Find indices of local minima (troughs)
    trough_indices = argrelextrema(arr, np.less, order=valid_order)[0]
    
    # Convert to Python lists
    peaks = peak_indices.tolist()
    troughs = trough_indices.tolist()
    
    return peaks, troughs

def check_head_and_shoulders(series_window: pd.Series, peaks: list, troughs: list,
                             shoulder_tol: float = 0.1, trough_tol: float = 0.2):
    """
    Determine if the given series window exhibits a head-and-shoulders pattern.
    
    Parameters:
    - series_window: Pandas Series for the subsequence being analyzed.
    - peaks: List of indices of local maxima in the window.
    - troughs: List of indices of local minima in the window.
    - shoulder_tol: Tolerance for shoulder height similarity.
    - trough_tol: Tolerance for trough alignment.
    
    Returns:
    - bool indicating whether a head-and-shoulders pattern is present.
    """
    # Need at least 3 peaks and 2 troughs for a head-and-shoulders
    if len(peaks) < 3 or len(troughs) < 2:
        return False
    
    arr = np.asarray(series_window.values, dtype=float)
    
    # Identify the highest peak (head)
    head_idx = max(peaks, key=lambda i: arr[i])
    head_val = arr[head_idx]
    
    # Find peaks to the left and right of the head
    left_peaks = [i for i in peaks if i < head_idx]
    right_peaks = [i for i in peaks if i > head_idx]
    
    # Need at least one peak on each side
    if not left_peaks or not right_peaks:
        return False
    
    # Define shoulders as the peaks closest to the head on each side
    left_shoulder_idx = max(left_peaks)
    right_shoulder_idx = min(right_peaks)
    
    # Ensure correct order: left_shoulder < head < right_shoulder
    if not (left_shoulder_idx < head_idx < right_shoulder_idx):
        return False
    
    left_shoulder_val = arr[left_shoulder_idx]
    right_shoulder_val = arr[right_shoulder_idx]
    
    # Find troughs between shoulders and head
    try:
        # Find trough between left shoulder and head
        segment = arr[left_shoulder_idx:head_idx]
        if len(segment) == 0:
            return False
        trough1_rel = int(np.argmin(segment))
        trough1_idx = left_shoulder_idx + trough1_rel
        
        # Find trough between head and right shoulder
        segment = arr[head_idx:right_shoulder_idx]
        if len(segment) == 0:
            return False
        trough2_rel = int(np.argmin(segment))
        trough2_idx = head_idx + trough2_rel
        
        trough1_val = arr[trough1_idx]
        trough2_val = arr[trough2_idx]
        
        # Rule 1: Head is higher than both shoulders
        if not (head_val > left_shoulder_val and head_val > right_shoulder_val):
            return False
            
        # Rule 2: Shoulders should be similar in height
        shoulder_height_diff = abs(left_shoulder_val - right_shoulder_val)
        if shoulder_height_diff > shoulder_tol * head_val:
            return False
            
        # Rule 3: Troughs should be at similar levels (neckline should be roughly horizontal)
        trough_height_diff = abs(trough1_val - trough2_val)
        higher_trough_val = max(trough1_val, trough2_val)
        if higher_trough_val > 0 and trough_height_diff > trough_tol * higher_trough_val:
            return False
            
        return True
        
    except (IndexError, ValueError) as e:
        # Handle any indexing errors or other issues
        return False

def safe_date_format(dt_obj, format_str='%Y-%m-%d'):
    """Safely format a datetime object or return a placeholder if it's not a datetime"""
    if pd.api.types.is_datetime64_any_dtype(dt_obj) or isinstance(dt_obj, datetime):
        try:
            return dt_obj.strftime(format_str)
        except:
            return str(dt_obj)
    else:
        return str(dt_obj)

def plotly_visualize_price_with_patterns(price_series, patterns, m):
    """Visualize the price series with detected patterns using Plotly"""
    fig = go.Figure()
    
    # Add price series
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add rectangles for each pattern
    for idx in patterns:
        if idx >= len(price_series) or idx + m > len(price_series):
            continue  # Skip invalid indices
            
        start_time = price_series.index[idx]
        end_time = price_series.index[min(idx + m - 1, len(price_series) - 1)]
        
        # Add a shaded rectangle for each pattern
        fig.add_shape(
            type="rect",
            x0=start_time,
            x1=end_time,
            y0=min(price_series.iloc[idx:idx+m]),
            y1=max(price_series.iloc[idx:idx+m]),
            fillcolor="rgba(255, 165, 0, 0.3)",
            opacity=0.7,
            layer="below",
            line_width=0,
        )
    
    fig.update_layout(
        title=f"Price Chart with Detected Head and Shoulders Patterns",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plotly_visualize_pattern(segment, pattern_idx, peaks, troughs, is_pattern):
    """Visualize an individual pattern using Plotly"""
    fig = go.Figure()
    
    if len(segment) == 0:
        return fig
    
    # Plot price series
    fig.add_trace(go.Scatter(
        x=segment.index,
        y=segment.values,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    if is_pattern and len(peaks) >= 3:
        arr = np.asarray(segment.values, dtype=float)
        
        try:
            # Identify pattern points
            head_idx = max(peaks, key=lambda j: arr[j])
            left_peaks = [j for j in peaks if j < head_idx]
            right_peaks = [j for j in peaks if j > head_idx]
            
            if left_peaks:
                left_sh_idx = max(left_peaks)
                fig.add_trace(go.Scatter(
                    x=[segment.index[left_sh_idx]],
                    y=[segment.iloc[left_sh_idx]],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    text=['Left Shoulder'],
                    textposition='top center',
                    name='Left Shoulder'
                ))
                
                # Left trough
                if left_sh_idx < head_idx and left_sh_idx + 1 < len(arr):
                    trough_segment = arr[left_sh_idx:head_idx]
                    if len(trough_segment) > 0:
                        trough1_rel = int(np.argmin(trough_segment))
                        trough1_idx = left_sh_idx + trough1_rel
                        if 0 <= trough1_idx < len(segment):
                            fig.add_trace(go.Scatter(
                                x=[segment.index[trough1_idx]],
                                y=[segment.iloc[trough1_idx]],
                                mode='markers+text',
                                marker=dict(symbol='triangle-down', size=12, color='red'),
                                text=['Trough 1'],
                                textposition='bottom center',
                                name='Trough 1'
                            ))
            
            # Head
            if 0 <= head_idx < len(segment):
                fig.add_trace(go.Scatter(
                    x=[segment.index[head_idx]],
                    y=[segment.iloc[head_idx]],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=16, color='blue'),
                    text=['Head'],
                    textposition='top center',
                    name='Head'
                ))
            
            if right_peaks:
                right_sh_idx = min(right_peaks)
                if 0 <= right_sh_idx < len(segment):
                    fig.add_trace(go.Scatter(
                        x=[segment.index[right_sh_idx]],
                        y=[segment.iloc[right_sh_idx]],
                        mode='markers+text',
                        marker=dict(symbol='triangle-up', size=12, color='green'),
                        text=['Right Shoulder'],
                        textposition='top center',
                        name='Right Shoulder'
                    ))
                
                # Right trough
                if head_idx < right_sh_idx and head_idx + 1 < len(arr):
                    trough_segment = arr[head_idx:right_sh_idx]
                    if len(trough_segment) > 0:
                        trough2_rel = int(np.argmin(trough_segment))
                        trough2_idx = head_idx + trough2_rel
                        if 0 <= trough2_idx < len(segment):
                            fig.add_trace(go.Scatter(
                                x=[segment.index[trough2_idx]],
                                y=[segment.iloc[trough2_idx]],
                                mode='markers+text',
                                marker=dict(symbol='triangle-down', size=12, color='red'),
                                text=['Trough 2'],
                                textposition='bottom center',
                                name='Trough 2'
                            ))
        except Exception as e:
            # If any errors occur during visualization, add the error message
            fig.add_annotation(
                text=f"Error visualizing pattern: {str(e)}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="red")
            )
    else:
        fig.add_annotation(
            text="Pattern criteria not met or insufficient peaks",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=20, color="red")
        )
    
    fig.update_layout(
        title=f"Pattern starting at index {pattern_idx}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        showlegend=True
    )
    
    return fig