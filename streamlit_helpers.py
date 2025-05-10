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
        if start_date and end_date:
            df = yf.download(symbol, interval=interval, start=start_date, end=end_date, progress=False)
        else:
            df = yf.download(symbol, interval=interval, period=period, progress=False)
        
        if df.empty:
            return None
            
        price_series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        
        price_series = price_series.dropna()
        
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
    if len(price_array) <= m:
        raise ValueError(f"Price array length ({len(price_array)}) must be greater than window length ({m})")
    
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
    
    distances = mp[:, 0]
    sorted_indices = np.argsort(distances)
    used = np.zeros(len(mp), dtype=bool)
    
    for idx in sorted_indices:
        if len(motifs) >= top_k:
            break
        if np.isinf(distances[idx]):
            continue
        j = int(mp[idx, 1])
        if j == -1:
            continue
        if used[idx] or used[j]:
            continue
        if abs(idx - j) < m:
            continue
        motifs.append((idx, j))
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
    arr = np.asarray(series_window.values, dtype=float)
    
    valid_order = min(order, (len(arr) - 1) // 2)
    if valid_order < 1:
        valid_order = 1
    
    peak_indices = argrelextrema(arr, np.greater, order=valid_order)[0]
    trough_indices = argrelextrema(arr, np.less, order=valid_order)[0]
    
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
    if len(peaks) < 3 or len(troughs) < 2:
        return False
    
    arr = np.asarray(series_window.values, dtype=float)
    
    head_idx = max(peaks, key=lambda i: arr[i])
    head_val = arr[head_idx]
    
    left_peaks = [i for i in peaks if i < head_idx]
    right_peaks = [i for i in peaks if i > head_idx]
    
    if not left_peaks or not right_peaks:
        return False
    
    left_shoulder_idx = max(left_peaks)
    right_shoulder_idx = min(right_peaks)
    
    if not (left_shoulder_idx < head_idx < right_shoulder_idx):
        return False
    
    left_shoulder_val = arr[left_shoulder_idx]
    right_shoulder_val = arr[right_shoulder_idx]
    
    try:
        segment = arr[left_shoulder_idx:head_idx]
        if len(segment) == 0:
            return False
        trough1_rel = int(np.argmin(segment))
        trough1_idx = left_shoulder_idx + trough1_rel
        
        segment = arr[head_idx:right_shoulder_idx]
        if len(segment) == 0:
            return False
        trough2_rel = int(np.argmin(segment))
        trough2_idx = head_idx + trough2_rel
        
        trough1_val = arr[trough1_idx]
        trough2_val = arr[trough2_idx]
        
        if not (head_val > left_shoulder_val and head_val > right_shoulder_val):
            return False
            
        shoulder_height_diff = abs(left_shoulder_val - right_shoulder_val)
        if shoulder_height_diff > shoulder_tol * head_val:
            return False
            
        trough_height_diff = abs(trough1_val - trough2_val)
        higher_trough_val = max(trough1_val, trough2_val)
        if higher_trough_val > 0 and trough_height_diff > trough_tol * higher_trough_val:
            return False
            
        return True
        
    except (IndexError, ValueError) as e:
        return False

def check_double_top_bottom(series_window: pd.Series, peaks: list, troughs: list, similarity_tol: float = 0.05):
    """
    Determine if a given subsequence (with identified peaks/troughs) forms a double top or double bottom pattern.
    - series_window: Pandas Series of price values for the segment.
    - peaks: list of indices of local maxima in the segment.
    - troughs: list of indices of local minima in the segment.
    - similarity_tol: tolerance for peak/trough height similarity (fraction of value).
                      For example, 0.05 means peaks can differ by at most 5% in height.
    Returns:
    - A tuple (pattern_type, key_indices) where:
         pattern_type is "double top" or "double bottom" (or None if no pattern detected),
         key_indices is a list of the key points [idx1, idx2, idx3] that define the pattern:
            For double top: [first_peak_idx, trough_idx, second_peak_idx]
            For double bottom: [first_trough_idx, peak_idx, second_trough_idx]
    """
    arr = np.asarray(series_window.values, dtype=float)
    n = len(arr)
    pattern_type = None
    key_points = []

    if len(peaks) >= 2:
        peak_values = [(arr[i], i) for i in peaks]
        peak_values.sort(key=lambda x: x[0], reverse=True)
        top1_val, top1_idx = peak_values[0]
        top2_val, top2_idx = peak_values[1]
        
        if top1_idx != top2_idx:
            p1, p2 = sorted([top1_idx, top2_idx])
            if p2 - p1 > 1:
                mid_segment = arr[p1:p2+1]
                valley_idx_rel = np.argmin(mid_segment)
                valley_idx = p1 + valley_idx_rel
                valley_val = arr[valley_idx]
                if valley_idx > p1 and valley_idx < p2:
                    height_diff = abs(arr[top1_idx] - arr[top2_idx])
                    max_height = max(arr[top1_idx], arr[top2_idx])
                    if max_height == 0:
                        height_ratio = 0.0
                    else:
                        height_ratio = height_diff / max_height
                    if height_ratio <= similarity_tol:
                        pattern_type = "double top"
                        key_points = [p1, valley_idx, p2]

    if pattern_type is None and len(troughs) >= 2:
        trough_values = [(arr[i], i) for i in troughs]
        trough_values.sort(key=lambda x: x[0])
        bot1_val, bot1_idx = trough_values[0]
        bot2_val, bot2_idx = trough_values[1]
        if bot1_idx != bot2_idx:
            t1, t2 = sorted([bot1_idx, bot2_idx])
            if t2 - t1 > 1:
                mid_segment = arr[t1:t2+1]
                peak_idx_rel = np.argmax(mid_segment)
                peak_idx = t1 + peak_idx_rel
                peak_val = arr[peak_idx]
                if peak_idx > t1 and peak_idx < t2:
                    depth_diff = abs(arr[bot1_idx] - arr[bot2_idx])
                    max_depth = max(abs(arr[bot1_idx]), abs(arr[bot2_idx]))
                    depth_ratio = depth_diff / max_depth if max_depth != 0 else 0.0
                    if depth_ratio <= similarity_tol:
                        pattern_type = "double bottom"
                        key_points = [t1, peak_idx, t2]

    if pattern_type:
        return pattern_type, key_points
    else:
        return None, []

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
    
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    for idx in patterns:
        if idx >= len(price_series) or idx + m > len(price_series):
            continue
            
        start_time = price_series.index[idx]
        end_time = price_series.index[min(idx + m - 1, len(price_series) - 1)]
        
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

def plotly_visualize_double_patterns(price_series, patterns, m):
    """
    Visualize the price series with detected double top/bottom patterns using Plotly.
    
    Parameters:
    - price_series: Pandas Series of price data.
    - patterns: List of tuples (start_idx, pattern_type, key_points), where key_points are relative indices.
    - m: Window length used for pattern detection.
    
    Returns:
    - fig: Plotly figure object.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    for idx, pattern_type, key_points in patterns:
        if idx >= len(price_series) or idx + m > len(price_series):
            continue
            
        start_time = price_series.index[idx]
        end_time = price_series.index[min(idx + m - 1, len(price_series) - 1)]
        
        color = "rgba(255, 165, 0, 0.3)" if pattern_type == "double top" else "rgba(0, 255, 0, 0.3)"
        
        fig.add_shape(
            type="rect",
            x0=start_time,
            x1=end_time,
            y0=min(price_series.iloc[idx:idx+m]),
            y1=max(price_series.iloc[idx:idx+m]),
            fillcolor=color,
            opacity=0.7,
            layer="below",
            line_width=0,
        )
        
        midpoint = price_series.index[idx + (m // 2)]
        fig.add_annotation(
            x=midpoint,
            y=price_series.iloc[idx + (m // 2)],
            text=pattern_type.upper(),
            showarrow=False,
            font=dict(color="black", size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    fig.update_layout(
        title=f"Price Chart with Detected Double Top/Bottom Patterns",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plotly_visualize_double_pattern(segment, start_idx, pattern_info):
    """
    Visualize an individual double top/bottom pattern using Plotly.
    
    Parameters:
    - segment: Pandas Series representing the subsequence containing the pattern.
    - start_idx: Starting index of the pattern in the original price series.
    - pattern_info: Tuple (pattern_type, key_points) where key_points are relative indices within the segment.
    
    Returns:
    - fig: Plotly figure object showing the pattern with annotations.
    """
    fig = go.Figure()
    pattern_type, key_points = pattern_info
    
    if len(segment) == 0 or pattern_type is None or not key_points:
        return fig
    
    fig.add_trace(go.Scatter(
        x=segment.index,
        y=segment.values,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    marker_color = 'red' if pattern_type == 'double top' else 'green'
    
    if pattern_type == 'double top':
        fig.add_trace(go.Scatter(
            x=[segment.index[key_points[0]]],
            y=[segment.iloc[key_points[0]]],
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color=marker_color),
            text=['First Peak'],
            textposition='top center',
            name='First Peak'
        ))
        
        fig.add_trace(go.Scatter(
            x=[segment.index[key_points[1]]],
            y=[segment.iloc[key_points[1]]],
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=12, color=marker_color),
            text=['Valley'],
            textposition='bottom center',
            name='Valley'
        ))
        
        fig.add_trace(go.Scatter(
            x=[segment.index[key_points[2]]],
            y=[segment.iloc[key_points[2]]],
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color=marker_color),
            text=['Second Peak'],
            textposition='top center',
            name='Second Peak'
        ))
        
    else:
        fig.add_trace(go.Scatter(
            x=[segment.index[key_points[0]]],
            y=[segment.iloc[key_points[0]]],
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=12, color=marker_color),
            text=['First Trough'],
            textposition='bottom center',
            name='First Trough'
        ))
        
        fig.add_trace(go.Scatter(
            x=[segment.index[key_points[1]]],
            y=[segment.iloc[key_points[1]]],
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color=marker_color),
            text=['Peak'],
            textposition='top center',
            name='Peak'
        ))
        
        fig.add_trace(go.Scatter(
            x=[segment.index[key_points[2]]],
            y=[segment.iloc[key_points[2]]],
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=12, color=marker_color),
            text=['Second Trough'],
            textposition='bottom center',
            name='Second Trough'
        ))
    
    if pattern_type == 'double top':
        neckline_y = segment.iloc[key_points[1]]
        fig.add_shape(
            type="line",
            x0=segment.index[0],
            x1=segment.index[-1],
            y0=neckline_y,
            y1=neckline_y,
            line=dict(color="red", width=1, dash="dash"),
            name="Neckline"
        )
    elif pattern_type == 'double bottom':
        neckline_y = segment.iloc[key_points[1]]
        fig.add_shape(
            type="line",
            x0=segment.index[0],
            x1=segment.index[-1],
            y0=neckline_y,
            y1=neckline_y,
            line=dict(color="green", width=1, dash="dash"),
            name="Neckline"
        )
    
    fig.update_layout(
        title=f"{pattern_type.title()} Pattern",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        showlegend=True
    )
    
    return fig