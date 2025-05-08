import pandas as pd
import numpy as np
import yfinance as yf
import stumpy
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt

def fetch_price_data(symbol: str, period: str = '6mo', interval: str = '1h', start_date=None, end_date=None) -> pd.DataFrame:
    """
    Fetch historical price data for the given ticker and timeframe using yfinance.
    - symbol: Stock symbol (e.g. 'AAPL').
    - period: Lookback period (e.g. '6mo', '1y').
    - interval: Data frequency (e.g. '1h' for hourly).
    - start_date: Optional start date for custom date range
    - end_date: Optional end date for custom date range
    Returns: DataFrame with Datetime index and OHLCV columns (Open, High, Low, Close, Adj Close, Volume).
    """
    try:
        # Use yfinance to download data with either period or date range
        if start_date is not None and end_date is not None:
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        else:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if df.empty:
            return None
            
        # Ensure data is properly sorted by time
        df.sort_index(inplace=True)
        
        # Return the Close price series
        return df['Close']
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def compute_matrix_profile(price_series: np.ndarray, window_size: int):
    """
    Compute the matrix profile of the price series for the given subsequence window length.
    - price_series: 1D array or pandas Series of prices (e.g., closing prices).
    - window_size: Length of the subsequence window for motif search.
    Returns: A tuple (profile_distances, profile_indices) where:
      profile_distances is a 1D numpy array of matrix profile distances,
      profile_indices is a 1D numpy array of indices of the nearest neighbor for each subsequence.
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
    
    # Ensure input is a numpy array of type float (for STUMPY)
    n = len(values)
    if window_size >= n:
        raise ValueError("Window size must be smaller than the length of the time series.")
    
    # Compute matrix profile using stumpy.stump (self-join on the time series)
    mp = stumpy.stump(values, m=window_size)
    # The result mp is an array of shape (n - window_size + 1, 4):
    # mp[:,0] = matrix profile distances, mp[:,1] = index of nearest neighbor
    profile_distances = mp[:, 0]
    profile_indices = mp[:, 1].astype(int)
    return profile_distances, profile_indices

def extract_motif_candidates(profile_distances: np.ndarray, profile_indices: np.ndarray,
                             window_size: int, k: int = 3):
    """
    Extract the top-k non-overlapping motif pairs from the matrix profile.
    - profile_distances: 1D array of matrix profile distances.
    - profile_indices: 1D array of nearest neighbor indices for each subsequence.
    - window_size: The subsequence length used in matrix profile computation.
    - k: Number of motif pairs to retrieve.
    Returns: List of tuples [(i, j, dist), ...] for each motif pair, where:
      i and j are the start indices of the two motif subsequences, and dist is their distance.
    """
    motifs = []
    # Make a copy of the profile distances so we can modify it without affecting original
    distance = profile_distances.copy()
    n = len(distance)
    
    for _ in range(k):
        # Find the smallest distance in the profile
        idx = np.nanargmin(distance)  # use nanargmin to ignore any NaN (if present)
        min_dist = distance[idx]
        if not np.isfinite(min_dist):
            break  # no more valid motifs (we've exhausted all non-overlapping regions)
        
        # Get the nearest neighbor index for this subsequence
        j = int(profile_indices[idx])
        i = int(idx)
        
        if i == j:
            # This would indicate a self-match (shouldn't happen with proper matrix profile)
            distance[idx] = np.inf
            continue
            
        # Ensure i < j for consistency in ordering
        if j < i:
            i, j = j, i
            
        motifs.append((i, j, min_dist))
        
        # Exclude a window-length region around each of these indices from further consideration
        start_i = max(0, i - window_size + 1)
        end_i = min(n, i + window_size - 1)
        start_j = max(0, j - window_size + 1)
        end_j = min(n, j + window_size - 1)
        
        distance[start_i:end_i+1] = np.inf
        distance[start_j:end_j+1] = np.inf
        
    return motifs

def detect_extrema(series: np.ndarray, order: int = 3):
    """
    Identify local extrema (peaks and troughs) in the given series.
    - series: 1D array of values (e.g., price subsequence).
    - order: How many points on each side to use for comparison when finding extrema.
    Returns: (peaks_idx, troughs_idx) as arrays of indices for local maxima and local minima.
    """
    if len(series) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
        
    # Use argrelextrema to find indices of local maxima and minima
    peaks_idx = argrelextrema(series, np.greater, order=order)[0]
    troughs_idx = argrelextrema(series, np.less, order=order)[0]
    
    # Check the edges manually: include index 0 or last index if they qualify as extremum
    n = len(series)
    if n < 2:
        return peaks_idx, troughs_idx  # trivial case, no interior points
        
    # Edge at start
    if n > 1:
        if series[0] > series[1]:
            # Insert at beginning if not already present
            if len(peaks_idx) == 0 or peaks_idx[0] != 0:
                peaks_idx = np.insert(peaks_idx, 0, 0)
        if series[0] < series[1]:
            if len(troughs_idx) == 0 or troughs_idx[0] != 0:
                troughs_idx = np.insert(troughs_idx, 0, 0)
                
    # Edge at end
    if series[-1] > series[-2]:
        if len(peaks_idx) == 0 or peaks_idx[-1] != n-1:
            peaks_idx = np.append(peaks_idx, n-1)
    if series[-1] < series[-2]:
        if len(troughs_idx) == 0 or troughs_idx[-1] != n-1:
            troughs_idx = np.append(troughs_idx, n-1)
            
    return peaks_idx, troughs_idx

def check_wedge_pattern(subsequence: np.ndarray, peaks_idx: np.ndarray, troughs_idx: np.ndarray,
                        slope_tolerance: float = 0.1):
    """
    Determine if a given price subsequence represents a rising wedge, falling wedge, or none.
    - subsequence: 1D array of price values (e.g., closing prices) for the candidate pattern.
    - peaks_idx: Indices of local maxima within the subsequence.
    - troughs_idx: Indices of local minima within the subsequence.
    - slope_tolerance: Tolerance (as a fraction) for slope difference to consider lines converging.
    Returns: 'Rising Wedge', 'Falling Wedge', or None.
    Also returns the computed slope of the peak trendline and trough trendline for reference.
    """
    # Need at least two peaks and two troughs to form converging trend lines
    if len(peaks_idx) < 2 or len(troughs_idx) < 2:
        return None, 0.0, 0.0  # Not enough extrema to form a wedge pattern
        
    # Fit a line (linear regression) through all peak points
    # Using numpy polyfit with x as the index (time) and y as the price
    x_peaks = peaks_idx
    y_peaks = subsequence[peaks_idx]
    x_troughs = troughs_idx
    y_troughs = subsequence[troughs_idx]
    
    # If all x are the same (unlikely if peaks_idx are distinct), can't fit; check len instead
    slope_peaks, intercept_peaks = np.polyfit(x_peaks, y_peaks, 1)
    slope_troughs, intercept_troughs = np.polyfit(x_troughs, y_troughs, 1)
    
    # Classify based on slopes
    pattern_type = None
    
    # Check for rising wedge: both trendlines slope upward
    if slope_peaks > 0 and slope_troughs > 0:
        # The lower (trough) line should have a larger slope than the upper (peak) line
        # Use tolerance: require trough slope >= peak slope * (1 + tolerance)
        if slope_troughs >= slope_peaks * (1 + slope_tolerance):
            pattern_type = "Rising Wedge"
            
    # Check for falling wedge: both trendlines slope downward
    if slope_peaks < 0 and slope_troughs < 0:
        # The upper (peak) line should have a more negative slope (steeper down) than the lower line
        if abs(slope_peaks) >= abs(slope_troughs) * (1 + slope_tolerance):
            pattern_type = "Falling Wedge"
            
    return pattern_type, slope_peaks, slope_troughs

def safe_date_format(date):
    """
    Safely convert a date or timestamp to a string format, handling various input types.
    """
    try:
        if isinstance(date, pd.Timestamp):
            return date.strftime("%Y-%m-%d %H:%M")
        elif isinstance(date, (datetime, np.datetime64)):
            return pd.Timestamp(date).strftime("%Y-%m-%d %H:%M")
        else:
            return str(date)
    except:
        return "Unknown date"

def plotly_visualize_wedge_patterns(df: pd.Series, patterns: list, window_size: int):
    """
    Create an overview Plotly line chart of the price series with wedge patterns highlighted.
    - df: Series containing the price data (expects a DateTime index).
    - patterns: List of pattern dictionaries, each with keys like 'start_idx', 'end_idx', 'type'.
    - window_size: The window size used to detect patterns
    Returns: Plotly Figure object with the price series and shaded regions for each wedge.
    """
    fig = go.Figure()
    
    # Add the price line - ensure we're accessing the correct data regardless of Series or DataFrame
    if isinstance(df, pd.DataFrame):
        x_values = df.index
        y_values = df.iloc[:, 0].values  # Use first column
    else:
        x_values = df.index
        y_values = df.values
        
    # Create the price line trace
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Close Price',
                             line=dict(color='blue')))
                             
    # Add shaded regions for each pattern
    for pat in patterns:
        # Ensure indices are within bounds
        if pat['start_idx'] >= len(df) or pat['end_idx'] >= len(df):
            continue
            
        start = df.index[pat['start_idx']]
        end = df.index[pat['end_idx']]
        pat_type = pat['type']
        
        # Color by pattern type: rising wedge (bearish) in red, falling wedge (bullish) in green
        color = 'rgba(255,0,0,0.2)' if pat_type == "Rising Wedge" else 'rgba(0,128,0,0.2)'
        
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.3, line_width=0,
                      annotation_text=pat_type, annotation_position="top left")
                      
    # Set layout with explicit dimensions and margin
    fig.update_layout(
        title_text="Detected Wedge Patterns Overview",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"  # Use a clean white template
    )
    return fig

def plotly_visualize_wedge_pattern(df: pd.Series, pattern: dict, window_size: int):
    """
    Create a detailed Plotly chart for a single wedge pattern with trendlines.
    - df: Series with price data (DateTime index).
    - pattern: Dictionary with details of the pattern.
    - window_size: Window size used to detect patterns
    Returns: Plotly Figure object zoomed in on the pattern with annotations.
    """
    start_idx = pattern['start_idx']
    end_idx = pattern['end_idx']
    pat_type = pattern['type']
    
    # Define a plotting range (add some padding before and after the pattern for context)
    pad = max(5, int(0.1 * (end_idx - start_idx)))  # 5 points or 10% of pattern length
    plot_start = max(0, start_idx - pad)
    plot_end = min(len(df) - 1, end_idx + pad)
    
    # Get data slice for plotting and handle Series vs DataFrame
    if isinstance(df, pd.DataFrame):
        data_slice = df.iloc[plot_start:plot_end + 1, 0]  # First column if DataFrame
    else:
        data_slice = df.iloc[plot_start:plot_end + 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_slice.index, y=data_slice.values, mode='lines', 
                             name='Close Price', line=dict(color='blue')))
                             
    # Plot trendlines: connect first and last peak, and first and last trough (in the pattern region)
    peaks_idx = pattern['peaks_idx']
    troughs_idx = pattern['troughs_idx']
    
    # Function to safely get y values for points
    def get_y_value(df, idx):
        if isinstance(df, pd.DataFrame):
            return float(df.iloc[idx, 0])
        else:
            return float(df.iloc[idx])
    
    if len(peaks_idx) >= 2:
        # Absolute times and prices for first and last peak in the pattern
        x0 = df.index[peaks_idx[0] + start_idx] if peaks_idx[0] < (end_idx - start_idx + 1) else df.index[start_idx]
        x1 = df.index[peaks_idx[-1] + start_idx]
        y0 = get_y_value(df, peaks_idx[0] + start_idx)
        y1 = get_y_value(df, peaks_idx[-1] + start_idx)
        
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', name='Resistance Line',
                                 line=dict(color='firebrick', width=2, dash='dash')))
                                 
    if len(troughs_idx) >= 2:
        x0 = df.index[troughs_idx[0] + start_idx]
        x1 = df.index[troughs_idx[-1] + start_idx]
        y0 = get_y_value(df, troughs_idx[0] + start_idx)
        y1 = get_y_value(df, troughs_idx[-1] + start_idx)
        
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', name='Support Line',
                                 line=dict(color='green', width=2, dash='dash')))
                                 
    # Mark the extrema points on the chart for clarity
    peak_times = [df.index[i + start_idx] for i in pattern['peaks_idx']]
    peak_values = [get_y_value(df, i + start_idx) for i in pattern['peaks_idx']]
    
    trough_times = [df.index[i + start_idx] for i in pattern['troughs_idx']]
    trough_values = [get_y_value(df, i + start_idx) for i in pattern['troughs_idx']]
    
    fig.add_trace(go.Scatter(x=peak_times, y=peak_values, mode='markers', 
                             name='Peaks', marker_symbol='triangle-up', 
                             marker_color='firebrick', marker_size=8))
                             
    fig.add_trace(go.Scatter(x=trough_times, y=trough_values, mode='markers', 
                             name='Troughs', marker_symbol='triangle-down', 
                             marker_color='green', marker_size=8))
                             
    # If breakout point is identified, annotate it
    breakout_idx = pattern.get('breakout_idx')
    breakout_dir = pattern.get('breakout_dir')
    
    if breakout_idx is not None and breakout_idx < len(df):
        xb = df.index[breakout_idx]
        yb = get_y_value(df, breakout_idx)
        
        # Text for breakout direction
        breakout_text = f"Breakout {breakout_dir}"
        
        # Position annotation: above point for downward break, below point for upward break
        annot_yoffset = -30 if breakout_dir == 'Down' else 30
        
        fig.add_annotation(x=xb, y=yb, text=breakout_text, showarrow=True, 
                           arrowhead=1, ax=0, ay=annot_yoffset,
                           arrowcolor=('red' if breakout_dir == 'Down' else 'green'))
                           
    # Configure layout
    title = f"{pat_type} (Start: {safe_date_format(df.index[start_idx])} to End: {safe_date_format(df.index[end_idx])})"
    
    # Set layout with explicit dimensions and margin
    fig.update_layout(
        title_text=title,
        xaxis_title="Date", 
        yaxis_title="Price",
        height=500,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"  # Use a clean white template
    )
                     
    return fig

def visualize_wedge_patterns(price_series: pd.Series, patterns: list, window_size: int):
    """
    Plot the price series with highlighted wedge patterns using Matplotlib.
    - price_series: Pandas Series of the price data (indexed by time).
    - patterns: List of pattern dictionaries, each with keys like 'start_idx', 'end_idx', 'type', 'peaks_idx', 'troughs_idx'.
    - window_size: The length of each pattern subsequence (same as used for motif search).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(price_series.index, price_series.values, label="Price", color='blue')
    ax = plt.gca()
    
    # Define markers and colors for peaks/troughs
    peak_marker = '^'   # triangle up for peaks
    trough_marker = 'v'  # triangle down for troughs
    
    for pat in patterns:
        start_idx = pat['start_idx']
        end_idx = pat['end_idx']
        pat_type = pat['type']
        
        # Shade the region of the pattern
        color = 'red' if pat_type == "Rising Wedge" else 'green'
        alpha = 0.2
        ax.axvspan(price_series.index[start_idx], price_series.index[end_idx], 
                  color=color, alpha=alpha)
        
        # Plot trendlines for peaks and troughs
        peaks_idx = pat['peaks_idx']
        troughs_idx = pat['troughs_idx']
        
        if len(peaks_idx) >= 2:
            # Get absolute indices for the peaks
            abs_peak_idxs = [start_idx + idx for idx in peaks_idx]
            # Plot the peak trendline
            x_peak_points = [price_series.index[idx] for idx in abs_peak_idxs]
            y_peak_points = [price_series.iloc[idx] for idx in abs_peak_idxs]
            # Use first and last peak for the trendline
            plt.plot([x_peak_points[0], x_peak_points[-1]], 
                     [y_peak_points[0], y_peak_points[-1]], 
                     'r--', linewidth=2, label=f"Peak Trendline ({pat_type})")
            # Mark all peaks with triangle markers
            for i in range(len(abs_peak_idxs)):
                ax.plot(x_peak_points[i], y_peak_points[i], 
                        peak_marker, color='red', markersize=8)
        
        if len(troughs_idx) >= 2:
            # Get absolute indices for the troughs
            abs_trough_idxs = [start_idx + idx for idx in troughs_idx]
            # Plot the trough trendline
            x_trough_points = [price_series.index[idx] for idx in abs_trough_idxs]
            y_trough_points = [price_series.iloc[idx] for idx in abs_trough_idxs]
            # Use first and last trough for the trendline
            plt.plot([x_trough_points[0], x_trough_points[-1]], 
                     [y_trough_points[0], y_trough_points[-1]], 
                     'g--', linewidth=2, label=f"Trough Trendline ({pat_type})")
            # Mark all troughs with triangle markers
            for i in range(len(abs_trough_idxs)):
                ax.plot(x_trough_points[i], y_trough_points[i], 
                        trough_marker, color='green', markersize=8)
        
        # Annotate the pattern type in the middle of the pattern region
        mid_idx = start_idx + (end_idx - start_idx) // 2
        y_pos = price_series.iloc[mid_idx]
        y_offset = (price_series.max() - price_series.min()) * 0.05  # 5% of the price range
        
        if pat_type == "Rising Wedge":
            y_pos += y_offset  # Position above for rising wedge
            va = 'bottom'
        else:  # Falling Wedge
            y_pos -= y_offset  # Position below for falling wedge
            va = 'top'
            
        ax.text(price_series.index[mid_idx], y_pos, 
                pat_type, color='black', fontsize=10, fontweight='bold',
                ha='center', va=va)
        
        # Mark breakout point if available
        if 'breakout_idx' in pat and pat['breakout_idx'] is not None:
            breakout_idx = pat['breakout_idx']
            if breakout_idx < len(price_series):
                breakout_dir = pat['breakout_dir']
                color = 'red' if breakout_dir == 'Down' else 'green'
                marker = 'v' if breakout_dir == 'Down' else '^'
                ax.plot(price_series.index[breakout_idx], price_series.iloc[breakout_idx],
                        marker=marker, color=color, markersize=10, markeredgecolor='black')
                
                # Add breakout annotation
                ax.annotate(f"Breakout {breakout_dir}", 
                            xy=(price_series.index[breakout_idx], price_series.iloc[breakout_idx]),
                            xytext=(15, 15 if breakout_dir == 'Up' else -15),
                            textcoords='offset points',
                            arrowprops=dict(facecolor=color, shrink=0.05),
                            fontsize=9)
    
    plt.title("Detected Wedge Patterns in Price Series")
    plt.xlabel("Time")
    plt.ylabel("Price")
    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    
    return plt.gcf()  # Return the figure for display in Streamlit

def visualize_wedge_patterns_simple(price_series: pd.Series, patterns: list, window_size: int):
    """
    Plot the price series with highlighted wedge patterns in the same style as double_top_bottom.py.
    - price_series: Pandas Series of the price data (indexed by time).
    - patterns: List of pattern dictionaries, each with keys like 'start_idx', 'end_idx', 'type', etc.
    - window_size: The length of each pattern subsequence.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(price_series.index, price_series.values, label="Price", color='blue')
    ax = plt.gca()
    
    # Define markers for peaks/troughs
    peak_marker = '^'   # triangle up for peaks
    trough_marker = 'v'  # triangle down for troughs
    
    for pattern in patterns:
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        pattern_type = pattern['type']
        peaks_idx = pattern['peaks_idx']
        troughs_idx = pattern['troughs_idx']
        
        # Shade the region of the pattern
        ax.axvspan(price_series.index[start_idx], price_series.index[end_idx], 
                   color='red' if pattern_type == "Rising Wedge" else 'green', 
                   alpha=0.3)
        
        # Plot markers for peaks and troughs
        for peak_idx in peaks_idx:
            abs_peak_idx = start_idx + peak_idx
            ax.plot(price_series.index[abs_peak_idx], price_series.iloc[abs_peak_idx], 
                    peak_marker, color='red', markersize=8)
            
        for trough_idx in troughs_idx:
            abs_trough_idx = start_idx + trough_idx
            ax.plot(price_series.index[abs_trough_idx], price_series.iloc[abs_trough_idx], 
                    trough_marker, color='green', markersize=8)
        
        # Add trendlines
        if len(peaks_idx) >= 2:
            # Get the first and last peak point for trendline
            first_peak_idx = start_idx + peaks_idx[0]
            last_peak_idx = start_idx + peaks_idx[-1]
            
            # Draw peak trendline
            plt.plot([price_series.index[first_peak_idx], price_series.index[last_peak_idx]],
                     [price_series.iloc[first_peak_idx], price_series.iloc[last_peak_idx]],
                     'r--', linewidth=2)
            
        if len(troughs_idx) >= 2:
            # Get the first and last trough point for trendline
            first_trough_idx = start_idx + troughs_idx[0]
            last_trough_idx = start_idx + troughs_idx[-1]
            
            # Draw trough trendline
            plt.plot([price_series.index[first_trough_idx], price_series.index[last_trough_idx]],
                     [price_series.iloc[first_trough_idx], price_series.iloc[last_trough_idx]],
                     'g--', linewidth=2)
        
        # Annotate the pattern type text near the middle of the pattern region
        mid_idx = start_idx + (end_idx - start_idx) // 2
        plt.text(price_series.index[mid_idx], price_series.iloc[mid_idx], 
                pattern_type, 
                color='black', fontsize=9, fontweight='bold',
                ha='center', va='bottom' if pattern_type == "Rising Wedge" else 'top')
        
        # Mark breakout point if available
        if pattern.get('breakout_idx') is not None and pattern.get('breakout_dir') is not None:
            breakout_idx = pattern['breakout_idx']
            if breakout_idx < len(price_series):
                breakout_marker = 'v' if pattern['breakout_dir'] == 'Down' else '^'
                breakout_color = 'red' if pattern['breakout_dir'] == 'Down' else 'green'
                
                plt.plot(price_series.index[breakout_idx], price_series.iloc[breakout_idx],
                         breakout_marker, color=breakout_color, markersize=10, markeredgecolor='black')
                
                plt.text(price_series.index[breakout_idx], price_series.iloc[breakout_idx],
                         f"Breakout {pattern['breakout_dir']}", 
                         color=breakout_color, fontsize=8, fontweight='bold',
                         ha='center', va='top' if pattern['breakout_dir'] == 'Down' else 'bottom')
    
    plt.title("Detected Wedge Patterns in Price Series")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    
    return plt.gcf()  # Return the figure for Streamlit display
