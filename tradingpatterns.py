import pandas as pd
import numpy as np

def detect_head_shoulder(df, window=3):
    """
    Detect Head and Shoulder and Inverse Head and Shoulder patterns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        window (int): Rolling window size.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'head_shoulder_pattern' 
                      populated with detected patterns.
    """
    roll_window = window
    # Create rolling max/min for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    # Shift the High and Low series
    shifted_high_1    = df['High'].shift(1)
    shifted_high_neg1 = df['High'].shift(-1)
    shifted_low_1     = df['Low'].shift(1)
    shifted_low_neg1  = df['Low'].shift(-1)
    
    # Use numpy arrays for element-wise comparison
    mask_head_shoulder_np = (
        (df['high_roll_max'].values > shifted_high_1.values) &
        (df['high_roll_max'].values > shifted_high_neg1.values) &
        (df['High'].values < shifted_high_1.values) &
        (df['High'].values < shifted_high_neg1.values)
    )
    mask_inv_head_shoulder_np = (
        (df['low_roll_min'].values < shifted_low_1.values) &
        (df['low_roll_min'].values < shifted_low_neg1.values) &
        (df['Low'].values > shifted_low_1.values) &
        (df['Low'].values > shifted_low_neg1.values)
    )
    # Convert numpy masks to pandas Series aligned with df.index
    mask_head_shoulder = pd.Series(mask_head_shoulder_np, index=df.index)
    mask_inv_head_shoulder = pd.Series(mask_inv_head_shoulder_np, index=df.index)
    
    # Initialize column with object dtype
    df['head_shoulder_pattern'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    
    return df

def detect_multiple_tops_bottoms(df, window=3):
    """
    Detect multiple top (Double Top) and multiple bottom (Double Bottom) patterns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int): Rolling window size.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'multiple_top_bottom_pattern' 
                      populated with detected patterns.
    """
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['close_roll_max'] = df['Close'].rolling(window=roll_window).max()
    df['close_roll_min'] = df['Close'].rolling(window=roll_window).min()
    
    shifted_high_1 = df['High'].shift(1)
    shifted_close_1 = df['Close'].shift(1)
    shifted_low_1 = df['Low'].shift(1)
    
    mask_top_np = (df['high_roll_max'].values >= shifted_high_1.values) & (df['close_roll_max'].values < shifted_close_1.values)
    mask_bottom_np = (df['low_roll_min'].values <= shifted_low_1.values) & (df['close_roll_min'].values > shifted_close_1.values)
    
    mask_top = pd.Series(mask_top_np, index=df.index)
    mask_bottom = pd.Series(mask_bottom_np, index=df.index)
    
    df['multiple_top_bottom_pattern'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'
    
    return df

def calculate_support_resistance(df, window=3):
    """
    Calculate support and resistance levels using rolling mean and standard deviation.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        window (int): Rolling window size.
    
    Returns:
        pd.DataFrame: The DataFrame with new 'support' and 'resistance' columns.
    """
    roll_window = window
    std_dev = 2
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    mean_high = df['High'].rolling(window=roll_window).mean()
    std_high = df['High'].rolling(window=roll_window).std()
    mean_low = df['Low'].rolling(window=roll_window).mean()
    std_low = df['Low'].rolling(window=roll_window).std()
    
    df['support'] = mean_low - std_dev * std_low
    df['resistance'] = mean_high + std_dev * std_high
    
    return df

def detect_triangle_pattern(df, window=3):
    """
    Detect ascending and descending triangle patterns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int): Rolling window size.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'triangle_pattern' 
                      populated with detected patterns.
    """
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    shifted_high_1 = df['High'].shift(1)
    shifted_low_1 = df['Low'].shift(1)
    shifted_close_1 = df['Close'].shift(1)
    
    mask_asc_np = (
        (df['high_roll_max'].values >= shifted_high_1.values) &
        (df['low_roll_min'].values <= shifted_low_1.values) &
        (df['Close'].values > shifted_close_1.values)
    )
    mask_desc_np = (
        (df['high_roll_max'].values <= shifted_high_1.values) &
        (df['low_roll_min'].values >= shifted_low_1.values) &
        (df['Close'].values < shifted_close_1.values)
    )
    
    mask_asc = pd.Series(mask_asc_np, index=df.index)
    mask_desc = pd.Series(mask_desc_np, index=df.index)
    
    df['triangle_pattern'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'
    
    return df

def detect_wedge(df, window=3):
    """
    Detect wedge patterns (Wedge Up and Wedge Down).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int): Rolling window size.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'wedge_pattern' 
                      populated with detected patterns.
    """
    roll_window = window
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    # Use .iloc for proper indexing in rolling.apply
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(
        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0
    )
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(
        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0
    )
    
    shifted_high_1 = df['High'].shift(1)
    shifted_low_1 = df['Low'].shift(1)
    
    mask_wedge_up_np = (
        (df['high_roll_max'].values >= shifted_high_1.values) &
        (df['low_roll_min'].values <= shifted_low_1.values) &
        (df['trend_high'].values == 1) &
        (df['trend_low'].values == 1)
    )
    mask_wedge_down_np = (
        (df['high_roll_max'].values <= shifted_high_1.values) &
        (df['low_roll_min'].values >= shifted_low_1.values) &
        (df['trend_high'].values == -1) &
        (df['trend_low'].values == -1)
    )
    
    mask_wedge_up = pd.Series(mask_wedge_up_np, index=df.index)
    mask_wedge_down = pd.Series(mask_wedge_down_np, index=df.index)
    
    df['wedge_pattern'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    
    return df

def detect_channel(df, window=3):
    """
    Detect channel patterns (Channel Up and Channel Down).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int): Rolling window size.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'channel_pattern' 
                      populated with detected patterns.
    """
    roll_window = window
    channel_range = 0.1
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(
        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0
    )
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(
        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1 if (x.iloc[-1] - x.iloc[0]) < 0 else 0
    )
    
    shifted_high_1 = df['High'].shift(1)
    shifted_low_1 = df['Low'].shift(1)
    
    mask_channel_up_np = (
        (df['high_roll_max'].values >= shifted_high_1.values) &
        (df['low_roll_min'].values <= shifted_low_1.values) &
        ((df['high_roll_max'].values - df['low_roll_min'].values) <= channel_range * (df['high_roll_max'].values + df['low_roll_min'].values) / 2) &
        (df['trend_high'].values == 1) &
        (df['trend_low'].values == 1)
    )
    mask_channel_down_np = (
        (df['high_roll_max'].values <= shifted_high_1.values) &
        (df['low_roll_min'].values >= shifted_low_1.values) &
        ((df['high_roll_max'].values - df['low_roll_min'].values) <= channel_range * (df['high_roll_max'].values + df['low_roll_min'].values) / 2) &
        (df['trend_high'].values == -1) &
        (df['trend_low'].values == -1)
    )
    
    mask_channel_up = pd.Series(mask_channel_up_np, index=df.index)
    mask_channel_down = pd.Series(mask_channel_down_np, index=df.index)
    
    df['channel_pattern'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    
    return df

def detect_double_top_bottom(df, window=3, threshold=0.05):
    """
    Detect Double Top and Double Bottom patterns.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        window (int): Rolling window size.
        threshold (float): Threshold to check for the pattern range.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'double_pattern' 
                      populated with detected patterns.
    """
    roll_window = window
    range_threshold = threshold

    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    
    shifted_high_1 = df['High'].shift(1)
    shifted_high_neg1 = df['High'].shift(-1)
    shifted_low_1 = df['Low'].shift(1)
    shifted_low_neg1 = df['Low'].shift(-1)
    
    mask_double_top_np = (
        (df['high_roll_max'].values >= shifted_high_1.values) &
        (df['high_roll_max'].values >= shifted_high_neg1.values) &
        (df['High'].values < shifted_high_1.values) &
        (df['High'].values < shifted_high_neg1.values) &
        ((shifted_high_1.values - shifted_low_1.values) <= range_threshold * (shifted_high_1.values + shifted_low_1.values) / 2) &
        ((shifted_high_neg1.values - shifted_low_neg1.values) <= range_threshold * (shifted_high_neg1.values + shifted_low_neg1.values) / 2)
    )
    mask_double_bottom_np = (
        (df['low_roll_min'].values <= shifted_low_1.values) &
        (df['low_roll_min'].values <= shifted_low_neg1.values) &
        (df['Low'].values > shifted_low_1.values) &
        (df['Low'].values > shifted_low_neg1.values) &
        ((shifted_high_1.values - shifted_low_1.values) <= range_threshold * (shifted_high_1.values + shifted_low_1.values) / 2) &
        ((shifted_high_neg1.values - shifted_low_neg1.values) <= range_threshold * (shifted_high_neg1.values + shifted_low_neg1.values) / 2)
    )

    mask_double_top = pd.Series(mask_double_top_np, index=df.index)
    mask_double_bottom = pd.Series(mask_double_bottom_np, index=df.index)
    
    df['double_pattern'] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'
    
    return df

def detect_trendline(df, window=2):
    """
    Detect trendlines using linear regression on the Close price.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Close' column.
        window (int): Rolling window size for linear regression.
    
    Returns:
        pd.DataFrame: The DataFrame with new columns 'slope', 'intercept', 'support', and 'resistance'.
    """
    roll_window = window
    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(window, len(df)):
        x = np.array(range(i - window, i))
        y = df['Close'].iloc[i - window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    mask_support = df['slope'] > 0
    mask_resistance = df['slope'] < 0

    df['support'] = np.nan
    df['resistance'] = np.nan

    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']

    return df

def find_pivots(df):
    """
    Identify pivot points (higher high, lower low, lower high, higher low) in the data.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'high' and 'low' columns.
    
    Returns:
        pd.DataFrame: The DataFrame with a new column 'signal' containing pivot signals.
    """
    high_diffs = df['high'].diff()
    low_diffs = df['low'].diff()

    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'

    return df
