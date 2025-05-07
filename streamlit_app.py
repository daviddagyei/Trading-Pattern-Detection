import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import the streamlit_helpers functions instead of head_shoulder.py
from streamlit_helpers import (
    fetch_price_data, 
    compute_matrix_profile,
    detect_extrema,
    check_head_and_shoulders,
    extract_motif_candidates,
    safe_date_format,
    plotly_visualize_price_with_patterns,
    plotly_visualize_pattern
)

# Set page configuration
st.set_page_config(
    page_title="Head and Shoulders Pattern Detector",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📊 Head and Shoulders Pattern Detector")
st.markdown("""
This app detects Head and Shoulders patterns in stock price data using matrix profile algorithms.
Customize your analysis parameters below and visualize the detected patterns.
""")

# Create sidebar for parameter inputs
st.sidebar.header("Parameters")

# Stock selection section
st.sidebar.subheader("Stock Selection")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")

# Time frame section
st.sidebar.subheader("Time Frame")

# Add a radio button to select between period-based and date range selection
time_selection = st.sidebar.radio("Select Time Frame Method", ["Predefined Period", "Custom Date Range"])

if time_selection == "Predefined Period":
    # Predefined period selection (as before)
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", 
                       "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", 
                       "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=5)
    interval_code = interval_options[interval]

    period_options = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", 
                     "3 months": "3mo", "6 months": "6mo", "1 year": "1y", 
                     "2 years": "2y", "5 years": "5y", "10 years": "10y", "max": "max"}
    period = st.sidebar.selectbox("Period", options=list(period_options.keys()), index=4)
    period_code = period_options[period]
    
    # Set date range variables to None
    start_date = None
    end_date = None
    
else:  # Custom Date Range
    # Interval selection remains the same
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", 
                       "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", 
                       "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=5)
    interval_code = interval_options[interval]
    
    # Date range pickers
    from datetime import datetime, timedelta
    
    # Default date range (end date is today, start date is 6 months ago)
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=180)  # 6 months ago
    
    start_date = st.sidebar.date_input("Start Date", value=default_start_date)
    end_date = st.sidebar.date_input("End Date", value=default_end_date)
    
    # Convert to string format YYYY-MM-DD for yfinance
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    
    # Set period_code to None as we're using date range
    period_code = None
    
    # Validate date range
    if start_date >= end_date:
        st.sidebar.error("End date must be after start date")

# Pattern detection parameters
st.sidebar.subheader("Pattern Detection Parameters")
window_length = st.sidebar.slider("Window Length (m)", min_value=10, max_value=100, value=30, 
                                help="Number of data points in each subsequence for matrix profile")
top_k = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value=3, 
                         help="Number of motif pairs to extract from the matrix profile")
order = st.sidebar.slider("Extrema Detection Order", min_value=1, max_value=10, value=3, 
                         help="Points to compare on each side for detecting local extrema")

# Pattern threshold parameters
st.sidebar.subheader("Pattern Thresholds")
shoulder_tol = st.sidebar.slider("Shoulder Height Tolerance", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                               help="Maximum allowed difference between shoulders (as a percentage of head height)")
trough_tol = st.sidebar.slider("Trough Alignment Tolerance", min_value=0.01, max_value=0.5, value=0.2, step=0.01,
                             help="Maximum allowed difference between troughs (as a percentage of the higher trough)")

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Credits
st.sidebar.markdown("---")
st.sidebar.caption("Created with ❤️ using Streamlit")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price Chart")
    price_chart_placeholder = st.empty()

with col2:
    st.subheader("Pattern Detection Stats")
    stats_placeholder = st.empty()

# Results section
st.markdown("---")
patterns_section = st.container()
with patterns_section:
    st.subheader("Detected Patterns")
    patterns_placeholder = st.empty()

# Define function to run the analysis when button is clicked
def run_head_shoulders_analysis():
    # Initialize local window_length from the global slider value
    # This fixes the UnboundLocalError by ensuring window_length is set before use
    local_window_length = window_length
    
    with st.spinner('Fetching price data...'):
        # Get price data
        try:
            prices = fetch_price_data(symbol=symbol, interval=interval_code, period=period_code, 
                                     start_date=start_date, end_date=end_date)
            
            # Check if we got data
            if prices is None or len(prices) == 0:
                if time_selection == "Predefined Period":
                    st.error(f"No data returned for symbol {symbol} with interval {interval} and period {period}.")
                else:
                    st.error(f"No data returned for symbol {symbol} with interval {interval} between {start_date} and {end_date}.")
                return
                
            st.session_state.prices = prices
            
            # Update price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices.index, 
                y=prices.values,
                mode='lines',
                name='Price'
            ))
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500
            )
            price_chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Check if we have enough data for the chosen window length
            if len(prices) <= local_window_length:
                st.error(f"Not enough data points ({len(prices)}) for the selected window length ({local_window_length}). Either choose a smaller window length or fetch more data with a longer period.")
                
                # Get first and last date safely
                first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
                
                # Get current price as float (scalar value) instead of Series
                current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
                
                stats_placeholder.info(f"""
                **Data Summary:**
                - Symbol: {symbol}
                - Data points: {len(prices)}
                - Date range: {first_date} to {last_date}
                - Current price: {current_price:.2f}
                
                **Recommendation:** Reduce window length to {max(5, len(prices) // 2)} or less.
                """)
                return
            
            # Get first and last date safely
            first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
            last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
            
            # Get current price as float (scalar value) instead of Series
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            
            stats_placeholder.info(f"""
            **Data Summary:**
            - Symbol: {symbol}
            - Data points: {len(prices)}
            - Date range: {first_date} to {last_date}
            - Current price: {current_price:.2f}
            """)
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            import traceback
            st.text(traceback.format_exc())
            return
    
    with st.spinner('Computing matrix profile...'):
        try:
            # Check for NaN values and handle them
            if prices.isna().any():
                st.warning(f"Found {prices.isna().sum()} NaN values in the data. These will be removed.")
                prices = prices.dropna()
                if len(prices) == 0:
                    st.error("After removing NaN values, no data remains. Please try a different stock or time period.")
                    return
            
            # Convert to numpy array
            price_array = np.asarray(prices.values, dtype=float)
            
            # Debug info about the array
            st.info(f"Price array shape: {price_array.shape}, Min: {price_array.min():.2f}, Max: {price_array.max():.2f}")
            
            # Check if the array contains all identical values
            value_range = price_array.max() - price_array.min()
            if value_range < 1e-5:
                st.error("The price data contains all identical values (or nearly identical). Matrix profile cannot be computed on constant data.")
                return
            
            # Validate window length is appropriate
            if len(price_array) <= 1:
                st.error(f"Not enough valid data points (only {len(price_array)}). Please select a different stock or time period.")
                return
            
            if local_window_length >= len(price_array):
                # Suggest a reasonable window size
                suggested_window = max(5, len(price_array) // 4)
                st.error(f"Window length ({local_window_length}) must be less than the number of data points ({len(price_array)}). Try using {suggested_window} instead.")
                # Auto-adjust the window length if it's too large
                local_window_length = suggested_window
                st.info(f"Automatically adjusted window length to {local_window_length}")
            
            # Ensure window length is at least 4 (minimum required for meaningful patterns)
            if local_window_length < 4:
                local_window_length = 4
                st.info(f"Window length adjusted to minimum value of 4")
            
            # Compute matrix profile with the validated window length
            st.info(f"Computing matrix profile with window length = {local_window_length}")
            mp = compute_matrix_profile(price_array, local_window_length)
            
            # Extract motif candidates
            motifs = extract_motif_candidates(mp, local_window_length, top_k=top_k)
            
            # Check if no motifs were found
            if not motifs:
                st.warning("No motifs found. Try adjusting parameters:")
                st.markdown("""
                **Suggestions:**
                - Try a different window length
                - Consider a different time period or interval
                - Check if the price series has enough variation
                """)
                return
                
            # Check each motif for head and shoulders pattern
            pattern_indices = []
            
            pattern_info = []
            for idx, (idx1, idx2) in enumerate(motifs):
                # Ensure indices are within bounds
                if idx1 >= len(prices) or idx2 >= len(prices) or idx1 + local_window_length > len(prices) or idx2 + local_window_length > len(prices):
                    st.warning(f"Skipping motif pair ({idx1}, {idx2}) due to index out of bounds")
                    continue
                
                # Check first subsequence
                segment1 = prices.iloc[idx1:idx1+local_window_length]
                peaks1, troughs1 = detect_extrema(segment1, order=min(order, local_window_length // 3))  # Adjust order based on window length
                is_pattern1 = check_head_and_shoulders(segment1, peaks1, troughs1, 
                                                     shoulder_tol=shoulder_tol, 
                                                     trough_tol=trough_tol)
                
                # Check second subsequence
                segment2 = prices.iloc[idx2:idx2+local_window_length]
                peaks2, troughs2 = detect_extrema(segment2, order=min(order, local_window_length // 3))  # Adjust order based on window length
                is_pattern2 = check_head_and_shoulders(segment2, peaks2, troughs2,
                                                     shoulder_tol=shoulder_tol,
                                                     trough_tol=trough_tol)
                
                # Format dates safely
                date1 = safe_date_format(prices.index[idx1]) if idx1 < len(prices.index) else "N/A"
                date2 = safe_date_format(prices.index[idx2]) if idx2 < len(prices.index) else "N/A"
                
                pattern_info.append({
                    "Motif": idx + 1,
                    "Subsequence 1 Index": idx1,
                    "Is Pattern 1": is_pattern1,
                    "Subsequence 1 Date": date1,
                    "Subsequence 2 Index": idx2,
                    "Is Pattern 2": is_pattern2,
                    "Subsequence 2 Date": date2
                })
                
                if is_pattern1:
                    pattern_indices.append(idx1)
                if is_pattern2:
                    pattern_indices.append(idx2)
                    
            # Display motif statistics
            st.dataframe(pd.DataFrame(pattern_info), use_container_width=True)
            
            # Visualize results
            if len(pattern_indices) > 0:
                # Overall price chart with patterns
                price_chart = plotly_visualize_price_with_patterns(prices, pattern_indices, local_window_length)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Individual pattern visualizations
                for idx in pattern_indices:
                    # Format date safely
                    pattern_date = safe_date_format(prices.index[idx]) if idx < len(prices.index) else "N/A"
                    st.subheader(f"Pattern at {pattern_date} (Index: {idx})")
                    
                    segment = prices.iloc[idx:idx+local_window_length]
                    peaks, troughs = detect_extrema(segment, order=min(order, local_window_length // 3))
                    is_pattern = check_head_and_shoulders(segment, peaks, troughs, 
                                                         shoulder_tol=shoulder_tol,
                                                         trough_tol=trough_tol)
                    pattern_fig = plotly_visualize_pattern(segment, idx, peaks, troughs, is_pattern)
                    st.plotly_chart(pattern_fig, use_container_width=True)
                    
                    # Add pattern details
                    if is_pattern:
                        arr = np.asarray(segment.values, dtype=float)
                        head_idx = max(peaks, key=lambda j: arr[j])
                        left_peaks = [j for j in peaks if j < head_idx]
                        right_peaks = [j for j in peaks if j > head_idx]
                        
                        if left_peaks and right_peaks:
                            left_sh_idx = max(left_peaks)
                            right_sh_idx = min(right_peaks)
                            
                            head_val = segment.iloc[head_idx]
                            left_val = segment.iloc[left_sh_idx]
                            right_val = segment.iloc[right_sh_idx]
                            
                            # Format dates safely
                            segment_start = safe_date_format(segment.index[0]) if len(segment.index) > 0 else "N/A"
                            segment_end = safe_date_format(segment.index[-1]) if len(segment.index) > 0 else "N/A"
                            
                            st.markdown(f"""
                            **Pattern Details:**
                            - Head height: {head_val:.2f}
                            - Left shoulder height: {left_val:.2f}
                            - Right shoulder height: {right_val:.2f}
                            - Shoulder height difference: {abs(left_val - right_val):.2f} ({abs(left_val - right_val)/head_val*100:.1f}% of head)
                            - Pattern start: {segment_start}
                            - Pattern end: {segment_end}
                            """)
            else:
                st.warning("No head and shoulders patterns detected with the current parameters.")
                st.markdown("""
                **Try adjusting the parameters:**
                - Increase the window length for larger patterns
                - Try different extrema detection order values
                - Adjust the tolerance thresholds
                """)
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            import traceback
            st.text(traceback.format_exc())

# Run the analysis if button is clicked
if run_analysis:
    run_head_shoulders_analysis()
else:
    # Welcome message on initial load
    st.info("""
    👈 Customize your analysis parameters in the sidebar and click "Run Analysis" to start.
    
    This app will:
    1. Fetch historical price data for your selected stock
    2. Compute the matrix profile to find recurring patterns
    3. Identify head and shoulders patterns based on your criteria
    4. Visualize detected patterns with detailed annotations
    """)