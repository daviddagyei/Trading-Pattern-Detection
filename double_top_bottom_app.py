import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import the streamlit_helpers functions
from streamlit_helpers import (
    fetch_price_data, 
    compute_matrix_profile,
    extract_motif_candidates,
    detect_extrema,
    check_double_top_bottom,
    safe_date_format,
    plotly_visualize_double_patterns,
    plotly_visualize_double_pattern
)

# Set page configuration
st.set_page_config(
    page_title="Double Top/Bottom Pattern Detector",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Double Top/Bottom Pattern Detector")
st.markdown("""
This app detects Double Top and Double Bottom patterns in stock price data using matrix profile algorithms.
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
    # Predefined period selection
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
similarity_tol = st.sidebar.slider("Peak/Trough Similarity Tolerance", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                help="Maximum allowed difference between peaks or troughs (as a percentage)")

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
def run_double_pattern_analysis():
    # Initialize local window_length from the global slider value
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
                
            # Check each motif for double top/bottom pattern
            pattern_info = []
            detected_patterns = []
            
            for idx, (idx1, idx2) in enumerate(motifs):
                # Ensure indices are within bounds
                if idx1 >= len(prices) or idx2 >= len(prices) or idx1 + local_window_length > len(prices) or idx2 + local_window_length > len(prices):
                    st.warning(f"Skipping motif pair ({idx1}, {idx2}) due to index out of bounds")
                    continue
                
                # Check first subsequence
                segment1 = prices.iloc[idx1:idx1+local_window_length]
                peaks1, troughs1 = detect_extrema(segment1, order=min(order, local_window_length // 3))
                pattern1, key_points1 = check_double_top_bottom(segment1, peaks1, troughs1, similarity_tol=similarity_tol)
                
                # Check second subsequence
                segment2 = prices.iloc[idx2:idx2+local_window_length]
                peaks2, troughs2 = detect_extrema(segment2, order=min(order, local_window_length // 3))
                pattern2, key_points2 = check_double_top_bottom(segment2, peaks2, troughs2, similarity_tol=similarity_tol)
                
                # Format dates safely
                date1 = safe_date_format(prices.index[idx1]) if idx1 < len(prices.index) else "N/A"
                date2 = safe_date_format(prices.index[idx2]) if idx2 < len(prices.index) else "N/A"
                
                pattern_info.append({
                    "Motif": idx + 1,
                    "Subsequence 1 Index": idx1,
                    "Pattern 1": pattern1 if pattern1 else "None",
                    "Subsequence 1 Date": date1,
                    "Subsequence 2 Index": idx2,
                    "Pattern 2": pattern2 if pattern2 else "None",
                    "Subsequence 2 Date": date2
                })
                
                if pattern1:
                    detected_patterns.append((idx1, pattern1, key_points1))
                if pattern2:
                    detected_patterns.append((idx2, pattern2, key_points2))
                    
            # Display motif statistics
            patterns_df = pd.DataFrame(pattern_info)
            patterns_placeholder.dataframe(patterns_df, use_container_width=True)
            
            # Visualize results
            if len(detected_patterns) > 0:
                # Overall price chart with patterns highlighted
                st.subheader("Price Chart with Detected Patterns")
                price_chart = plotly_visualize_double_patterns(prices, detected_patterns, local_window_length)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Individual pattern visualizations
                st.subheader("Individual Pattern Analysis")
                
                # Create tabs for each detected pattern
                if len(detected_patterns) > 0:
                    tabs = st.tabs([f"Pattern {i+1}: {pattern_type}" for i, (_, pattern_type, _) in enumerate(detected_patterns)])
                    
                    for i, (pattern_idx, pattern_type, key_points) in enumerate(detected_patterns):
                        with tabs[i]:
                            # Format date safely
                            pattern_date = safe_date_format(prices.index[pattern_idx]) if pattern_idx < len(prices.index) else "N/A"
                            st.write(f"Pattern at {pattern_date} (Index: {pattern_idx})")
                            
                            segment = prices.iloc[pattern_idx:pattern_idx+local_window_length]
                            pattern_fig = plotly_visualize_double_pattern(segment, pattern_idx, (pattern_type, key_points))
                            st.plotly_chart(pattern_fig, use_container_width=True)
                            
                            # Add pattern details
                            arr = np.asarray(segment.values, dtype=float)
                            
                            if pattern_type == "double top":
                                peak1_idx, valley_idx, peak2_idx = key_points
                                peak1_val = segment.iloc[peak1_idx]
                                valley_val = segment.iloc[valley_idx]
                                peak2_val = segment.iloc[peak2_idx]
                                
                                # Calculate percentage drop from peaks to valley
                                avg_peak_height = (peak1_val + peak2_val) / 2
                                valley_drop_pct = ((avg_peak_height - valley_val) / avg_peak_height) * 100
                                
                                # Format dates safely
                                peak1_date = safe_date_format(segment.index[peak1_idx]) if len(segment.index) > peak1_idx else "N/A"
                                valley_date = safe_date_format(segment.index[valley_idx]) if len(segment.index) > valley_idx else "N/A"
                                peak2_date = safe_date_format(segment.index[peak2_idx]) if len(segment.index) > peak2_idx else "N/A"
                                
                                st.markdown(f"""
                                **Pattern Details:**
                                - First Peak: {peak1_val:.2f} on {peak1_date}
                                - Valley: {valley_val:.2f} on {valley_date}
                                - Second Peak: {peak2_val:.2f} on {peak2_date}
                                - Peak Height Difference: {abs(peak1_val - peak2_val):.2f} ({abs(peak1_val - peak2_val) / max(peak1_val, peak2_val) * 100:.1f}%)
                                - Valley Depth: {valley_drop_pct:.1f}% from average peak height
                                
                                **Trading Implication:**
                                Double Top patterns often indicate a bearish reversal when the price breaks below the valley level (neckline).
                                """)
                                
                            elif pattern_type == "double bottom":
                                trough1_idx, peak_idx, trough2_idx = key_points
                                trough1_val = segment.iloc[trough1_idx]
                                peak_val = segment.iloc[peak_idx]
                                trough2_val = segment.iloc[trough2_idx]
                                
                                # Calculate percentage rise from troughs to peak
                                avg_trough_level = (trough1_val + trough2_val) / 2
                                peak_rise_pct = ((peak_val - avg_trough_level) / avg_trough_level) * 100
                                
                                # Format dates safely
                                trough1_date = safe_date_format(segment.index[trough1_idx]) if len(segment.index) > trough1_idx else "N/A"
                                peak_date = safe_date_format(segment.index[peak_idx]) if len(segment.index) > peak_idx else "N/A"
                                trough2_date = safe_date_format(segment.index[trough2_idx]) if len(segment.index) > trough2_idx else "N/A"
                                
                                st.markdown(f"""
                                **Pattern Details:**
                                - First Trough: {trough1_val:.2f} on {trough1_date}
                                - Peak: {peak_val:.2f} on {peak_date}
                                - Second Trough: {trough2_val:.2f} on {trough2_date}
                                - Trough Level Difference: {abs(trough1_val - trough2_val):.2f} ({abs(trough1_val - trough2_val) / max(abs(trough1_val), abs(trough2_val)) * 100:.1f}%)
                                - Peak Rise: {peak_rise_pct:.1f}% from average trough level
                                
                                **Trading Implication:**
                                Double Bottom patterns often indicate a bullish reversal when the price breaks above the peak level (neckline).
                                """)
                            
                            # Statistical metrics
                            st.markdown("**Statistical Metrics:**")
                            post_pattern_end = pattern_idx + local_window_length
                            post_pattern_data = None
                            
                            # Get data after the pattern if available
                            if post_pattern_end < len(prices):
                                max_post_points = min(50, len(prices) - post_pattern_end)  # Look ahead up to 50 points or what's available
                                if max_post_points > 0:
                                    post_pattern_data = prices.iloc[post_pattern_end:post_pattern_end + max_post_points]
                                    
                                    # Show follow-through (price action after pattern completion)
                                    if pattern_type == "double top":
                                        neckline_level = valley_val
                                        broke_neckline = any(post_pattern_data < neckline_level)
                                        min_post_price = post_pattern_data.min()
                                        post_move_pct = ((neckline_level - min_post_price) / neckline_level) * 100 if broke_neckline else 0
                                        
                                        st.markdown(f"""
                                        - Follow-through: {'✅ Price broke below neckline' if broke_neckline else '❌ Price did not break below neckline'}
                                        - Maximum decline after pattern: {post_move_pct:.1f}% (if broke neckline)
                                        """)
                                    
                                    elif pattern_type == "double bottom":
                                        neckline_level = peak_val
                                        broke_neckline = any(post_pattern_data > neckline_level)
                                        max_post_price = post_pattern_data.max()
                                        post_move_pct = ((max_post_price - neckline_level) / neckline_level) * 100 if broke_neckline else 0
                                        
                                        st.markdown(f"""
                                        - Follow-through: {'✅ Price broke above neckline' if broke_neckline else '❌ Price did not break above neckline'}
                                        - Maximum rise after pattern: {post_move_pct:.1f}% (if broke neckline)
                                        """)
            else:
                st.warning("No Double Top/Bottom patterns detected with the current parameters.")
                st.markdown("""
                **Try adjusting the parameters:**
                - Increase the window length for larger patterns
                - Try different extrema detection order values
                - Adjust the similarity tolerance threshold
                """)
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            import traceback
            st.text(traceback.format_exc())

# Run the analysis if button is clicked
if run_analysis:
    run_double_pattern_analysis()
else:
    # Welcome message on initial load
    st.info("""
    👈 Customize your analysis parameters in the sidebar and click "Run Analysis" to start.
    
    This app will:
    1. Fetch historical price data for your selected stock
    2. Compute the matrix profile to find recurring patterns
    3. Identify double top and double bottom patterns based on your criteria
    4. Visualize detected patterns with detailed annotations
    
    **What are Double Top/Bottom Patterns?**
    - **Double Top:** A bearish reversal pattern consisting of two peaks at roughly the same level with a valley between them. When the price breaks below the valley level (neckline), it signals potential downward movement.
    - **Double Bottom:** A bullish reversal pattern consisting of two troughs at roughly the same level with a peak between them. When the price breaks above the peak level (neckline), it signals potential upward movement.
    """)