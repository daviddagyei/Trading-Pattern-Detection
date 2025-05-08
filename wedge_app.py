import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import helper functions
from wedges_helpers import (
    fetch_price_data,
    compute_matrix_profile,
    extract_motif_candidates,
    detect_extrema,
    check_wedge_pattern,
    safe_date_format,
    plotly_visualize_wedge_patterns,  # Use Plotly visualizations
    plotly_visualize_wedge_pattern    # Use Plotly visualizations
)

# Set page configuration
st.set_page_config(
    page_title="Wedge Pattern Detector",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📈 Wedge Pattern Detector")
st.markdown("""
This app detects Rising Wedge and Falling Wedge patterns in stock price data using matrix profile algorithms.
Customize your analysis parameters below and visualize the detected patterns.
""")

# Create sidebar for parameter inputs
st.sidebar.header("Parameters")

# Stock selection section
st.sidebar.subheader("Stock Selection")
ticker = st.sidebar.text_input("Stock Symbol", value="AAPL")

# Time frame section
st.sidebar.subheader("Time Frame")

# Add a radio button to select between period-based and date range selection
time_selection = st.sidebar.radio("Select Time Frame Method", ["Predefined Period", "Custom Date Range"])

if time_selection == "Predefined Period":
    # Predefined period selection
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", 
                       "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", 
                       "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=4)  # Default to 1h
    interval_code = interval_options[interval]

    period_options = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", 
                     "3 months": "3mo", "6 months": "6mo", "1 year": "1y", 
                     "2 years": "2y", "5 years": "5y", "10 years": "10y", "max": "max"}
    period = st.sidebar.selectbox("Period", options=list(period_options.keys()), index=4)  # Default to 6mo
    period_code = period_options[period]
    
    # Set date range variables to None
    start_date = None
    end_date = None
    
else:  # Custom Date Range
    # Interval selection remains the same
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", 
                       "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", 
                       "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=4)
    interval_code = interval_options[interval]
    
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
window_size = st.sidebar.slider("Window Size", min_value=10, max_value=200, value=50, step=5, 
                               help="Number of data points in each subsequence for matrix profile")
top_k = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value=5, 
                         help="Number of motif pairs to extract from the matrix profile")
extrema_order = st.sidebar.slider("Extrema Detection Order", min_value=1, max_value=10, value=3, 
                           help="Points to compare on each side for detecting local extrema")

# Pattern threshold parameters
st.sidebar.subheader("Pattern Thresholds")
slope_tol = st.sidebar.slider("Slope Tolerance (%)", min_value=1, max_value=50, value=10, 
                             help="Tolerance for slope difference to consider trendlines converging")
slope_tol = slope_tol / 100.0  # convert percentage to fraction

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Credits
st.sidebar.markdown("---")
st.sidebar.caption("Created with ❤️ using Streamlit")

# Main content - simpler structure just like double_top_bottom_app.py
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
def run_wedge_pattern_analysis():
    # Initialize local window_size from the global slider value
    local_window_size = window_size
    
    with st.spinner('Fetching price data...'):
        # Get price data
        try:
            prices = fetch_price_data(symbol=ticker, interval=interval_code, period=period_code, 
                                     start_date=start_date, end_date=end_date)
            
            # Check if we got data
            if prices is None or len(prices) == 0:
                if time_selection == "Predefined Period":
                    st.error(f"No data returned for symbol {ticker} with interval {interval} and period {period}.")
                else:
                    st.error(f"No data returned for symbol {ticker} with interval {interval} between {start_date} and {end_date}.")
                return
                
            st.session_state.prices = prices
            
            # Create a basic price chart (simple line chart)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(prices.index, prices.values, label='Price', color='blue')
            ax.set_title(f"{ticker} Price Chart")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            price_chart_placeholder.pyplot(fig)
            
            # Check if we have enough data for the chosen window size
            if len(prices) <= local_window_size:
                st.error(f"Not enough data points ({len(prices)}) for the selected window size ({local_window_size}). Either choose a smaller window size or fetch more data with a longer period.")
                
                # Get first and last date safely
                first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
                
                # Get current price as float (scalar value) instead of Series
                current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
                
                stats_placeholder.info(f"""
                **Data Summary:**
                - Symbol: {ticker}
                - Data points: {len(prices)}
                - Date range: {first_date} to {last_date}
                - Current price: {current_price:.2f}
                
                **Recommendation:** Reduce window size to {max(5, len(prices) // 2)} or less.
                """)
                return
            
            # Get first and last date safely
            first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
            last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
            
            # Get current price as float
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0.0
            
            stats_placeholder.info(f"""
            **Data Summary:**
            - Symbol: {ticker}
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
            if prices.isna().any().bool():
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
            
            # Validate window size is appropriate
            if len(price_array) <= 1:
                st.error(f"Not enough valid data points (only {len(price_array)}). Please select a different stock or time period.")
                return
            
            if local_window_size >= len(price_array):
                # Suggest a reasonable window size
                suggested_window = max(5, len(price_array) // 4)
                st.error(f"Window size ({local_window_size}) must be less than the number of data points ({len(price_array)}). Try using {suggested_window} instead.")
                # Auto-adjust the window size if it's too large
                local_window_size = suggested_window
                st.info(f"Automatically adjusted window size to {local_window_size}")
            
            # Ensure window size is at least 4 (minimum required for meaningful patterns)
            if local_window_size < 4:
                local_window_size = 4
                st.info(f"Window size adjusted to minimum value of 4")
            
            # Compute matrix profile with the validated window size
            st.info(f"Computing matrix profile with window size = {local_window_size}")
            profile_distances, profile_indices = compute_matrix_profile(price_array, local_window_size)
            
            # Extract motif candidates
            motifs = extract_motif_candidates(profile_distances, profile_indices, local_window_size, k=top_k)
            
            # Check if no motifs were found
            if not motifs:
                st.warning("No motifs found. Try adjusting parameters:")
                st.markdown("""
                **Suggestions:**
                - Try a different window size
                - Consider a different time period or interval
                - Check if the price series has enough variation
                """)
                return
                
            # Analyze each motif pair for wedge patterns
            patterns = []  # list to store details of each detected wedge occurrence
            used_indices = set()  # to avoid duplicate entries
            
            # Loop through each motif pair (each pair yields two subsequences)
            for (idx1, idx2, dist) in motifs:
                for idx in (idx1, idx2):
                    if idx in used_indices:
                        continue  # skip if already processed this subsequence
                        
                    # Ensure indices are within bounds
                    if idx + local_window_size > len(prices):
                        st.warning(f"Skipping motif at index {idx} due to index out of bounds")
                        continue
                        
                    subseq = price_array[idx: idx + local_window_size]
                    
                    # Detect extrema in this subsequence
                    peaks_idx, troughs_idx = detect_extrema(subseq, order=min(extrema_order, local_window_size // 3))
                    
                    # Classify pattern type via trendline slope analysis
                    pat_type, slope_peak, slope_trough = check_wedge_pattern(subseq, peaks_idx, troughs_idx, slope_tolerance=slope_tol)
                    
                    if pat_type is None:
                        # Not a wedge pattern
                        continue
                        
                    # Determine pattern timeframe and breakout analysis
                    start_idx = idx
                    end_idx = idx + local_window_size - 1
                    start_time = prices.index[start_idx]
                    end_time = prices.index[end_idx]
                    duration = end_idx - start_idx + 1  # number of data points in pattern
                    
                    # Determine breakout direction by looking at price action after the pattern
                    breakout_dir = None
                    breakout_idx = None
                    
                    # Look ahead a horizon (we use the pattern length as the horizon for analysis or until data end)
                    horizon = local_window_size
                    post_start = end_idx + 1
                    post_end = min(len(prices) - 1, end_idx + horizon)
                    
                    if post_start <= post_end:
                        future_prices = price_array[post_start: post_end + 1]
                        
                        # Calculate levels for breakout reference
                        if pat_type == "Rising Wedge":
                            # Rising wedge breakout is typically downward through the lower trendline
                            # Project the trough trendline to the end of the pattern
                            x_troughs = troughs_idx
                            y_troughs = subseq[troughs_idx]
                            slope, intercept = np.polyfit(x_troughs, y_troughs, 1)
                            breakout_level = slope * local_window_size + intercept
                            
                            # Check if price breaks below this level
                            below_level = np.where(future_prices < breakout_level)[0]
                            if below_level.size > 0:
                                breakout_dir = 'Down'
                                breakout_idx = post_start + below_level[0]
                                
                        elif pat_type == "Falling Wedge":
                            # Falling wedge breakout is typically upward through the upper trendline
                            # Project the peak trendline to the end of the pattern
                            x_peaks = peaks_idx
                            y_peaks = subseq[peaks_idx]
                            slope, intercept = np.polyfit(x_peaks, y_peaks, 1)
                            breakout_level = slope * local_window_size + intercept
                            
                            # Check if price breaks above this level
                            above_level = np.where(future_prices > breakout_level)[0]
                            if above_level.size > 0:
                                breakout_dir = 'Up'
                                breakout_idx = post_start + above_level[0]
                    
                    # Store pattern details
                    pattern_info = {
                        'type': pat_type,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'slope_peaks': float(slope_peak) if not isinstance(slope_peak, (list, np.ndarray)) else float(slope_peak[0]),
                        'slope_troughs': float(slope_trough) if not isinstance(slope_trough, (list, np.ndarray)) else float(slope_trough[0]),
                        'breakout_dir': breakout_dir,
                        'breakout_idx': breakout_idx,
                        'peaks_idx': peaks_idx,       # relative indices within subseq
                        'troughs_idx': troughs_idx    # relative indices within subseq
                    }
                    
                    patterns.append(pattern_info)
                    used_indices.add(idx)
                    
            # If no patterns detected, inform the user
            if not patterns:
                st.warning("No wedge patterns found for the given parameters.")
                st.markdown("""
                **Try adjusting the parameters:**
                - Try a different window size
                - Adjust the extrema detection order
                - Modify the slope tolerance
                """)
            else:
                # Sort patterns by start time
                patterns.sort(key=lambda x: x['start_idx'])
                
                # Create a Plotly figure for the overview chart showing all patterns
                overview_fig = plotly_visualize_wedge_patterns(prices, patterns, local_window_size)
                patterns_placeholder.plotly_chart(overview_fig, use_container_width=True)
                
                # Display the summary table
                st.subheader("Detected Wedge Pattern Details")
                
                # Create DataFrame for display
                summary_data = []
                for pat in patterns:
                    summary_data.append({
                        "Pattern Type": pat['type'],
                        "Start": safe_date_format(pat['start_time']),
                        "End": safe_date_format(pat['end_time']),
                        "Duration (pts)": pat['duration'],
                        "Slope (Peaks)": f"{pat['slope_peaks']:.4f}",
                        "Slope (Troughs)": f"{pat['slope_troughs']:.4f}",
                        "Breakout": (pat['breakout_dir'] or "N/A")
                    })
                
                pattern_df = pd.DataFrame(summary_data)
                st.dataframe(pattern_df, use_container_width=True)
                
                # Individual pattern visualizations using tabs
                st.subheader("Individual Pattern Analysis")
                
                # Create tabs for each detected pattern
                if len(patterns) > 0:
                    tabs = st.tabs([f"Pattern {i+1}: {p['type']}" for i, p in enumerate(patterns)])
                    
                    for i, pattern in enumerate(patterns):
                        with tabs[i]:
                            # Format date safely
                            pattern_date = safe_date_format(pattern['start_time'])
                            st.write(f"{pattern['type']} at {pattern_date} (Index: {pattern['start_idx']})")
                            
                            # Use the detailed visualization function for the selected pattern
                            pattern_detail_fig = plotly_visualize_wedge_pattern(prices, pattern, local_window_size)
                            st.plotly_chart(pattern_detail_fig, use_container_width=True)
                            
                            # Display pattern details
                            st.markdown(f"""
                            **Pattern Details:**
                            - **Pattern Type:** {pattern['type']}
                            - **Start Date:** {safe_date_format(pattern['start_time'])}
                            - **End Date:** {safe_date_format(pattern['end_time'])}
                            - **Duration:** {pattern['duration']} data points
                            - **Peak Trendline Slope:** {pattern['slope_peaks']:.4f}
                            - **Trough Trendline Slope:** {pattern['slope_troughs']:.4f}
                            - **Breakout Direction:** {pattern['breakout_dir'] or "Not detected"}
                            """)
                            
                            # Add pattern-specific interpretation
                            st.markdown("**Pattern Interpretation:**")
                            if pattern['type'] == "Rising Wedge":
                                st.markdown("""
                                **Rising Wedge (Bearish Pattern):**
                                - Characterized by converging trendlines both sloping upward
                                - The lower trendline has a steeper slope than the upper trendline
                                - Usually signals a potential reversal from an uptrend to a downtrend
                                - Confirmation occurs when price breaks below the lower trendline
                                """)
                            else:  # Falling Wedge
                                st.markdown("""
                                **Falling Wedge (Bullish Pattern):**
                                - Characterized by converging trendlines both sloping downward
                                - The upper trendline has a steeper slope than the lower trendline
                                - Usually signals a potential reversal from a downtrend to an uptrend
                                - Confirmation occurs when price breaks above the upper trendline
                                """)
                else:
                    st.info("No patterns detected to view in detail.")
                
                # Pattern interpretation section for overview
                st.subheader("Pattern Types Found")
                
                # Display interpretation for the detected pattern types
                rising_wedge_detected = any(p['type'] == "Rising Wedge" for p in patterns)
                falling_wedge_detected = any(p['type'] == "Falling Wedge" for p in patterns)
                
                if rising_wedge_detected:
                    st.markdown("""
                    **Rising Wedge (Bearish Pattern):**
                    - Characterized by converging trendlines both sloping upward
                    - The lower trendline has a steeper slope than the upper trendline
                    - Usually signals a potential reversal from an uptrend to a downtrend
                    - Confirmation occurs when price breaks below the lower trendline
                    """)
                
                if falling_wedge_detected:
                    st.markdown("""
                    **Falling Wedge (Bullish Pattern):**
                    - Characterized by converging trendlines both sloping downward
                    - The upper trendline has a steeper slope than the lower trendline
                    - Usually signals a potential reversal from a downtrend to an uptrend
                    - Confirmation occurs when price breaks above the upper trendline
                    """)
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            import traceback
            st.text(traceback.format_exc())

# Run the analysis if button is clicked
if run_analysis:
    run_wedge_pattern_analysis()
else:
    # Welcome message on initial load
    st.info("""
    👈 Customize your analysis parameters in the sidebar and click "Run Analysis" to start.
    
    This app will:
    1. Fetch historical price data for your selected stock
    2. Compute the matrix profile to find recurring patterns
    3. Identify rising and falling wedge patterns based on your criteria
    4. Visualize detected patterns with detailed annotations
    
    **What are Wedge Patterns?**
    - **Rising Wedge:** A bearish reversal pattern with converging trendlines both sloping upward. The lower trendline rises more steeply than the upper trendline. A downward breakout signals potential bearish movement.
    - **Falling Wedge:** A bullish reversal pattern with converging trendlines both sloping downward. The upper trendline falls more steeply than the lower trendline. An upward breakout signals potential bullish movement.
    """)
