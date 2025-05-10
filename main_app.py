import streamlit as st

# --- Unified Trading Pattern Detection App ---
# This app merges Head & Shoulders, Double Top/Bottom, and Wedge pattern detectors.

# Sidebar: App selection
st.sidebar.title("Select Pattern Detector")
app_mode = st.sidebar.radio(
    "Choose a pattern detection tool:",
    ("Head and Shoulders", "Double Top/Bottom", "Wedge")
)

# Placeholder for each app's logic (to be filled in next steps)
def run_head_shoulders_app():
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
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
    st.title("📊 Head and Shoulders Pattern Detector")
    st.markdown("""
    This app detects Head and Shoulders patterns in stock price data using matrix profile algorithms.
    Customize your analysis parameters below and visualize the detected patterns.
    """)
    st.sidebar.header("Parameters")
    st.sidebar.subheader("Stock Selection")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", key="hs_symbol")
    st.sidebar.subheader("Time Frame")
    time_selection = st.sidebar.radio("Select Time Frame Method", ["Predefined Period", "Custom Date Range"], key="hs_time_selection")
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    if time_selection == "Predefined Period":
        interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=5, key="hs_interval")
        interval_code = interval_options[interval]
        period_options = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y", "max": "max"}
        period = st.sidebar.selectbox("Period", options=list(period_options.keys()), index=4, key="hs_period")
        period_code = period_options[period]
        start_date = None
        end_date = None
    else:
        interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=5, key="hs_interval")
        interval_code = interval_options[interval]
        from datetime import datetime, timedelta
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=180)
        start_date = st.sidebar.date_input("Start Date", value=default_start_date, key="hs_start_date")
        end_date = st.sidebar.date_input("End Date", value=default_end_date, key="hs_end_date")
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        period_code = None
        if start_date >= end_date:
            st.sidebar.error("End date must be after start date")
    st.sidebar.subheader("Pattern Detection Parameters")
    window_length = st.sidebar.slider("Window Length (m)", min_value=10, max_value=100, value=30, key="hs_window_length")
    top_k = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value=3, key="hs_top_k")
    order = st.sidebar.slider("Extrema Detection Order", min_value=1, max_value=10, value=3, key="hs_order")
    st.sidebar.subheader("Pattern Thresholds")
    shoulder_tol = st.sidebar.slider("Shoulder Height Tolerance", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="hs_shoulder_tol")
    trough_tol = st.sidebar.slider("Trough Alignment Tolerance", min_value=0.01, max_value=0.5, value=0.2, step=0.01, key="hs_trough_tol")
    run_analysis = st.sidebar.button("Run Analysis", key="hs_run_analysis", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption("Created with ❤️ using Streamlit")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Price Chart")
        price_chart_placeholder = st.empty()
    with col2:
        st.subheader("Pattern Detection Stats")
        stats_placeholder = st.empty()
    st.markdown("---")
    patterns_section = st.container()
    with patterns_section:
        st.subheader("Detected Patterns")
        patterns_placeholder = st.empty()
    def run_head_shoulders_analysis():
        local_window_length = window_length
        with st.spinner('Fetching price data...'):
            try:
                prices = fetch_price_data(symbol=symbol, interval=interval_code, period=period_code, start_date=start_date, end_date=end_date)
                if prices is None or len(prices) == 0:
                    if time_selection == "Predefined Period":
                        st.error(f"No data returned for symbol {symbol} with interval {interval} and period {period}.")
                    else:
                        st.error(f"No data returned for symbol {symbol} with interval {interval} between {start_date} and {end_date}.")
                    return
                st.session_state.prices = prices
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name='Price'))
                fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price", height=500)
                price_chart_placeholder.plotly_chart(fig, use_container_width=True)
                if len(prices) <= local_window_length:
                    st.error(f"Not enough data points ({len(prices)}) for the selected window length ({local_window_length}). Either choose a smaller window length or fetch more data with a longer period.")
                    first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                    last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
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
                first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
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
                if prices.isna().any():
                    st.warning(f"Found {prices.isna().sum()} NaN values in the data. These will be removed.")
                    prices = prices.dropna()
                    if len(prices) == 0:
                        st.error("After removing NaN values, no data remains. Please try a different stock or time period.")
                        return
                price_array = np.asarray(prices.values, dtype=float)
                st.info(f"Price array shape: {price_array.shape}, Min: {price_array.min():.2f}, Max: {price_array.max():.2f}")
                value_range = price_array.max() - price_array.min()
                if value_range < 1e-5:
                    st.error("The price data contains all identical values (or nearly identical). Matrix profile cannot be computed on constant data.")
                    return
                if len(price_array) <= 1:
                    st.error(f"Not enough valid data points (only {len(price_array)}). Please select a different stock or time period.")
                    return
                if local_window_length >= len(price_array):
                    suggested_window = max(5, len(price_array) // 4)
                    st.error(f"Window length ({local_window_length}) must be less than the number of data points ({len(price_array)}). Try using {suggested_window} instead.")
                    local_window_length = suggested_window
                    st.info(f"Automatically adjusted window length to {local_window_length}")
                if local_window_length < 4:
                    local_window_length = 4
                    st.info(f"Window length adjusted to minimum value of 4")
                st.info(f"Computing matrix profile with window length = {local_window_length}")
                mp = compute_matrix_profile(price_array, local_window_length)
                motifs = extract_motif_candidates(mp, local_window_length, top_k=top_k)
                if not motifs:
                    st.warning("No motifs found. Try adjusting parameters:")
                    st.markdown("""
                    **Suggestions:**
                    - Try a different window length
                    - Consider a different time period or interval
                    - Check if the price series has enough variation
                    """)
                    return
                pattern_indices = []
                pattern_info = []
                for idx, (idx1, idx2) in enumerate(motifs):
                    if idx1 >= len(prices) or idx2 >= len(prices) or idx1 + local_window_length > len(prices) or idx2 + local_window_length > len(prices):
                        st.warning(f"Skipping motif pair ({idx1}, {idx2}) due to index out of bounds")
                        continue
                    segment1 = prices.iloc[idx1:idx1+local_window_length]
                    peaks1, troughs1 = detect_extrema(segment1, order=min(order, local_window_length // 3))
                    is_pattern1 = check_head_and_shoulders(segment1, peaks1, troughs1, shoulder_tol=shoulder_tol, trough_tol=trough_tol)
                    segment2 = prices.iloc[idx2:idx2+local_window_length]
                    peaks2, troughs2 = detect_extrema(segment2, order=min(order, local_window_length // 3))
                    is_pattern2 = check_head_and_shoulders(segment2, peaks2, troughs2, shoulder_tol=shoulder_tol, trough_tol=trough_tol)
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
                st.dataframe(pd.DataFrame(pattern_info), use_container_width=True)
                if len(pattern_indices) > 0:
                    price_chart = plotly_visualize_price_with_patterns(prices, pattern_indices, local_window_length)
                    st.plotly_chart(price_chart, use_container_width=True)
                    for idx in pattern_indices:
                        pattern_date = safe_date_format(prices.index[idx]) if idx < len(prices.index) else "N/A"
                        st.subheader(f"Pattern at {pattern_date} (Index: {idx})")
                        segment = prices.iloc[idx:idx+local_window_length]
                        peaks, troughs = detect_extrema(segment, order=min(order, local_window_length // 3))
                        is_pattern = check_head_and_shoulders(segment, peaks, troughs, shoulder_tol=shoulder_tol, trough_tol=trough_tol)
                        pattern_fig = plotly_visualize_pattern(segment, idx, peaks, troughs, is_pattern)
                        st.plotly_chart(pattern_fig, use_container_width=True)
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
    if run_analysis:
        run_head_shoulders_analysis()
    else:
        st.info("""
        👈 Customize your analysis parameters in the sidebar and click \"Run Analysis\" to start.
        This app will:
        1. Fetch historical price data for your selected stock
        2. Compute the matrix profile to find recurring patterns
        3. Identify head and shoulders patterns based on your criteria
        4. Visualize detected patterns with detailed annotations
        """)

def run_double_top_bottom_app():
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
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
    st.title("📊 Double Top/Bottom Pattern Detector")
    st.markdown("""
    This app detects Double Top and Double Bottom patterns in stock price data using matrix profile algorithms.
    Customize your analysis parameters below and visualize the detected patterns.
    """)
    st.sidebar.header("Parameters")
    st.sidebar.subheader("Stock Selection")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", key="dtb_symbol")
    st.sidebar.subheader("Time Frame")
    time_selection = st.sidebar.radio("Select Time Frame Method", ["Predefined Period", "Custom Date Range"], key="dtb_time_selection")
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    if time_selection == "Predefined Period":
        interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=5, key="dtb_interval")
        interval_code = interval_options[interval]
        period_options = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y", "max": "max"}
        period = st.sidebar.selectbox("Period", options=list(period_options.keys()), index=4, key="dtb_period")
        period_code = period_options[period]
        start_date = None
        end_date = None
    else:
        interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=5, key="dtb_interval")
        interval_code = interval_options[interval]
        from datetime import datetime, timedelta
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=180)
        start_date = st.sidebar.date_input("Start Date", value=default_start_date, key="dtb_start_date")
        end_date = st.sidebar.date_input("End Date", value=default_end_date, key="dtb_end_date")
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        period_code = None
        if start_date >= end_date:
            st.sidebar.error("End date must be after start date")
    st.sidebar.subheader("Pattern Detection Parameters")
    window_length = st.sidebar.slider("Window Length (m)", min_value=10, max_value=100, value=30, key="dtb_window_length")
    top_k = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value=3, key="dtb_top_k")
    order = st.sidebar.slider("Extrema Detection Order", min_value=1, max_value=10, value=3, key="dtb_order")
    st.sidebar.subheader("Pattern Thresholds")
    similarity_tol = st.sidebar.slider("Peak/Trough Similarity Tolerance", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="dtb_similarity_tol")
    run_analysis = st.sidebar.button("Run Analysis", key="dtb_run_analysis", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption("Created with ❤️ using Streamlit")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Price Chart")
        price_chart_placeholder = st.empty()
    with col2:
        st.subheader("Pattern Detection Stats")
        stats_placeholder = st.empty()
    st.markdown("---")
    patterns_section = st.container()
    with patterns_section:
        st.subheader("Detected Patterns")
        patterns_placeholder = st.empty()
    def run_double_pattern_analysis():
        local_window_length = window_length
        with st.spinner('Fetching price data...'):
            try:
                prices = fetch_price_data(symbol=symbol, interval=interval_code, period=period_code, start_date=start_date, end_date=end_date)
                if prices is None or len(prices) == 0:
                    if time_selection == "Predefined Period":
                        st.error(f"No data returned for symbol {symbol} with interval {interval} and period {period}.")
                    else:
                        st.error(f"No data returned for symbol {symbol} with interval {interval} between {start_date} and {end_date}.")
                    return
                st.session_state.prices = prices
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name='Price'))
                fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price", height=500)
                price_chart_placeholder.plotly_chart(fig, use_container_width=True)
                if len(prices) <= local_window_length:
                    st.error(f"Not enough data points ({len(prices)}) for the selected window length ({local_window_length}). Either choose a smaller window length or fetch more data with a longer period.")
                    first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                    last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
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
                first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
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
                if prices.isna().any():
                    st.warning(f"Found {prices.isna().sum()} NaN values in the data. These will be removed.")
                    prices = prices.dropna()
                    if len(prices) == 0:
                        st.error("After removing NaN values, no data remains. Please try a different stock or time period.")
                        return
                price_array = np.asarray(prices.values, dtype=float)
                st.info(f"Price array shape: {price_array.shape}, Min: {price_array.min():.2f}, Max: {price_array.max():.2f}")
                value_range = price_array.max() - price_array.min()
                if value_range < 1e-5:
                    st.error("The price data contains all identical values (or nearly identical). Matrix profile cannot be computed on constant data.")
                    return
                if len(price_array) <= 1:
                    st.error(f"Not enough valid data points (only {len(price_array)}). Please select a different stock or time period.")
                    return
                if local_window_length >= len(price_array):
                    suggested_window = max(5, len(price_array) // 4)
                    st.error(f"Window length ({local_window_length}) must be less than the number of data points ({len(price_array)}). Try using {suggested_window} instead.")
                    local_window_length = suggested_window
                    st.info(f"Automatically adjusted window length to {local_window_length}")
                if local_window_length < 4:
                    local_window_length = 4
                    st.info(f"Window length adjusted to minimum value of 4")
                st.info(f"Computing matrix profile with window length = {local_window_length}")
                mp = compute_matrix_profile(price_array, local_window_length)
                motifs = extract_motif_candidates(mp, local_window_length, top_k=top_k)
                if not motifs:
                    st.warning("No motifs found. Try adjusting parameters:")
                    st.markdown("""
                    **Suggestions:**
                    - Try a different window length
                    - Consider a different time period or interval
                    - Check if the price series has enough variation
                    """)
                    return
                pattern_info = []
                detected_patterns = []
                for idx, (idx1, idx2) in enumerate(motifs):
                    if idx1 >= len(prices) or idx2 >= len(prices) or idx1 + local_window_length > len(prices) or idx2 + local_window_length > len(prices):
                        st.warning(f"Skipping motif pair ({idx1}, {idx2}) due to index out of bounds")
                        continue
                    segment1 = prices.iloc[idx1:idx1+local_window_length]
                    peaks1, troughs1 = detect_extrema(segment1, order=min(order, local_window_length // 3))
                    pattern1, key_points1 = check_double_top_bottom(segment1, peaks1, troughs1, similarity_tol=similarity_tol)
                    segment2 = prices.iloc[idx2:idx2+local_window_length]
                    peaks2, troughs2 = detect_extrema(segment2, order=min(order, local_window_length // 3))
                    pattern2, key_points2 = check_double_top_bottom(segment2, peaks2, troughs2, similarity_tol=similarity_tol)
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
                patterns_df = pd.DataFrame(pattern_info)
                patterns_placeholder.dataframe(patterns_df, use_container_width=True)
                if len(detected_patterns) > 0:
                    st.subheader("Price Chart with Detected Patterns")
                    price_chart = plotly_visualize_double_patterns(prices, detected_patterns, local_window_length)
                    st.plotly_chart(price_chart, use_container_width=True)
                    st.subheader("Individual Pattern Analysis")
                    if len(detected_patterns) > 0:
                        tabs = st.tabs([f"Pattern {i+1}: {pattern_type}" for i, (_, pattern_type, _) in enumerate(detected_patterns)])
                        for i, (pattern_idx, pattern_type, key_points) in enumerate(detected_patterns):
                            with tabs[i]:
                                pattern_date = safe_date_format(prices.index[pattern_idx]) if pattern_idx < len(prices.index) else "N/A"
                                st.write(f"Pattern at {pattern_date} (Index: {pattern_idx})")
                                segment = prices.iloc[pattern_idx:pattern_idx+local_window_length]
                                pattern_fig = plotly_visualize_double_pattern(segment, pattern_idx, (pattern_type, key_points))
                                st.plotly_chart(pattern_fig, use_container_width=True)
                                arr = np.asarray(segment.values, dtype=float)
                                if pattern_type == "double top":
                                    peak1_idx, valley_idx, peak2_idx = key_points
                                    peak1_val = segment.iloc[peak1_idx]
                                    valley_val = segment.iloc[valley_idx]
                                    peak2_val = segment.iloc[peak2_idx]
                                    avg_peak_height = (peak1_val + peak2_val) / 2
                                    valley_drop_pct = ((avg_peak_height - valley_val) / avg_peak_height) * 100
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
                                    avg_trough_level = (trough1_val + trough2_val) / 2
                                    peak_rise_pct = ((peak_val - avg_trough_level) / avg_trough_level) * 100
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
                                st.markdown("**Statistical Metrics:**")
                                post_pattern_end = pattern_idx + local_window_length
                                post_pattern_data = None
                                if post_pattern_end < len(prices):
                                    max_post_points = min(50, len(prices) - post_pattern_end)
                                    if max_post_points > 0:
                                        post_pattern_data = prices.iloc[post_pattern_end:post_pattern_end + max_post_points]
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
    if run_analysis:
        run_double_pattern_analysis()
    else:
        st.info("""
        👈 Customize your analysis parameters in the sidebar and click \"Run Analysis\" to start.
        This app will:
        1. Fetch historical price data for your selected stock
        2. Compute the matrix profile to find recurring patterns
        3. Identify double top and double bottom patterns based on your criteria
        4. Visualize detected patterns with detailed annotations
        **What are Double Top/Bottom Patterns?**
        - **Double Top:** A bearish reversal pattern consisting of two peaks at roughly the same level with a valley between them. When the price breaks below the valley level (neckline), it signals potential downward movement.
        - **Double Bottom:** A bullish reversal pattern consisting of two troughs at roughly the same level with a peak between them. When the price breaks above the peak level (neckline), it signals potential upward movement.
        """)

def run_wedge_app():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from wedges_helpers import (
        fetch_price_data,
        compute_matrix_profile,
        extract_motif_candidates,
        detect_extrema,
        check_wedge_pattern,
        safe_date_format,
        plotly_visualize_wedge_patterns,
        plotly_visualize_wedge_pattern
    )
    st.title("📈 Wedge Pattern Detector")
    st.markdown("""
    This app detects Rising Wedge and Falling Wedge patterns in stock price data using matrix profile algorithms.
    Customize your analysis parameters below and visualize the detected patterns.
    """)
    st.sidebar.header("Parameters")
    st.sidebar.subheader("Stock Selection")
    ticker = st.sidebar.text_input("Stock Symbol", value="AAPL", key="wedge_symbol")
    st.sidebar.subheader("Time Frame")
    time_selection = st.sidebar.radio("Select Time Frame Method", ["Predefined Period", "Custom Date Range"], key="wedge_time_selection")
    interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
    if time_selection == "Predefined Period":
        interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=4, key="wedge_interval")
        interval_code = interval_options[interval]
        period_options = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y", "max": "max"}
        period = st.sidebar.selectbox("Period", options=list(period_options.keys()), index=4, key="wedge_period")
        period_code = period_options[period]
        start_date = None
        end_date = None
    else:
        interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=4, key="wedge_interval")
        interval_code = interval_options[interval]
        from datetime import datetime, timedelta
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=180)
        start_date = st.sidebar.date_input("Start Date", value=default_start_date, key="wedge_start_date")
        end_date = st.sidebar.date_input("End Date", value=default_end_date, key="wedge_end_date")
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        period_code = None
        if start_date >= end_date:
            st.sidebar.error("End date must be after start date")
    st.sidebar.subheader("Pattern Detection Parameters")
    window_size = st.sidebar.slider("Window Size", min_value=10, max_value=200, value=50, step=5, key="wedge_window_size")
    top_k = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value=5, key="wedge_top_k")
    extrema_order = st.sidebar.slider("Extrema Detection Order", min_value=1, max_value=10, value=3, key="wedge_order")
    st.sidebar.subheader("Pattern Thresholds")
    slope_tol = st.sidebar.slider("Slope Tolerance (%)", min_value=1, max_value=50, value=10, key="wedge_slope_tol")
    slope_tol = slope_tol / 100.0
    run_analysis = st.sidebar.button("Run Analysis", key="wedge_run_analysis", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption("Created with ❤️ using Streamlit")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Price Chart")
        price_chart_placeholder = st.empty()
    with col2:
        st.subheader("Pattern Detection Stats")
        stats_placeholder = st.empty()
    st.markdown("---")
    patterns_section = st.container()
    with patterns_section:
        st.subheader("Detected Patterns")
        patterns_placeholder = st.empty()
    def run_wedge_pattern_analysis():
        local_window_size = window_size
        with st.spinner('Fetching price data...'):
            try:
                prices = fetch_price_data(symbol=ticker, interval=interval_code, period=period_code, start_date=start_date, end_date=end_date)
                if prices is None or len(prices) == 0:
                    if time_selection == "Predefined Period":
                        st.error(f"No data returned for symbol {ticker} with interval {interval} and period {period}.")
                    else:
                        st.error(f"No data returned for symbol {ticker} with interval {interval} between {start_date} and {end_date}.")
                    return
                st.session_state.prices = prices
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prices.index, prices.values, label='Price', color='blue')
                ax.set_title(f"{ticker} Price Chart")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                price_chart_placeholder.pyplot(fig)
                if len(prices) <= local_window_size:
                    st.error(f"Not enough data points ({len(prices)}) for the selected window size ({local_window_size}). Either choose a smaller window size or fetch more data with a longer period.")
                    first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                    last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
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
                first_date = safe_date_format(prices.index[0]) if len(prices) > 0 else "N/A"
                last_date = safe_date_format(prices.index[-1]) if len(prices) > 0 else "N/A"
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
                if prices.isna().any().bool():
                    st.warning(f"Found {prices.isna().sum()} NaN values in the data. These will be removed.")
                    prices = prices.dropna()
                    if len(prices) == 0:
                        st.error("After removing NaN values, no data remains. Please try a different stock or time period.")
                        return
                price_array = np.asarray(prices.values, dtype=float)
                st.info(f"Price array shape: {price_array.shape}, Min: {price_array.min():.2f}, Max: {price_array.max():.2f}")
                value_range = price_array.max() - price_array.min()
                if value_range < 1e-5:
                    st.error("The price data contains all identical values (or nearly identical). Matrix profile cannot be computed on constant data.")
                    return
                if len(price_array) <= 1:
                    st.error(f"Not enough valid data points (only {len(price_array)}). Please select a different stock or time period.")
                    return
                if local_window_size >= len(price_array):
                    suggested_window = max(5, len(price_array) // 4)
                    st.error(f"Window size ({local_window_size}) must be less than the number of data points ({len(price_array)}). Try using {suggested_window} instead.")
                    local_window_size = suggested_window
                    st.info(f"Automatically adjusted window size to {local_window_size}")
                if local_window_size < 4:
                    local_window_size = 4
                    st.info(f"Window size adjusted to minimum value of 4")
                st.info(f"Computing matrix profile with window size = {local_window_size}")
                profile_distances, profile_indices = compute_matrix_profile(price_array, local_window_size)
                motifs = extract_motif_candidates(profile_distances, profile_indices, local_window_size, k=top_k)
                if not motifs:
                    st.warning("No motifs found. Try adjusting parameters:")
                    st.markdown("""
                    **Suggestions:**
                    - Try a different window size
                    - Consider a different time period or interval
                    - Check if the price series has enough variation
                    """)
                    return
                patterns = []
                used_indices = set()
                for (idx1, idx2, dist) in motifs:
                    for idx in (idx1, idx2):
                        if idx in used_indices:
                            continue
                        if idx + local_window_size > len(prices):
                            st.warning(f"Skipping motif at index {idx} due to index out of bounds")
                            continue
                        subseq = price_array[idx: idx + local_window_size]
                        peaks_idx, troughs_idx = detect_extrema(subseq, order=min(extrema_order, local_window_size // 3))
                        pat_type, slope_peak, slope_trough = check_wedge_pattern(subseq, peaks_idx, troughs_idx, slope_tolerance=slope_tol)
                        if pat_type is None:
                            continue
                        start_idx = idx
                        end_idx = idx + local_window_size - 1
                        start_time = prices.index[start_idx]
                        end_time = prices.index[end_idx]
                        duration = end_idx - start_idx + 1
                        breakout_dir = None
                        breakout_idx = None
                        horizon = local_window_size
                        post_start = end_idx + 1
                        post_end = min(len(prices) - 1, end_idx + horizon)
                        if post_start <= post_end:
                            future_prices = price_array[post_start: post_end + 1]
                            if pat_type == "Rising Wedge":
                                x_troughs = troughs_idx
                                y_troughs = subseq[troughs_idx]
                                slope, intercept = np.polyfit(x_troughs, y_troughs, 1)
                                breakout_level = slope * local_window_size + intercept
                                below_level = np.where(future_prices < breakout_level)[0]
                                if below_level.size > 0:
                                    breakout_dir = 'Down'
                                    breakout_idx = post_start + below_level[0]
                            elif pat_type == "Falling Wedge":
                                x_peaks = peaks_idx
                                y_peaks = subseq[peaks_idx]
                                slope, intercept = np.polyfit(x_peaks, y_peaks, 1)
                                breakout_level = slope * local_window_size + intercept
                                above_level = np.where(future_prices > breakout_level)[0]
                                if above_level.size > 0:
                                    breakout_dir = 'Up'
                                    breakout_idx = post_start + above_level[0]
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
                            'peaks_idx': peaks_idx,
                            'troughs_idx': troughs_idx
                        }
                        patterns.append(pattern_info)
                        used_indices.add(idx)
                if not patterns:
                    st.warning("No wedge patterns found for the given parameters.")
                    st.markdown("""
                    **Try adjusting the parameters:**
                    - Try a different window size
                    - Adjust the extrema detection order
                    - Modify the slope tolerance
                    """)
                else:
                    patterns.sort(key=lambda x: x['start_idx'])
                    overview_fig = plotly_visualize_wedge_patterns(prices, patterns, local_window_size)
                    patterns_placeholder.plotly_chart(overview_fig, use_container_width=True)
                    st.subheader("Detected Wedge Pattern Details")
                    summary_data = []
                    for pat in patterns:
                        summary_data.append({
                            "Pattern Type": pat['type'],
                            "Start": safe_date_format(pat['start_time']),
                            "End": safe_date_format(pat['end_time']),
                            "Duration (pts)": pat['duration'],
                            "Slope (Peaks)": f"{pat['slope_peaks']:.4f}",
                            "Slope (Troughs)": f"{pat['slope_troughs']:.4f}",  # <-- fix key here
                            "Breakout": (pat['breakout_dir'] or "N/A")
                        })
                    pattern_df = pd.DataFrame(summary_data)
                    st.dataframe(pattern_df, use_container_width=True)
                    st.subheader("Individual Pattern Analysis")
                    if len(patterns) > 0:
                        tabs = st.tabs([f"Pattern {i+1}: {p['type']}" for i, p in enumerate(patterns)])
                        for i, pattern in enumerate(patterns):
                            with tabs[i]:
                                pattern_date = safe_date_format(pattern['start_time'])
                                st.write(f"{pattern['type']} at {pattern_date} (Index: {pattern['start_idx']})")
                                pattern_detail_fig = plotly_visualize_wedge_pattern(prices, pattern, local_window_size)
                                st.plotly_chart(pattern_detail_fig, use_container_width=True)
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
                                st.markdown("**Pattern Interpretation:**")
                                if pattern['type'] == "Rising Wedge":
                                    st.markdown("""
                                    **Rising Wedge (Bearish Pattern):**
                                    - Characterized by converging trendlines both sloping upward
                                    - The lower trendline has a steeper slope than the upper trendline
                                    - Usually signals a potential reversal from an uptrend to a downtrend
                                    - Confirmation occurs when price breaks below the lower trendline
                                    """)
                                else:
                                    st.markdown("""
                                    **Falling Wedge (Bullish Pattern):**
                                    - Characterized by converging trendlines both sloping downward
                                    - The upper trendline has a steeper slope than the lower trendline
                                    - Usually signals a potential reversal from a downtrend to an uptrend
                                    - Confirmation occurs when price breaks above the upper trendline
                                    """)
                    else:
                        st.info("No patterns detected to view in detail.")
                    st.subheader("Pattern Types Found")
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
    if run_analysis:
        run_wedge_pattern_analysis()
    else:
        st.info("""
        👈 Customize your analysis parameters in the sidebar and click \"Run Analysis\" to start.
        This app will:
        1. Fetch historical price data for your selected stock
        2. Compute the matrix profile to find recurring patterns
        3. Identify rising and falling wedge patterns based on your criteria
        4. Visualize detected patterns with detailed annotations
        **What are Wedge Patterns?**
        - **Rising Wedge:** A bearish reversal pattern with converging trendlines both sloping upward. The lower trendline rises more steeply than the upper trendline. A downward breakout signals potential bearish movement.
        - **Falling Wedge:** A bullish reversal pattern with converging trendlines both sloping downward. The upper trendline falls more steeply than the lower trendline. An upward breakout signals potential bullish movement.
        """)

# App router
if app_mode == "Head and Shoulders":
    run_head_shoulders_app()
elif app_mode == "Double Top/Bottom":
    run_double_top_bottom_app()
elif app_mode == "Wedge":
    run_wedge_app()
