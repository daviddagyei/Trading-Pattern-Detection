from pattern_base import PatternDetector
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

class WedgeDetector(PatternDetector):
    def get_name(self):
        return "Wedge"

    def sidebar_params(self, st):
        params = {}
        st.sidebar.header("Parameters")
        st.sidebar.subheader("Stock Selection")
        params['ticker'] = st.sidebar.text_input("Stock Symbol", value="AAPL")
        st.sidebar.subheader("Time Frame")
        time_selection = st.sidebar.radio("Select Time Frame Method", ["Predefined Period", "Custom Date Range"])
        params['time_selection'] = time_selection
        interval_options = {"1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m", 
            "30 minutes": "30m", "1 hour": "1h", "1 day": "1d", "5 days": "5d", "1 week": "1wk", "1 month": "1mo", "3 months": "3mo"}
        if time_selection == "Predefined Period":
            interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=4)
            params['interval_code'] = interval_options[interval]
            period_options = {"1 day": "1d", "5 days": "5d", "1 month": "1mo", 
                "3 months": "3mo", "6 months": "6mo", "1 year": "1y", 
                "2 years": "2y", "5 years": "5y", "10 years": "10y", "max": "max"}
            period = st.sidebar.selectbox("Period", options=list(period_options.keys()), index=4)
            params['period_code'] = period_options[period]
            params['start_date'] = None
            params['end_date'] = None
        else:
            interval = st.sidebar.selectbox("Interval", options=list(interval_options.keys()), index=4)
            params['interval_code'] = interval_options[interval]
            from datetime import datetime, timedelta
            default_end_date = datetime.now()
            default_start_date = default_end_date - timedelta(days=180)
            start_date = st.sidebar.date_input("Start Date", value=default_start_date)
            end_date = st.sidebar.date_input("End Date", value=default_end_date)
            params['start_date'] = start_date.strftime("%Y-%m-%d")
            params['end_date'] = end_date.strftime("%Y-%m-%d")
            params['period_code'] = None
            if params['start_date'] >= params['end_date']:
                st.sidebar.error("End date must be after start date")
        st.sidebar.subheader("Pattern Detection Parameters")
        params['window_size'] = st.sidebar.slider("Window Size", min_value=10, max_value=200, value=50, step=5)
        params['top_k'] = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value=5)
        params['extrema_order'] = st.sidebar.slider("Extrema Detection Order", min_value=1, max_value=10, value=3)
        st.sidebar.subheader("Pattern Thresholds")
        slope_tol = st.sidebar.slider("Slope Tolerance (%)", min_value=1, max_value=50, value=10)
        params['slope_tol'] = slope_tol / 100.0
        params['run_analysis'] = st.sidebar.button("Run Analysis", type="primary")
        st.sidebar.markdown("---")
        st.sidebar.caption("Created with ❤️ using Streamlit")
        return params

    def run_detection(self, st, params):
        ticker = params['ticker']
        interval_code = params['interval_code']
        period_code = params['period_code']
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        time_selection = params['time_selection']
        window_size = params['window_size']
        top_k = params['top_k']
        extrema_order = params['extrema_order']
        slope_tol = params['slope_tol']
        run_analysis = params['run_analysis']

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

        if not run_analysis:
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
            return

        local_window_size = window_size
        with st.spinner('Fetching price data...'):
            try:
                prices = fetch_price_data(symbol=ticker, interval=interval_code, period=period_code, 
                                         start_date=start_date, end_date=end_date)
                if prices is None or len(prices) == 0:
                    if time_selection == "Predefined Period":
                        st.error(f"No data returned for symbol {ticker} with interval {interval_code} and period {period_code}.")
                    else:
                        st.error(f"No data returned for symbol {ticker} with interval {interval_code} between {start_date} and {end_date}.")
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
                            "Slope (Troughs)": f"{pat['slope_troughs']:.4f}",
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
                return
