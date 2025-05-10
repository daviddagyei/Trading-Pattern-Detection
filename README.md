# Trading Pattern Detection App

A modular, extensible Streamlit application for detecting classic trading chart patterns (Head & Shoulders, Double Top/Bottom, Wedge) in financial time series data. The app is designed with an object-oriented architecture, making it easy to add new pattern detectors and maintain the codebase.

## Features

- **Pattern Detection:**
  - Head & Shoulders
  - Double Top / Double Bottom
  - Rising and Falling Wedges
- **Interactive Streamlit UI:**
  - Select pattern type and parameters from the sidebar
  - Visualize detected patterns on price charts (Matplotlib and Plotly)
- **Modular OOP Design:**
  - Each pattern detector is a separate class/file, inheriting from a common abstract base class
  - Easy to add new patterns by implementing a new detector class
- **Helper Utilities:**
  - Matrix profile motif search (via STUMPY)
  - Extrema detection
  - Data fetching (via yfinance)
  - Visualization helpers

## File Structure

```
Trading-Pattern-Detection-1/
├── main_app.py                  # Streamlit entry point, pattern router
├── pattern_base.py              # Abstract base class for pattern detectors
├── head_shoulders_detector.py   # Head & Shoulders detector class
├── double_top_bottom_detector.py# Double Top/Bottom detector class
├── wedge_detector.py            # Wedge pattern detector class
├── wedges_helpers.py            # Wedge-specific helpers (motif search, extrema, plotting)
├── streamlit_helpers.py         # Shared Streamlit UI helpers
├── README.md                    # This file
├── tests/                       # Parameter tuning and test scripts
└── ... (other helpers, cache, etc.)
```

## Getting Started

### 1. Install Dependencies

Create a virtual environment (recommended) and install required packages:

```bash
pip install streamlit yfinance stumpy plotly matplotlib pandas numpy scipy
```

### 2. Run the App

From the project root directory:

```bash
streamlit run main_app.py
```

The app will open in your browser. Use the sidebar to select a pattern, set parameters, and visualize detections.

## How It Works

- **main_app.py**: Streamlit UI. Lets the user select a pattern type, then instantiates and runs the corresponding detector class.
- **pattern_base.py**: Defines the `PatternDetector` abstract base class. All detectors inherit from this and implement the required interface.
- **[pattern]_detector.py**: Each file implements a detector class for a specific pattern, handling parameter input, detection logic, and visualization.
- **wedges_helpers.py**: Contains motif search, extrema detection, and visualization functions for wedge patterns.
- **streamlit_helpers.py**: Shared UI and utility functions for Streamlit.

## Detection Approach

This app uses a combination of time series motif discovery, local extrema analysis, and trendline fitting to robustly detect classic chart patterns in financial data. The approach is designed to be data-driven and adaptable to different timeframes and assets.

### Pattern Detection Methodology

- **Matrix Profile Motif Search:**
  - For patterns like wedges and double tops/bottoms, the app uses the matrix profile (via the STUMPY library) to efficiently identify repeated or similar subsequences (motifs) in the price series. This helps surface candidate regions where patterns may exist, even in noisy data.
- **Local Extrema Detection:**
  - Within each candidate region, the app finds local maxima (peaks) and minima (troughs) using a sliding window approach. This is crucial for identifying the key points that define chart patterns (e.g., the three peaks/troughs in a Head & Shoulders pattern).
- **Trendline Fitting and Classification:**
  - For wedge patterns, the app fits linear trendlines to the peaks and troughs using least-squares regression. The slopes and relative positions of these lines are used to classify the pattern as a rising wedge, falling wedge, or neither.
- **Pattern-Specific Logic:**
  - Each detector class implements logic tailored to its pattern. For example, the Head & Shoulders detector checks for the characteristic left shoulder, head, and right shoulder structure, while the Double Top/Bottom detector looks for two prominent peaks or troughs separated by a valley or peak.
- **Breakout Detection:**
  - Optionally, the app can annotate breakout points, where the price moves outside the pattern's boundaries, which is often of interest to traders.

### Function Overview

- **fetch_price_data:** Downloads historical price data using yfinance, returning a pandas Series of closing prices.
- **compute_matrix_profile:** Computes the matrix profile for a price series and window size, returning motif distances and indices for efficient subsequence similarity search.
- **extract_motif_candidates:** Extracts the top-k non-overlapping motif pairs from the matrix profile, providing candidate regions for pattern detection.
- **detect_extrema:** Identifies local maxima (peaks) and minima (troughs) in a price subsequence, which are used to anchor pattern boundaries.
- **check_wedge_pattern:** Given a subsequence and its extrema, fits trendlines and classifies the region as a rising wedge, falling wedge, or neither, based on the slopes and convergence of the lines.
- **safe_date_format:** Utility to format dates for display in plots and annotations.
- **plotly_visualize_wedge_patterns / plotly_visualize_wedge_pattern:** Create interactive Plotly charts to visualize detected wedge patterns, including trendlines, extrema, and breakout points.
- **visualize_wedge_patterns / visualize_wedge_patterns_simple:** Generate Matplotlib plots for detected patterns, highlighting regions, trendlines, and key points for quick analysis.

Each detector class (e.g., `HeadShouldersDetector`, `DoubleTopBottomDetector`, `WedgeDetector`) orchestrates these functions to:
- Accept user parameters from the Streamlit sidebar
- Fetch and preprocess data
- Run motif/extrema/trendline analysis
- Display results interactively in the app

This modular approach makes it easy to extend the app with new pattern types or detection techniques.

## Adding a New Pattern Detector

1. Create a new file, e.g. `my_pattern_detector.py`.
2. Inherit from `PatternDetector` in `pattern_base.py`.
3. Implement the required methods for parameter input, detection, and visualization.
4. Register your detector in `main_app.py` so it appears in the UI.

## Testing and Parameter Tuning

The `tests/` folder contains scripts for tuning detection parameters and validating pattern detection logic.

## Requirements

- Python 3.8+
- See `pip install` command above for required packages


## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [STUMPY](https://stumpy.readthedocs.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Plotly](https://plotly.com/python/)
- [Matplotlib](https://matplotlib.org/)

---

For questions or contributions, please open an issue or pull request.
