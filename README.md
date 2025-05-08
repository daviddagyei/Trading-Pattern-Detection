# Trading Pattern Detection

This repository contains a collection of tools for detecting common trading patterns in financial time series data. It uses matrix profile algorithms to identify potential chart patterns that traders look for when making decisions.

## Available Pattern Detectors

### 1. Double Top/Bottom Pattern Detector

The Double Top/Bottom detector identifies these common reversal patterns:

- **Double Top**: A bearish reversal pattern consisting of two peaks at roughly the same level with a valley between them
- **Double Bottom**: A bullish reversal pattern consisting of two troughs at roughly the same level with a peak between them

Run with:
```
streamlit run double_top_bottom_app.py
```

### 2. Head and Shoulders Pattern Detector

The Head and Shoulders detector identifies:

- **Head and Shoulders**: A bearish reversal pattern with a higher peak (head) between two lower peaks (shoulders)
- **Inverse Head and Shoulders**: A bullish reversal pattern with a lower trough (head) between two higher troughs (shoulders)

Run with:
```
streamlit run streamlit_app.py
```

## Technical Approach

The pattern detection uses:

1. **Matrix Profile**: An algorithm for time series data mining that finds recurring patterns (motifs)
2. **Extrema Detection**: Identifies local peaks and troughs in price segments
3. **Pattern Criteria Validation**: Applies specific rules to validate if a segment contains a particular pattern

## Features

- **Head and Shoulders Pattern Detection**: 
  - Identifies classic and inverse head and shoulders patterns
  - Uses statistical methods to validate pattern criteria
  - Provides detailed visualizations of detected patterns
  
- **Matrix Profile Analysis**: Uses the STUMPY library to efficiently find recurring patterns in time series data

- **Interactive Web Interface**: Streamlit-based application with:
  - Custom parameter adjustment for pattern detection
  - Interactive charts and visualizations
  - Ability to analyze multiple stocks and timeframes

## Dependencies

- Python 3.x
- pandas
- numpy
- scipy
- stumpy (for matrix profile computation)
- matplotlib
- plotly
- streamlit
- yfinance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/daviddagyei/Trading-Pattern-Detection.git
cd Trading-Pattern-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run one of the Streamlit apps:
   ```
   streamlit run double_top_bottom_app.py
   ```
   or
   ```
   streamlit run streamlit_app.py
   ```

2. In the web interface:
   - Enter a stock symbol 
   - Select time frame and interval
   - Adjust detection parameters
   - Click "Run Analysis"

## Parameters

- **Window Length**: The number of data points in each subsequence (adjust based on pattern size)
- **Number of Motifs**: How many potential pattern candidates to extract
- **Extrema Detection Order**: Points compared on each side for peak/trough detection
- **Similarity Tolerance**: Maximum allowed difference between pattern components (peaks/troughs)

### For Hourly Time Frame Analysis

Recommended starting parameters for hourly data:
- Window Length: 30 (approximately 4 trading days)
- Extrema Detection Order: 3
- Shoulder Height Tolerance: 0.1
- Trough Alignment Tolerance: 0.2

## Contributions

Contributions, issues, and feature requests are welcome!

## License

MIT License
