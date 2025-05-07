# Trading Pattern Detection

A Python application that uses advanced algorithms to detect head and shoulders patterns in financial market data using matrix profile techniques.

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

### Running the Streamlit App

The main interface is a Streamlit application that provides interactive controls:

```bash
streamlit run streamlit_app.py
```

This will open the application in your default web browser where you can:
- Select a stock symbol
- Choose time intervals and periods
- Adjust detection parameters
- Visualize detected patterns

### Parameter Configuration

The app allows you to adjust several parameters to fine-tune head and shoulders pattern detection:

- **Window Length**: Controls the number of data points in each pattern (larger for longer-term patterns)
- **Extrema Detection Order**: Controls sensitivity in identifying peaks and troughs
- **Shoulder Height Tolerance**: Determines acceptable difference between shoulders in head-and-shoulders patterns
- **Trough Alignment Tolerance**: Controls how horizontal the neckline should be in patterns

### For Hourly Time Frame Analysis

Recommended starting parameters for hourly data:
- Window Length: 30 (approximately 4 trading days)
- Extrema Detection Order: 3
- Shoulder Height Tolerance: 0.1
- Trough Alignment Tolerance: 0.2


## Dependencies

- pandas, numpy: Data manipulation
- yfinance: Market data acquisition
- STUMPY: Matrix Profile implementation for pattern detection
- Streamlit: Interactive web interface
- Plotly: Interactive chart visualization
- scipy: Signal processing for extrema detection

## Contributions

Contributions, issues, and feature requests are welcome!
