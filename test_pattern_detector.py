from data_fetcher import DataFetcher
from pattern_detector import ChartPatternDetector

def test_pattern_detector():
    """
    Test function to fetch sample data and run the pattern detection.
    It fetches data using DataFetcher, then passes the data to ChartPatternDetector,
    and finally prints out the detected patterns.
    """
    symbol = "AAPL"
    interval = "1d"
    start_date = "2023-01-01"
    end_date = "2023-01-31"

    fetcher = DataFetcher()
    df_quotes = fetcher.get_quotes(symbol, interval, start_date, end_date)
    
    detector = ChartPatternDetector(df_quotes)

    patterns = detector.detect_all(window=3)
    
    for pattern_type, pattern_list in patterns.items():
        print(f"\nDetected {pattern_type} patterns:")
        if pattern_list:
            for pattern in pattern_list:
                print(pattern)
        else:
            print("None detected.")

if __name__ == "__main__":
    test_pattern_detector()
