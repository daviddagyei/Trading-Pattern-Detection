import pandas as pd
from typing import List
from pattern_structures import Pattern
import tradingpatterns as tp 

class ChartPatternDetector:
    """
    A class to detect chart patterns by wrapping functions from the tradingpatterns module.
    This class provides methods to detect various trading patterns in a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the ChartPatternDetector with a copy of the input data.

        Parameters:
            data (pd.DataFrame): A DataFrame containing trading data with required columns.
        """
        self.data = data.copy()

    def find_head_shoulders(self, window: int = 3) -> List[Pattern]:
        """
        Detect head and shoulder patterns in the data.

        This method calls the `detect_head_shoulder` function from the tradingpatterns module,
        then iterates over the returned DataFrame to create Pattern objects for each occurrence.

        Parameters:
            window (int): The rolling window size used in the detection algorithm. Default is 3.

        Returns:
            List[Pattern]: A list of Pattern objects representing detected head and shoulder patterns.
        """
        df_patterns = tp.detect_head_shoulder(self.data.copy(), window=window)
        patterns = []
        for idx, row in df_patterns.iterrows():
            if pd.notna(row.get('head_shoulder_pattern')):
                patterns.append(Pattern(
                    pattern_type=row['head_shoulder_pattern'],
                    start_index=idx,  
                    end_index=idx,    
                    key_points={},    
                    detection_date=str(idx)
                ))
        return patterns

    def find_double_tops_bottoms(self, window: int = 3) -> List[Pattern]:
        """
        Detect double top and double bottom patterns in the data.

        This method uses the `detect_multiple_tops_bottoms` function from the tradingpatterns module,
        and then wraps the detected occurrences in Pattern objects.

        Parameters:
            window (int): The rolling window size used in the detection algorithm. Default is 3.

        Returns:
            List[Pattern]: A list of Pattern objects representing detected double top and double bottom patterns.
        """
        df_patterns = tp.detect_multiple_tops_bottoms(self.data.copy(), window=window)
        patterns = []
        for idx, row in df_patterns.iterrows():
            if pd.notna(row.get('multiple_top_bottom_pattern')):
                patterns.append(Pattern(
                    pattern_type=row['multiple_top_bottom_pattern'],
                    start_index=idx,
                    end_index=idx,
                    key_points={},
                    detection_date=str(idx)
                ))
        return patterns

    def find_triangle_patterns(self, window: int = 3) -> List[Pattern]:
        """
        Detect triangle patterns (ascending or descending) in the data.

        This method calls the `detect_triangle_pattern` function from the tradingpatterns module,
        then wraps the detection results into Pattern objects.

        Parameters:
            window (int): The rolling window size used in the detection algorithm. Default is 3.

        Returns:
            List[Pattern]: A list of Pattern objects representing detected triangle patterns.
        """
        df_patterns = tp.detect_triangle_pattern(self.data.copy(), window=window)
        patterns = []
        for idx, row in df_patterns.iterrows():
            if pd.notna(row.get('triangle_pattern')):
                patterns.append(Pattern(
                    pattern_type=row['triangle_pattern'],
                    start_index=idx,
                    end_index=idx,
                    key_points={},
                    detection_date=str(idx)
                ))
        return patterns

    def find_wedges(self, window: int = 3) -> List[Pattern]:
        """
        Detect wedge patterns in the data.

        This method calls the `detect_wedge` function from the tradingpatterns module,
        then iterates over the results to generate Pattern objects for wedge patterns.

        Parameters:
            window (int): The rolling window size used in the detection algorithm. Default is 3.

        Returns:
            List[Pattern]: A list of Pattern objects representing detected wedge patterns.
        """
        df_patterns = tp.detect_wedge(self.data.copy(), window=window)
        patterns = []
        if 'wedge_pattern' in df_patterns.columns:
            for idx, row in df_patterns.iterrows():
                if pd.notna(row.get('wedge_pattern')):
                    patterns.append(Pattern(
                        pattern_type=row['wedge_pattern'],
                        start_index=idx,
                        end_index=idx,
                        key_points={},
                        detection_date=str(idx)
                    ))
        return patterns

    def detect_all(self, window: int = 3) -> dict:
        """
        Run all available pattern detection methods and aggregate the results.

        This method calls individual detection methods (for head & shoulders, double tops/bottoms,
        triangles, wedges, etc.) and aggregates their outputs into a dictionary.

        Parameters:
            window (int): The rolling window size used in all detection algorithms. Default is 3.

        Returns:
            dict: A dictionary with pattern type names as keys and lists of Pattern objects as values.
        """
        return {
            "Head & Shoulders": self.find_head_shoulders(window),
            "Double Tops/Bottoms": self.find_double_tops_bottoms(window),
            "Triangles": self.find_triangle_patterns(window),
            "Wedges": self.find_wedges(window)
        }
