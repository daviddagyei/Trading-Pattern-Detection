from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Pattern:
    """
    Represents a detected chart pattern in a financial time series.

    Attributes:
        pattern_type (str): The type of pattern detected 
        start_index (Any): The starting index or timestamp of the detected pattern.
        end_index (Any): The ending index or timestamp of the detected pattern.
        key_points (Dict[str, Any]): A dictionary of key points that define the pattern 
        detection_date (str): The date when the pattern was detected, typically corresponding to the end index.
    """
    pattern_type: str
    start_index: Any
    end_index: Any
    key_points: Dict[str, Any]
    detection_date: str
