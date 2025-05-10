#!/usr/bin/env python3
"""
Apply Optimal Parameters to Head and Shoulders Pattern Detector
--------------------------------------------------------------

This script updates the default parameter values in streamlit_app.py
to use the optimal values determined by parameter tuning.
"""

import re
import os

def update_streamlit_app_params():
    """Update the default parameters in streamlit_app.py with optimal values."""
    
    # Path to the streamlit app
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'streamlit_app.py')
    
    # Optimal parameters based on tuning results
    optimal_params = {
        'window_length': (30, 50),  # (old, new)
        'top_k': (3, 5),            # (old, new)
        # Order and tolerances remain unchanged as they are already optimal
    }
    
    print(f"Updating parameters in {app_path}...")
    
    try:
        # Read the current file content
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Make replacements
        # Update window_length
        content = re.sub(
            r'window_length = st\.sidebar\.slider\("Window Length \(m\)", min_value=\d+, max_value=\d+, value=(\d+)',
            f'window_length = st.sidebar.slider("Window Length (m)", min_value=10, max_value=100, value={optimal_params["window_length"][1]}',
            content
        )
        
        # Update top_k
        content = re.sub(
            r'top_k = st\.sidebar\.slider\("Number of Motifs to Extract", min_value=\d+, max_value=\d+, value=(\d+)',
            f'top_k = st.sidebar.slider("Number of Motifs to Extract", min_value=1, max_value=10, value={optimal_params["top_k"][1]}',
            content
        )
        
        # Save the updated content
        with open(app_path, 'w') as f:
            f.write(content)
        
        print("Parameters successfully updated!")
        print(f"  • Window Length: {optimal_params['window_length'][0]} → {optimal_params['window_length'][1]}")
        print(f"  • Top K: {optimal_params['top_k'][0]} → {optimal_params['top_k'][1]}")
        print(f"  • Extrema Detection Order: 3 (unchanged - already optimal)")
        print(f"  • Shoulder Height Tolerance: 0.1 (unchanged - already optimal)")
        print(f"  • Trough Alignment Tolerance: 0.2 (unchanged - already optimal)")
        
    except Exception as e:
        print(f"Error updating parameters: {e}")

if __name__ == "__main__":
    update_streamlit_app_params()