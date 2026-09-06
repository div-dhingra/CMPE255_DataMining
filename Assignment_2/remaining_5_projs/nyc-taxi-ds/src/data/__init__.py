"""Data processing and understanding package."""
from src.data.generator import generate_synthetic_taxi_data
from src.data.loader import load_dataset, clean_and_filter_data, get_raw_data_path
from src.data.eda import generate_eda_report, compute_summary_statistics, compute_temporal_patterns, compute_hotspots
