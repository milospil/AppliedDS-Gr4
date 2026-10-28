'''
identify_trips.py
-----------------   
Script to segement time series data into different trips based on the trip start and end times. 
The trip start and end times are defined in the trips.csv file. The script will create a new table in the DuckDB database called trips, which will contain the segmented data for each trip.
'''

import duckdb as db
import polars as pl
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join("..", "..", "src"))
from cfg import DROP, STD_COLS, COUNT_COLS, SHORT, HERE, DATASET, PROCESSED, DB_PATH, load_extensions

