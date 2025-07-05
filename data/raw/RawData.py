import os
import sys
import pandas as pd

sys.path.append(os.path.abspath("../data"))

def load_csv_data():
    return pd.read_csv("data/raw/resume.csv")


df = load_csv_data()