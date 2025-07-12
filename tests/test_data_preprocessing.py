import sys, os
sys.path.append(os.path.abspath("../src"))
import pandas as pd
import numpy as np
from src.pipeline import full_resume_preprocessing as fp


df_sample = pd.read_csv('../data/raw/resume.csv').sample(3, random_state= 42)


df_resume = df_sample.drop("Category", axis= 1)

df_processed = fp.fit_transform(df_resume)


print(df_processed)