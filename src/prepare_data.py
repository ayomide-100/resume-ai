import sys
import os
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from src.data_preprocessing import *
from src.feature_engineering import *
from src.feature_engineering import tech_keywords as default_tech_keywords
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y=None):
        print("Type received in fit:", type(X))
        print("X columns:", X.columns if isinstance(X, pd.DataFrame) else "Series")
        if isinstance(X, pd.Series):
            X = X.to_frame()

        return self
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = X.copy()
        X["Resume"] = X["Resume"].apply(clean_text)
        X["Resume"] = X["Resume"].apply(remove_dates)
        X["Resume"] = X["Resume"].apply(remove_newlines)
        X["Resume"] = X["Resume"].apply(remove_urls)
        X["Resume"] = X["Resume"].apply(remove_non_ascii_regex)
        X["Resume"] = X["Resume"].apply(strip_column_whitespace_inplace)
        return X





class CustomFeatureEng(BaseEstimator, TransformerMixin):
    def __init__(self, tech_keywords=None) -> None:
        self.tech_keywords = tech_keywords if tech_keywords is not None else default_tech_keywords
    def fit(self, X, y =None):
        print("Type received in fit for feature eng:", type(X))
        print("X columns:", X.columns if isinstance(X, pd.DataFrame) else "Series")
        if isinstance(X, pd.Series):
            X = X.to_frame()
    

        return self
    
    def transform(self, X):
        print(type(X))
        print(X.columns)
        X = X.copy()

        if "Resume" not in X.columns:
            raise ValueError("Input Dataframe must contain a 'Resume' column")
        
        X["word_count"] = X["Resume"].apply(word_count)
        X["num_unique_words"] = X["Resume"].apply(num_unique_words)
        X["avg_word_length"] = X["Resume"].apply(average_word_length)
        X["num_tech_keywords"] = X["Resume"].apply(count_keywords)
        X["vocab_density"] = X["num_unique_words"] / X["word_count"].replace(0, 1)
        X["has_management_terms"] = X["Resume"].apply(has_management_terms)
        X["num_certs"] = X["Resume"].apply(num_certifications)
        X["num_langs"] = X["Resume"].apply(num_languages_mentioned)
        X["has_degree"] = X["Resume"].apply(has_degree_mention)

        return X

