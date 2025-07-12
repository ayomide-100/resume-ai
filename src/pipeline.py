import os, sys
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from prepare_data import CustomFeatureEng, CustomPreprocessor
from data_preprocessing import reflect_log_transform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))



class TfidfWrapper(BaseEstimator, TransformerMixin):
    """custom tfid wrapper

    Args:
        BaseEstimator (_class_): _to create transformers, from sklearn package_
        TransformerMixin (_class_): _to create transformers, from sklearn package_
    """
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = self._preprocess(X)
        return self.vectorizer.fit(X)

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = self._preprocess(X)
        return self.vectorizer.transform(X)
    
    def _preprocess(self, X):
       
        return [str(text) if text is not None else "" for text in X]
    


def ravel_values(x):
    return x.values.ravel()


flatten = FunctionTransformer(ravel_values, validate= False)
vectorizer = make_pipeline(flatten, TfidfWrapper(preprocessor = None, lowercase = False))


text_preprocessing = Pipeline([
    ("custom_text_preprocess", CustomPreprocessor())
])


feature_eng = Pipeline([
    ("feature_eng", CustomFeatureEng(tech_keywords=None)),
    ("scaler", StandardScaler())
])


log_transform_fucntion = FunctionTransformer(reflect_log_transform, validate= True)
new_cols = ["word_count","num_unique_words", "avg_word_length", "num_tech_keywords",
            "vocab_density", "has_management_terms", "num_certs", "num_langs", "has_degree"
            ]
log_transform = Pipeline([("log_trans", log_transform_fucntion, new_cols )])


full_preprocessing = ColumnTransformer([
    ("clean_resume", text_preprocessing, "Resume"),
    ("features", feature_eng, "Resume"),
    ("vectorizer", vectorizer, "Resume"),
    ("log_transformer", log_transform, new_cols)
    
])


LR_full_pipeline = Pipeline([
    ("preprocessing", full_preprocessing),
    ("log_reg",
     LogisticRegression(C= 10, penalty= 'l1', solver= 'liblinear', random_state= 42))
])


XGB_full_pipeline = Pipeline([
    ("preprocessing", full_preprocessing),
    ("xgb", XGBClassifier(colsample_bytree = 0.8, learning_rate= 0.1, 
                          max_depth= 3, n_estimators= 100, subsample= 1.0 ))
])


