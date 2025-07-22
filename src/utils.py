from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfWrapper(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_class_): _to create transformers, from sklearn package_
        TransformerMixin (_class_): _to create transformers, from sklearn package_
    """
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        X = self._preprocess(X)
        return self.vectorizer.fit(X)

    def transform(self, X):
        X = self._preprocess(X)
        return self.vectorizer.transform(X)
    
    def _preprocess(self, X):
       
        return [str(text) if text is not None else "" for text in X]
    


def ravel_values(x):
    return x.values.ravel()
