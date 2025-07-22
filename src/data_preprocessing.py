# Functions for data cleaning and preprocessing
import numpy as np
import re
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# Combining NLTK + Custom stopwords
nltk_stopwords = set(stopwords.words("english"))
custom_stopwords = {
    "skills", "experience", "objective", "work", "summary", "responsibilities",
    "job", "description", "curriculum", "vitae", "resume", "project", "organization"
}

all_stopwords = nltk_stopwords.union(custom_stopwords)

lemmatizer = WordNetLemmatizer()



def remove_dates(text: str) -> str:
    # Remove months and year-like patterns
    text = re.sub(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|'
                  r'sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b[\s\-.,]*\d{2,4}', '',
                  text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', text)  # e.g., 12/05/2020
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)  # e.g., 2020, 1999
    return text



def clean_text(text: str) -> str:
    """Removes data, punctuation and then makes text lowercase then
      tokenization and lemmatization

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    # Remove dates first
    text = remove_dates(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in all_stopwords
    ]

    return " ".join(clean_tokens)


def remove_urls(text: str) -> str:
    """
    Removes URLs in text
    """
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()




def remove_newlines(text: str) -> str:
    """
    Removes newline characters (\n, \r) from the text and replaces them with a space.
    """
    return text.replace('\n', ' ').replace('\r', ' ').strip()


def strip_column_whitespace_inplace(text):
    """
    Strips leading and trailing whitespace from all string entries in the specified column.
    This modifies the original DataFrame in place.

    
    """
    if isinstance(text, pd.Series):
        return text.astype(str).str.strip()
    elif isinstance(text, str):
        return text.strip()




def remove_non_ascii_regex(text: str) -> str:
    """
    Remove non-ASCII characters.
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)



def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    print(f"Removing outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]




def reflect_log_transform(series):
    if isinstance(series, pd.Series):
        max_val = series.max()
        reflected = max_val + 1 - series
        transformed = np.log1p(reflected)
        return transformed

