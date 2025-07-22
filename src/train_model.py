import pandas as pd
import sys, os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath('../data'))
sys.path.append(os.path.abspath('../models'))
from utils import TfidfWrapper, ravel_values
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



df_in = pd.read_csv('data/processed/processed_resume.csv')
df = df_in.drop('Unnamed: 0', axis= 1)
from sklearn.model_selection import train_test_split



def extract_features(text: str) ->list:
    return[]


X_full = df[['Resume', 'num_unique_words', 'avg_word_length',
       'num_tech_keywords', 'has_management_terms', 'has_degree', 'num_certs',
       'num_langs', 'word_count', 'vocab_density']]


y_raw = df["Category"]


X_train, X_test, y_train, y_test = train_test_split(X_full, y_raw, stratify=y_raw, test_size=0.2, random_state=42)

encoder = joblib.load('models/label_encoder.pkl')
pipeline = joblib.load('models/xgboost_model.pkl')



y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

print(X_train)
print(X_train.columns)
pipeline.fit(X_train, y_train)


print(f"Accuracy Score: {pipeline.score(X_test ,y_test)}")
y_test_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names= encoder.classes_))


from sklearn import set_config
set_config(display='diagram')

print(pipeline)



joblib.dump(pipeline, 'models/model.pkl')



