#  Resume AI App

A smart resume classification and job insights tool built with **Streamlit**, **Scikit-learn**, and **XGBoost**. Users input a resume, and the app predicts the most likely job field, suggests related job titles, and provides a salary estimate.

---

##  Features

-  **Resume Text Classification** using Logistic Regression / XGBoost
-  **Feature Engineering** for advanced NLP insights (e.g., vocab density, tech keywords)
-  **Confidence Score** with animated progress display
-  **Related Job Titles** suggestion
-  **Estimated Salary Range**
-  **Interactive UI** built with Streamlit

---

## Model Training
The model is trained on a dataset of labeled resumes with:

- Cleaned and tokenized text

- **10+ engineered features including**:

    - ``num_unique_words``, ``avg_word_length``, ``num_tech_keywords``

    - ``has_degree``, ``num_certs``, ``vocab_density`` and more

Supported Models: ``XGBClassifier``


##  Input Format
Paste raw resume text into the text box. You can include:

- Work experience

- Certifications

- Education

- Skills & projects

