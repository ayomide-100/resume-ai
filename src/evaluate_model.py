import pandas as pd
from feature_engineering import *
from data.external.salary_data import salary_with_titles


def vocab_density(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    return unique_words / (total_words if total_words > 0 else 1)



def predict_resume(text: str, pipeline, encoder):

    feature= {
        "Resume": text,
        "num_unique_words": num_unique_words(text),
        "avg_word_length": average_word_length(text),
        "num_tech_keywords": count_keywords(text),
        "has_management_terms": has_management_terms(text),
        "has_degree": has_degree_mention(text),
        "num_certs": num_certifications(text),
        "num_langs": num_languages_mentioned(text),
        "word_count": word_count(text),
        "vocab_density": vocab_density(text),

}
    
    X_input = pd.DataFrame([feature]) 
    print(X_input.T)
    print(X_input.columns.tolist())
    print(X_input.shape)
    
    pred = pipeline.predict(X_input)[0]
    proba_vector = pipeline.predict_proba(X_input)[0]
    pred = proba_vector.argmax()
    prob = proba_vector[pred]
    label = encoder.inverse_transform([pred])[0]
    tilte = str(label).title()
    if feature["num_certs"] >= 3 and feature["num_langs"] >= 4 and feature["has_management_terms"] == True:
        salary = salary_with_titles[tilte]
        exp = salary["senior"]
    elif feature["num_certs"] >= 2 and feature["num_langs"] >= 3 and feature["has_management_terms"] == False:
        salary = salary_with_titles[tilte]
        exp = salary["mid"]
    elif feature["num_certs"] >= 0 and feature["num_langs"] >= 1 and feature["has_management_terms"] == False:
        salary = salary_with_titles[tilte]
        exp = salary["junoir"]
    

    
    return {"prediction": label, "confidence": round(prob * 100, 2), 
            "salary range": exp["salary_range"], "potential job titles" : exp["job_titles"]}