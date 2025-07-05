tech_keywords = {
    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "r", "matlab", "scala", "go", "ruby", "perl", "sql", "pl/sql",

    # Data Science & Machine Learning
    "machine learning", "deep learning", "tensorflow", "keras", "pytorch", "sklearn", "scikit-learn", "xgboost", 
    "pandas", "numpy", "matplotlib", "seaborn", "statsmodels", "nlp", "text mining", "data science", 
    "data analysis", "data analytics", "exploratory data analysis", "predictive modeling", "classification", 
    "regression", "clustering", "unsupervised learning", "supervised learning",

    # Big Data & Data Engineering
    "hadoop", "spark", "hive", "pig", "mapreduce", "flink", "databricks", "airflow", "kafka",
    "etl", "data pipeline", "data warehouse", "data lake", "aws glue", "redshift",

    # Web Development & Design
    "html", "css", "javascript", "react", "angular", "vue", "bootstrap", "tailwind", "nodejs", "express", 
    "web design", "ux", "ui", "wordpress", "php", "django", "flask",

    # DevOps & Cloud
    "devops", "docker", "kubernetes", "jenkins", "ansible", "terraform", "aws", "azure", "gcp", 
    "ci/cd", "linux", "bash", "shell scripting",

    # Databases
    "mysql", "postgresql", "oracle", "mongodb", "sql server", "nosql", "db2", "sqlite", "dynamodb", "database", 

    # Testing & Automation
    "selenium", "junit", "pytest", "automation testing", "manual testing", "unit testing", "qa", "testcase",
    "cucumber", "load testing", "performance testing", "jmeter",

    # SAP & Enterprise Tools
    "sap", "sap hana", "sap abap", "sap bw", "sap fico", "sap mm", "sap sd", "sap basis",

    # Networking & Security
    "network security", "firewall", "vpn", "penetration testing", "ethical hacking", "wireshark", 
    "ids", "ips", "information security", "cybersecurity", "encryption", "nmap",

    # Blockchain
    "blockchain", "solidity", "web3", "smart contracts", "ethereum", "nft", "decentralized",

    # General Tech Tools
    "git", "github", "jira", "bitbucket", "vs code", "notepad++", "intellij", "eclipse",
    "tableau", "power bi", "excel", "sas", "lookml", "snowflake", "looker",

    # Business / Project / Analyst Roles
    "project management", "pmo", "agile", "scrum", "business analysis", "stakeholder", 
    "requirement gathering", "brd", "frd", "risk analysis", "reporting", "operations",

    # Engineering Keywords
    "solidworks", "autocad", "mechanical", "civil", "electrical", "plc", "embedded systems",
    "microcontrollers", "vlsi", "matlab", "control systems", "circuit design",

    # HR, Arts, Sales, and Other Soft Roles
    "recruitment", "employee engagement", "hr policies", "performance management", 
    "training", "talent acquisition", "sales", "marketing", "negotiation", "advertising", 
    "copywriting", "illustration", "drawing", "painting", "fitness", "nutrition", "yoga",
}

def count_keywords(text):
    return sum(1 for word in text.lower().split() if word in tech_keywords)


def num_unique_words(text):
    return len(set(text.split()))

def avg_word_lenght(text):
    return lambda text: sum(len(text) for w in text.split()) / max(len(text.split()), 1)


def has_management_terms(text: str) -> int:
    management_keywords = {"managed", "led", "coordinated", "supervised", "directed", "oversaw", "organized", "executed"}
    text_lower = text.lower()
    return int(any(term in text_lower for term in management_keywords))

import re

def num_certifications(text: str) -> int:
    cert_patterns = [
        r"\bcertified\b", r"\bcertification\b", r"\bcertificate\b",
        r"\bcert\b", r"\bcompleted\b.*?\bcourse\b", r"\btraining\b"
    ]
    count = 0
    text_lower = text.lower()
    for pattern in cert_patterns:
        count += len(re.findall(pattern, text_lower))
    return count

def num_languages_mentioned(text: str) -> int:
    languages = {
        # Programming
        "python", "java", "c++", "c#", "sql", "r", "javascript", "html", "css", "matlab", "bash", "perl",
        # Spoken
        "english", "french", "spanish", "german", "hindi", "mandarin", "arabic"
    }
    text_lower = text.lower()
    return sum(1 for lang in languages if lang in text_lower)


def has_degree_mention(text: str) -> int:
    degree_keywords = {"bsc", "msc", "mba", "phd", "btech", "mtech", "bachelor", "master", "undergraduate", "postgraduate"}
    text_lower = text.lower()
    return int(any(degree in text_lower for degree in degree_keywords))

def word_count(txt):
    return len(txt.split())
