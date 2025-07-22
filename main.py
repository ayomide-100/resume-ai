import pandas as pd
import sys, os
import re
import joblib
sys.path.append(os.path.abspath("src"))
from src.utils import TfidfWrapper, ravel_values
from src.evaluate_model import predict_resume
from src.feature_engineering import *
from src.data_preprocessing import *

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath('data'))


model = joblib.load('models/model.pkl')
encoder = joblib.load('models/label_encoder.pkl')


def vocab_density(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    
    return unique_words / (total_words if total_words > 0 else 1)







samp = """Jane Smith  
Email: jane.smith@example.com  
Phone: +1 (123) 456-7890  
Location: Seattle, WA  
Portfolio: janesmith.dev  
GitHub: github.com/janesmith  

Professional Summary:  
Detail-oriented and results-driven Software Engineer with 4+ years of experience in designing and developing scalable web applications. Passionate about clean code, system architecture, and continuous learning. Skilled in front-end and back-end development, CI/CD pipelines, and DevOps practices.

Technical Skills:  
- Languages: JavaScript, TypeScript, Python, Java, Go  
- Frameworks: React.js, Next.js, Node.js, Express.js, Django  
- Databases: PostgreSQL, MongoDB, MySQL  
- DevOps: Docker, Jenkins, GitHub Actions, Kubernetes  
- Cloud: AWS (EC2, S3, RDS, CloudFront), Azure, Netlify  
- Tools: Git, VS Code, Jira, Notion, Postman, Figma  

Certifications:  
- AWS Certified Developer – Associate  
- Full-Stack Web Development Certificate – freeCodeCamp  

Education:  
B.Sc. in Software Engineering – University of Washington (2015 – 2019)

Work Experience:

Software Engineer – CodeCrafters Inc. | Seattle, WA | Feb 2021 – Present  
- Built and maintained a cloud-native microservice architecture using Docker and Kubernetes.  
- Reduced API latency by 35% through optimized SQL queries and caching strategies.  
- Led frontend development for a customer portal built with React and TypeScript.

Junior Software Engineer – DevSpark Solutions | Remote | Jul 2019 – Jan 2021  
- Implemented REST APIs in Django to support mobile applications.  
- Wrote unit and integration tests achieving 85% code coverage.  
- Collaborated with designers and QA teams in Agile sprints.

Projects:  
- **DevTrack**: A productivity app built with MERN stack to manage developer tasks and track time.  
- **CodeShare**: Real-time collaborative code editor built using WebSockets and Node.js.

Languages:  
English (Fluent), German (Basic)

Interests:  
Open source contribution, mentoring junior developers, mountain biking
"""
def clean_input(samp: str) -> str:
    samp = remove_dates(samp)
    samp = remove_newlines(samp)
    samp = remove_non_ascii_regex(samp)
    samp = remove_urls(samp)
    samp = clean_text(samp)
    samp = strip_column_whitespace_inplace(samp)
    return samp


sample_resume = clean_input(samp= samp)



result = predict_resume(text= sample_resume, encoder= encoder, pipeline= model )
print(result)
