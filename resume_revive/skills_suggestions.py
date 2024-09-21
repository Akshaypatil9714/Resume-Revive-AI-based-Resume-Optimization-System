import re
import string
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('wordnet')  # Ensure the WordNet data is downloaded for lemmatization
nltk.download('omw-1.4')
nltk.download('punkt_tab')
import google.generativeai as genai
import os

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('/Users/akshaypatil/Resume_revive/bert_finetuned_job_resume')
model = BertForSequenceClassification.from_pretrained('/Users/akshaypatil/Resume_revive/bert_finetuned_job_resume', output_hidden_states=True)

# Custom stop words specific to resumes
custom_stop_words = [
    'etc', 'like', 'just', 'also', 'well', 'make', 'go', 'used', 'know', 'good',
    'get', 'need', 'use', 'want', 'see', 'choose', 'set', 'put', 'take', 'help',
    'sure', 'keep', 'let', 'provide', 'include', 'due', 'different', 'similar',
    'able', 'enough', 'every', 'likely', 'either', 'various', 'often', 'less',
    'more', 'much', 'last', 'first', 'possible', 'particular', 'real', 'such',
    'major', 'along', 'seem', 'many', 'another', 'however', 'new', 'old', 'high',
    'long'
]

# Extend the default NLTK stop words with custom ones
extended_stop_words = list(set(stopwords.words('english')).union(custom_stop_words))


# Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def preprocess_text(text):
    """Preprocess the given text for NLP tasks. This includes cleaning, lowercasing, removing punctuation,
    removing URLs and numbers, tokenizing, and lemmatizing while removing stop words."""
    # Lowercase and remove URLs and HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text.lower()) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    # Remove punctuation and newlines
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove new line
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    # Tokenize text using a regular expression tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # Initialize lemmatizer and lemmatize tokens, filtering out stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in extended_stop_words]
    return ' '.join(tokens)

def get_bert_embedding(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encoded_input, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    return last_hidden_state.mean(dim=1).squeeze().numpy()

def extract_keywords(text, n=50):
    # Use TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)
    tfidf_matrix = tfidf.fit_transform([text])
    feature_names = tfidf.get_feature_names_out()
    
    # Get feature scores
    scores = tfidf_matrix.sum(axis=0).A1
    
    # Create a list of (keyword, score) tuples
    keyword_scores = list(zip(feature_names, scores))
    
    # Sort by score and get top n
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, score in keyword_scores[:n]]

def match_skills(job_keywords, resume_text):
    resume_text = preprocess_text(resume_text)
    matched_skills = []
    missing_skills = []
    
    for keyword in job_keywords:
        if keyword in resume_text:
            matched_skills.append(keyword)
        else:
            missing_skills.append(keyword)
    
    return matched_skills, missing_skills

def suggest_keywords(resume_text, job_desc_text):
    """Suggest keywords by comparing embeddings using the fine-tuned model."""
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job_desc = preprocess_text(job_desc_text)

    job_desc_keywords = extract_keywords(preprocessed_job_desc, n=50)
            
    matched_skills, missing_skills = match_skills(job_desc_keywords, preprocessed_resume)

    return missing_skills

def get_gemini_suggestions(resume_text, job_desc_text, bert_suggestions):
    model = genai.GenerativeModel('gemini-1.5-flash-001')
    
    prompt = f"""
    Analyze the following resume and job description:

    Resume: {resume_text}

    Job Description: {job_desc_text}

    BERT model suggestions for missing skills: {bert_suggestions}

    Based on this information, please:
    1. List the top 10 most relevant skills from the resume for this job.
    2. Suggest 5 skills or keywords that are missing from the resume but important for the job.
    3. Provide a brief explanation of why these skills are important for the role.
    4. Offer 3 specific recommendations to improve the resume for this job application.
    """

    response = model.generate_content(prompt)
    return response.text

def suggest_keywords_hybrid(resume_text, job_desc_text):
    preprocessed_job_desc = preprocess_text(job_desc_text)
    
    job_keywords = extract_keywords(preprocessed_job_desc, n=50)
    
    matched_skills, missing_skills = match_skills(job_keywords, resume_text)
    
    # Use embeddings for additional context
    resume_embedding = get_bert_embedding(preprocess_text(resume_text))
    
    refined_missing_skills = []
    for skill in missing_skills:
        skill_embedding = get_bert_embedding(skill)
        similarity = cosine_similarity([skill_embedding], [resume_embedding])[0][0]
        if similarity < 0.6:  # Adjust this threshold as needed
            refined_missing_skills.append(skill)
    
    # Get Gemini suggestions
    try:
        gemini_suggestions = get_gemini_suggestions(resume_text, job_desc_text, refined_missing_skills)
    except Exception as e:
        print(f"Error getting Gemini suggestions: {e}")
        gemini_suggestions = "Unable to get Gemini suggestions due to an error."

    return gemini_suggestions