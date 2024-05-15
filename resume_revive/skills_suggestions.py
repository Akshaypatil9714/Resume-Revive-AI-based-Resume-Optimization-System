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

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('/Users/akshaypatil/Resume_revive/saved_model', output_hidden_states=True, num_labels=10)

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
    """Generate a BERT embedding for the given text using sequence classification model."""
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad(): # Ensure gradients are not computed to save resources
        outputs = model(**encoded_input)
    return outputs.hidden_states[-1].mean(dim=1).squeeze().numpy() # Return the mean of the last hidden state

def extract_keywords(text, n=50):
    """Extract keywords from text using TF-IDF, avoiding redundant subphrases."""
    # Initialize a CountVectorizer to convert text to a matrix of token counts
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)
    X = vectorizer.fit_transform([text])
    all_keywords = vectorizer.get_feature_names_out()

    # Filter out subphrases of longer phrases
    filtered_keywords = []
    all_keywords = sorted(all_keywords, key=len, reverse=True)  # Sort by length of keyword
    for keyword in all_keywords:
        if not any(re.search(r'\b' + re.escape(keyword) + r'\b', other) for other in filtered_keywords):
            filtered_keywords.append(keyword)
    
    return filtered_keywords[:n]

def suggest_keywords(resume_text, job_desc_text):
    """Suggest keywords by comparing embeddings."""
    # Preprocess both resume and job description texts
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job_desc = preprocess_text(job_desc_text)

    # Extract keywords from job description using TF-IDF
    job_desc_keywords = extract_keywords(preprocessed_job_desc)

    # Generate embeddings for resume and each keyword
    resume_embedding = get_bert_embedding(preprocessed_resume)
    keyword_embeddings = {kw: get_bert_embedding(kw) for kw in job_desc_keywords}

    # Identify keywords that are not well-represented in the resume
    missing_keywords = []
    for kw, emb in keyword_embeddings.items():
        sim = cosine_similarity([emb], [resume_embedding])
        if sim[0][0] < 0.5:  # Set a threshold for keyword similarity
            missing_keywords.append(kw)

    return missing_keywords

