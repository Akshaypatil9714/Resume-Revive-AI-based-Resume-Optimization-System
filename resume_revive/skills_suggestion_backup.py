# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# # Custom stop words specific to resumes
# custom_stop_words = [
#     'etc', 'like', 'just', 'also', 'well', 'make', 'go', 'used', 'know', 'good',
#     'get', 'need', 'use', 'want', 'see', 'choose', 'set', 'put', 'take', 'help',
#     'sure', 'keep', 'let', 'provide', 'include', 'due', 'different', 'similar',
#     'able', 'enough', 'every', 'likely', 'either', 'various', 'often', 'less',
#     'more', 'much', 'last', 'first', 'possible', 'particular', 'real', 'such',
#     'major', 'along', 'seem', 'many', 'another', 'however', 'new', 'old', 'high',
#     'long'
# ]

# # Extend the default NLTK stop words with custom ones
# extended_stop_words = list(set(stopwords.words('english')).union(custom_stop_words))

# # Function to preprocess text
# def preprocess_text(text):
#     import re
#     import string
#     from nltk.stem import WordNetLemmatizer
#     from nltk.tokenize import RegexpTokenizer
    
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>+', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\n', '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
    
#     tokenizer = RegexpTokenizer(r'\w+')
#     tokens = tokenizer.tokenize(text)
#     # stop_words = set(extended_stop_words.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in extended_stop_words]
    
#     return ' '.join(tokens)

# # Function to extract top TF-IDF features
# def extract_features(text, max_features=50):
#     vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), stop_words=extended_stop_words)
#     matrix = vectorizer.fit_transform([text])
#     feature_names = vectorizer.get_feature_names_out()
#     return feature_names

# # Function to suggest keywords by comparing resume and job description
# def suggest_keywords(resume_text, job_desc_text):
#     resume_keywords = extract_features(resume_text)
#     job_desc_keywords = extract_features(job_desc_text)
#     missing_keywords = set(job_desc_keywords) - set(resume_keywords)
#     return list(missing_keywords)


from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize DistilBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('/Users/akshaypatil/Resume_revive/saved_model', output_hidden_states=True, num_labels=10)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
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

# Function to preprocess text
def preprocess_text(text):
    import re
    import string
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # stop_words = set(extended_stop_words.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in extended_stop_words]
    
    return ' '.join(tokens)

# def get_distilbert_embedding(text):
#     """Generate a DistilBERT embedding for the given text."""
#     encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         output = model(**encoded_input)
#     return output.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling

def get_bert_embedding(text):
    """Generate a DistilBERT embedding for the given text using sequence classification model."""
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encoded_input)
    # Use hidden states from the last layer
    return outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()

# def extract_keywords(text, n=50):
#     """Extract keywords from text using TF-IDF."""
#     vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=n)
#     X = vectorizer.fit_transform([text])
#     keywords = vectorizer.get_feature_names_out()
#     return keywords

def extract_keywords(text, n=20):
    """Extract keywords from text using TF-IDF, avoiding redundant subphrases."""
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=1000)  # Increase max_features for better selection
    X = vectorizer.fit_transform([text])
    all_keywords = vectorizer.get_feature_names_out()

    # Filter out subphrases of longer phrases
    filtered_keywords = []
    all_keywords = sorted(all_keywords, key=len, reverse=True)  # Sort by length of keyword
    for keyword in all_keywords:
        if not any(keyword in other for other in filtered_keywords if other != keyword):
            filtered_keywords.append(keyword)
    
    return filtered_keywords[:n]  # Return the top 'n' keywords after filtering


# def extract_keywords_with_bert(text, top_n=20):
#     vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
#     tfidf_matrix = vectorizer.fit_transform([text])
#     feature_array = np.array(vectorizer.get_feature_names_out())
#     tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    
#     top_n_features = feature_array[tfidf_sorting][:top_n]
#     return top_n_features


def suggest_keywords(resume_text, job_desc_text):
    """Suggest keywords by comparing embeddings."""
    # Preprocess texts
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job_desc = preprocess_text(job_desc_text)

    # Extract keywords from job description
    job_desc_keywords = extract_keywords(preprocessed_job_desc)

    # Get embeddings
    resume_embedding = get_bert_embedding(preprocessed_resume)
    keyword_embeddings = {kw: get_bert_embedding(kw) for kw in job_desc_keywords}

    # Find missing keywords
    missing_keywords = []
    for kw, emb in keyword_embeddings.items():
        sim = cosine_similarity([emb], [resume_embedding])
        if sim < 0.5:  # Threshold to adjust based on needs
            missing_keywords.append(kw)

    return missing_keywords