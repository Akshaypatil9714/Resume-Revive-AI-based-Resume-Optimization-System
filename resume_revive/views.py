from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from .validate_file import validate_pdf
from django.core.exceptions import ValidationError
from .models import UploadedFile
from .pdf_analysis import count_pdf_pages, extract_text_from_pdf, check_grammar
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.files.base import ContentFile

import json
from django.http import JsonResponse, HttpResponseNotAllowed
from django.views.decorators.http import require_POST
import logging
import re
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib.auth.decorators import login_required
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import wordnet
from .skills_suggestions import suggest_keywords, preprocess_text
from datetime import datetime
import os

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet', force=True)

stop_words = set(stopwords.words('english'))

logger = logging.getLogger(__name__)

# Create your views here.

def home(request):
    if request.user.is_authenticated:
        # Fetch only the files uploaded by the logged-in user
        uploaded_files = UploadedFile.objects.filter(user=request.user)
    else:
        # If no user is logged in, do not display any files
        uploaded_files = UploadedFile.objects.none()
    return render(request, "resume_revive/index.html", {'uploaded_files': uploaded_files})

def signup (request):
    
    if request.method=="POST":
       
        username = request.POST['username']
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']
        email = request.POST ['email']
        psw= request.POST['psw']
        psw_repeat= request.POST['psw_repeat']
        
        myuser = User.objects.create_user(username, email, psw)
        myuser.first_name = firstname
        myuser.last_name = lastname
        
        myuser.save()
        
        messages.success (request, "Your Account has been successfully created.")
        return redirect('signin')
    
    
    return render (request, "resume_revive/signup.html")

def signin(request):
    
    if request.method == 'POST':
        username = request. POST['username']
        psw= request. POST['psw']
        
        user = authenticate(username=username, password=psw)
        
        if user is not None:
            login (request, user)
            fname = user.first_name
            messages.success (request, "Logged in successfully")
            return redirect('home') 
        else:
            messages.error (request, "Bad Credentials!")
            return redirect( 'signin')
    
    
    return render (request, "resume_revive/signin.html")

def signout (request):
    logout (request)
    messages.success (request, "Logged Out Successfully!")
    return redirect( 'signin')


def upload_file(request):
    if request.method == 'POST' and request.FILES.get('filename', None):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file = request.FILES['filename']
        job_description = request.POST.get('jobDescription', '')  # Retrieve job description from POST data

        if not request.user.is_authenticated:
            messages.error(request, "You must be logged in to upload files.")
            return redirect('login')  # Redirect to login page or appropriate URL

        # Create a new filename with the timestamp
        original_filename = file.name
        name, ext = os.path.splitext(original_filename)
        new_filename = f"{name}_{timestamp}{ext}"
        file_content = file.read()
        file_with_timestamp = ContentFile(file_content, name=new_filename)
        
        try:
            validate_pdf(file_with_timestamp)
            # Ensure the UploadedFile model includes a user field linked to Django's User model
            uploaded_file = UploadedFile(user=request.user, file=file_with_timestamp, job_description=job_description)
            uploaded_file.save()
            messages.success(request, "File and job description uploaded successfully.")
        except ValidationError as e:
            messages.error(request, str(e))

    return redirect('home') # Redirect to a home page or success page


def analyze_resume(resume_content):
    lower_content = resume_content.lower()
    sections_found = {
        "email": bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", resume_content)),
        "skills": "Skills" in lower_content or "technical skills" in lower_content,
        "experience": "Experience" or "work experience" in lower_content,
        "education": "education" in lower_content,
    }
    return sections_found


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def identify_repetitive_words(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    repetitive_words = {word: count for word, count in word_counts.items() if count > 1}
    synonyms = {word: get_synonyms(word) for word in repetitive_words}
    return repetitive_words, synonyms

@csrf_exempt
@require_POST
def analyze_pdf(request):
    logger.debug("Received request for analyze_pdf")
    body_unicode = request.body.decode('utf-8')
    body_data = json.loads(body_unicode)
    pdf_url = body_data.get('pdf_url')

    if not pdf_url:
        logger.error("No PDF URL provided in the request")
        return JsonResponse({'error': 'No PDF URL provided'}, status=400)

    try:
        uploaded_file = UploadedFile.objects.get(file__endswith=pdf_url.split('/')[-1])
        file_path = uploaded_file.file.path
        job_description = uploaded_file.job_description
        logger.debug("File path retrieved: %s", file_path)

        page_count = count_pdf_pages(file_path)
        logger.debug("Page count: %d", page_count)

        if page_count > 1:
            messages.error(request, "The resume is more than one page")
            logger.info("The resume is more than one page")
            return JsonResponse({
                'page_count': page_count,
                'message': 'The resume should be one page.',
                'errors': []
            })

        text = extract_text_from_pdf(file_path)
        logger.debug("Extracted text from PDF")
        
        
        #ATS score logic    
        data = [text, job_description]
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data)
        match_percentage = cosine_similarity(count_matrix)[0][1] * 100
        match_percentage = round(match_percentage, 2)
        
        # Analyzing resume for sections
        resume_analysis = analyze_resume(text)

        _, matches = check_grammar(text)
        logger.debug("Checked grammar")
        
        skill_suggestions = suggest_keywords(text, job_description)
        print("skills", skill_suggestions)
        
        
        errors = [{
            'message': match.message, 
            'corrected': match.replacements[0] if match.replacements else '',
            'from_pos': match.offset,
            'to_pos': match.offset + len(match.context),
            'error_context': match.context,  # This can help in debugging
            'rule_id': match.ruleId,  # This can also be helpful to understand the type of error
        } for match in matches]



        logger.debug("Prepared error data for response")

        repetitive_words, synonyms = identify_repetitive_words(text)
        
        return JsonResponse({
            'page_count': page_count,
            'text': text,
            'errors': errors,
            'resume_analysis': resume_analysis, 
            'matchPercentage': match_percentage,
            'repetitiveWords': repetitive_words,
            'synonyms': synonyms,
            'skillSuggestions': skill_suggestions,
        })

    except UploadedFile.DoesNotExist:
        logger.error("Uploaded file does not exist: %s", pdf_url)
        return JsonResponse({'error': 'File not found'}, status=404)
    except Exception as e:
        logger.exception("Unexpected error occurred: %s", str(e))
        return JsonResponse({'error': str(e)}, status=500)
