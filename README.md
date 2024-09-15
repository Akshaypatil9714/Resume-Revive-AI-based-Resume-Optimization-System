# Resume-Revive

Resume Revive is an advanced resume analysis and optimization tool designed to help job seekers improve their resumes. This system leverages the power of Django for web application management and the BERT (Bidirectional Encoder Representations from Transformers) model for deep natural language processing. It provides users with actionable insights by analyzing resumes against job descriptions, suggesting improvements, identifying key skills, and enhancing overall resume effectiveness to increase the chances of job acquisition.

## Features
- Resume Upload: Users can upload their resumes in PDF format.

- Job Description Matching: Analyzes resumes against provided job descriptions to suggest essential keywords and skills.

- Resume Optimization Suggestions: Offers synonyms for repetitive words, checks grammar, and formats resumes.

- User Authentication: Manages user sessions for personalized experience and security.

## Installation

**Prerequisites**

- Python 3.8 or higher

- Django 3.2

- pip and virtualenv

- Google Cloud SDK

## Setup

**Create and activate a virtual environment**

```bash
python3 -m venv env
```
```bash
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
**Install the requirements**
```bash
  pip install -r requirements.txt
```
**Configure Google Cloud SDK
- Set up Google Cloud SDK by following Google Cloud's instructions.
- Set your project ID:
```bash
  gcloud config set project resume-revive
```
**Run migrations**
```bash
python manage.py migrate
```
**Start the server**
```bash
python manage.py runserver
```
## Usage

After installation, visit http://127.0.0.1:8000/ in your web browser to start using the Resume Revive application. Register and log in to upload your resume and a job description, and receive detailed feedback and suggestions.

