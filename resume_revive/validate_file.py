from django.core.exceptions import ValidationError

def validate_pdf(file):
    # Validate file extension
    if not file.name.endswith('.pdf'):
        raise ValidationError('Only PDF files are allowed.')

    # Validate file size (e.g., 10MB limit)
    if file.size > 1000000:
        raise ValidationError('File size must not exceed 1MB.')