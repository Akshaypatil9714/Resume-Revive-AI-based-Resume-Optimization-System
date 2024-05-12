from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class UploadedFile(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    file = models.FileField(upload_to='uploads/%Y/%m/%d')
    job_description = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    