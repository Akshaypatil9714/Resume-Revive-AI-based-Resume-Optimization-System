# Generated by Django 5.0.1 on 2024-04-29 23:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("resume_revive", "0005_alter_uploadedfile_file"),
    ]

    operations = [
        migrations.AlterField(
            model_name="uploadedfile",
            name="file",
            field=models.FileField(upload_to="uploads/%Y/%m/%d"),
        ),
    ]