# Generated by Django 5.0.1 on 2024-05-03 21:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("resume_revive", "0006_alter_uploadedfile_file"),
    ]

    operations = [
        migrations.AlterField(
            model_name="uploadedfile",
            name="id",
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]
