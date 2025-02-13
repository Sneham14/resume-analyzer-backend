from django.db import models
from django.contrib.postgres.fields import JSONField  # If using PostgreSQL



class Resume(models.Model):
    file = models.FileField(upload_to="resumes/")
    extracted_text = models.TextField(blank=True, null=True)
    parsed_data = models.JSONField(blank=True, null=True)  # Store structured JSON

    def __str__(self):
        return self.file.name
    
    

class Candidate(models.Model):
    """Stores candidate profile details separately."""
    name = models.CharField(max_length=255)
    email = models.EmailField()
    phone = models.CharField(max_length=20, null=True, blank=True)
    experience = models.FloatField()  # Stores total experience in years
    job_title = models.JSONField()  # List of past job titles
    score = models.FloatField()  # Resume score

    def __str__(self):
        return f"{self.name} - {self.score}"
 
