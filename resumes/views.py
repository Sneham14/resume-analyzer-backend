from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from .models import Resume
from .serializers import ResumeSerializer
import fitz
from .services import extract_text_from_pdf, parse_resume_with_llm, score_resume

class ResumeUploadView(APIView):
    """
    API to upload resumes, extract text, parse using LLM (GPT-4o),
    and compute score based on experience and skills.
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = ResumeSerializer(data=request.data)

        if serializer.is_valid():
            try:
                resume_instance = serializer.save()
                file_path = resume_instance.file.path  

                # ✅ Extract Text from PDF
                extracted_text = extract_text_from_pdf(file_path)

                # ✅ Parse Resume using GPT-4o
                parsed_data = parse_resume_with_llm(extracted_text)

                # ✅ Get Filters from Request (Dynamic Input)
                min_experience = int(request.data.get("min_experience", 5))  # Default: 5 years
                required_skills = request.data.get("skills", "")  # ✅ Get user-inputted skills from UI
                job_skills = [skill.strip() for skill in required_skills.split(",") if skill.strip()]  # Convert to list

                # ✅ Score Resume (Includes Experience & Skill Match)
                scoring_results = score_resume(parsed_data, min_experience, job_skills)

                # ✅ Return Response (Now Includes Skill Match)
                return Response({
                    "parsed_data": parsed_data,
                    "score": scoring_results["score"],
                    "experience_details": scoring_results["details"]
                }, status=status.HTTP_201_CREATED)

            except Exception as e:
                print(f"❌ Error processing resume: {e}")
                return Response({"error": "Failed to process resume."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
