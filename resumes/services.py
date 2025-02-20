import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import spacy
import phonenumbers
import json
from datetime import datetime
import re
from dateutil.relativedelta import relativedelta
from utils.nlp_utils import expand_abbreviations_with_ai
from rapidfuzz import fuzz, process  # ✅ Add this line


# ✅ Load .env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize GPT-4o via LangChain
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyMuPDF"""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def parse_resume_with_llm(resume_text):
    """Uses GPT-4o to extract structured resume data including detected job roles."""
    if not isinstance(resume_text, str):
        print(f"❌ Error: Expected string input, got {type(resume_text)}")
        return {"error": "Invalid resume text format"}

    prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""
        Extract structured resume data from the following text:
        {resume_text}

        Return a **valid JSON object** with:
        - name (string)
        - email (string)
        - phone (string)
        - detected_job_roles (list) based on past experience and skills
        - skills (list)
        - education (list of dicts)
        - work_experience (list of dicts with company, position, duration)
        - internships (list of dicts) extracted separately
        - summary (string)

        Ensure the response is **valid JSON** without markdown, extra text, or formatting issues.
        """
    )

    chain = prompt | llm

    try:
        response = chain.invoke({"resume_text": resume_text})

        # ✅ Extract raw text from AIMessage
        raw_text = response.content if hasattr(response, "content") else str(response)

        # ✅ Remove unwanted characters more robustly
        raw_text = raw_text.strip().strip("```json").strip("```").strip()

        # ✅ Convert raw text to JSON safely
        structured_data = json.loads(raw_text)

        # ✅ Force correct experience calculation (DO NOT use GPT's value)
        if "total_experience" in structured_data:
            del structured_data["total_experience"]  # Remove incorrect AI-generated value

        return structured_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: {e}\nResponse:\n{raw_text}")
        return {"error": "AI returned invalid JSON"}
    except Exception as e:
        print(f"❌ GPT-4o Parsing Error: {e}")
        return {"error": "AI model failed to parse resume"}




def extract_experience_years(work_experience_list, ignore_internships=True):
    """
    Extracts total experience in years from work history.
    - Supports both "MMM YYYY - Present" and "MMMM YYYY - Present" formats.
    - Prevents double counting by merging overlapping job durations.
    - Ignores internships if required.
    """
    job_periods = []

    # ✅ Regex for matching date ranges (handles "March 2019 - Present" & "Mar 2019 - Present")
    date_pattern = r"([A-Za-z]+)\s+(\d{4})\s*-\s*([A-Za-z]+|\d{4})"

    for exp in work_experience_list:
        job_title = exp.get("position", "").lower()
        duration = exp.get("duration", "")

        # ✅ Ignore internships if required
        if ignore_internships and "intern" in job_title:
            continue

        match = re.search(date_pattern, duration)
        if match:
            start_month_str, start_year, end_str = match.groups()

            try:
                # ✅ Convert month names to numbers (Handles both "March" and "Mar")
                start_month = datetime.strptime(start_month_str[:3], "%b").month
                start_date = datetime(int(start_year), start_month, 1)

                # ✅ Handle "Present" case by rounding to the current month's start
                if end_str.lower() == "present":
                    end_date = datetime(datetime.today().year, datetime.today().month, 1)
                else:
                    end_month = datetime.strptime(end_str[:3], "%b").month
                    # ✅ Extract year from duration
                    if re.search(r'\d{4}', duration):  # Check if a year exists in duration
                        end_year = int(re.search(r'\d{4}', duration).group())  # Extract last year mentioned
                    else:
                        end_year = start_year  # Default: assume same year as start                   
                    end_date = datetime(end_year, end_month, 1)


                # ✅ Store job periods for overlap handling
                job_periods.append((start_date, end_date))

            except Exception as e:
                print(f"❌ Error parsing dates: {duration} -> {e}")

    # ✅ Merge overlapping job periods
    job_periods = sorted(job_periods, key=lambda x: x[0])  # Sort by start date
    merged_periods = []

    for period in job_periods:
        if not merged_periods or merged_periods[-1][1] < period[0]:  
            merged_periods.append(period)
        else:
            # ✅ Merge overlapping periods (Take max end date)
            merged_periods[-1] = (merged_periods[-1][0], max(merged_periods[-1][1], period[1]))

    # ✅ Convert merged periods to total experience in months
    total_experience_months = 0
    for start, end in merged_periods:
        total_experience_months += (end.year - start.year) * 12 + (end.month - start.month)

    # ✅ Convert months to years (Rounded to 2 decimal places)
    total_experience_years = round(total_experience_months / 12, 2)
    print(f"Extracted job periods (before merging): {job_periods}")
    print(f"Merged job periods (after overlap handling): {merged_periods}")
    print(f"Total experience months calculated: {total_experience_months}")
    print(f"Total experience years calculated: {total_experience_years}")

    return total_experience_years

from rapidfuzz import fuzz, process

def keyword_match(resume_skills, job_skills, threshold=80):
    """
    Matches resume skills against job-required skills dynamically.
    - Expands abbreviations dynamically using AI.
    - Uses fuzzy matching for skill variations.
    - Ensures missing skills are properly updated.
    """
    if not resume_skills or not job_skills:
        return 0, list(job_skills)  # No match if no skills provided

    # ✅ Normalize and expand abbreviations for both resume and job skills
    resume_skills = set(expand_abbreviations_with_ai(skill.lower().strip()) for skill in resume_skills)
    job_skills = set(expand_abbreviations_with_ai(skill.lower().strip()) for skill in job_skills)

    # ✅ Debugging Logs
    print(f"🔍 Resume Skills (Processed): {resume_skills}")
    print(f"🔍 Job Skills (Processed): {job_skills}")

    # ✅ Find exact matches first
    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - matched_skills  # ✅ Correct missing skills logic

    # ✅ Debug Exact Matches
    print(f"✅ Matched Skills (Exact): {matched_skills}")
    print(f"❌ Missing Skills Before Fuzzy Matching: {missing_skills}")

    # ✅ Use fuzzy matching for partial or similar matches
    fuzzy_matched = set()


    for job_skill in missing_skills.copy():
        match_result = process.extractOne(job_skill, list(resume_skills), scorer=fuzz.token_sort_ratio)

        # ✅ Debugging Output
        print(f"🔍 Debug: Match result for '{job_skill}': {match_result}")

        # ✅ Handle None Case Properly Before Unpacking
        if match_result is None:
            print(f"❗ No match found for: {job_skill}")
            continue  # Skip to the next skill

        try:
            # ✅ Safe Unpacking Using Try-Except
            best_match, score, *_ = match_result  # Unpack only first two values, ignore extra
        except ValueError:
            print(f"❗ Unexpected match format for '{job_skill}': {match_result}")
            continue  # Skip if not a valid match

        print(f"🔎 Fuzzy Match: '{job_skill}' ↔ '{best_match}' (Score: {score})")

        if score >= threshold:  # ✅ Only accept confident matches
            matched_skills.add(best_match)
            missing_skills.discard(job_skill)  # ✅ Only remove if confidently matched

        # ✅ Ensure missing skills are correctly updated
    missing_skills = job_skills - matched_skills

    # ✅ Final Debug Logs
    print(f"✅ Final Matched Skills: {matched_skills}")
    print(f"❌ Final Missing Skills (After Fuzzy Matching): {missing_skills}")

    # ✅ Calculate match percentage
    match_percentage = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0

    return round(match_percentage, 2), list(missing_skills)

def score_resume(parsed_data, min_experience, job_skills):
    """
    Scores the resume based on experience and required skills.
    - Uses keyword_match() for skill matching.
    - Penalizes candidates with 0% skill match.
    """
    if not parsed_data:
        return {"score": 0, "error": "Invalid parsed data"}

    # ✅ Extract Work Experience
    work_experience = parsed_data.get("work_experience", [])
    total_experience = extract_experience_years(work_experience)

    # ✅ Extract Candidate's Skills
    candidate_skills = parsed_data.get("skills", [])

    # ✅ Calculate Experience Score
    experience_score = 100 if total_experience >= min_experience else (total_experience / min_experience) * 100

    # ✅ Calculate Skills Match Score
    skill_match_score, missing_skills = keyword_match(candidate_skills, job_skills)

    # ✅ Apply a Minimum Skill Match Threshold
    if skill_match_score == 0:
        final_score = 0  # ❌ Disqualify if no skill match
    else:
        final_score = round(0.8 * experience_score + 0.2 * skill_match_score, 2)  # Weighted Score

    return {
        "score": final_score,
        "details": {
            "total_experience_years": total_experience,
            "required_experience": min_experience,
            "experience_match": experience_score,
            "skill_match": skill_match_score,
            "missing_skills": missing_skills
        }
    }
