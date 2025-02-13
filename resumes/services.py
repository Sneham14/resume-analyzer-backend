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
        - total_experience (float) in years
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

        # ✅ Remove triple backticks if present
        raw_text = raw_text.strip().strip("```json").strip("```").strip()

        # ✅ Convert raw text to JSON safely
        structured_data = json.loads(raw_text)

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
    - Supports "MMM YYYY - Present" format.
    - Converts experience months into years.
    - Ignores internships if needed.
    """
    total_experience_months = 0
    job_periods = []

    # ✅ Regex for matching date ranges (e.g., "Feb 2021 - Present")
    date_pattern = r"([A-Za-z]+)\s+(\d{4})\s*-\s*([A-Za-z]+|\d{4})"

    for exp in work_experience_list:
        job_title = exp.get("position", "").lower()
        company = exp.get("company", "")
        duration = exp.get("duration", "")

        # ✅ Ignore internships if required
        if ignore_internships and "intern" in job_title:
            continue

        match = re.search(date_pattern, duration)
        if match:
            start_month_str, start_year, end_str = match.groups()

            # ✅ Convert month names to numbers
            try:
                start_month = datetime.strptime(start_month_str, "%b").month
                start_date = datetime(int(start_year), start_month, 1)

                # ✅ Handle "Present" case
                if end_str.lower() == "present":
                    end_date = datetime.today()
                else:
                    end_month = datetime.strptime(end_str, "%b").month
                    end_year = int(end_str) if end_str.isdigit() else datetime.today().year
                    end_date = datetime(end_year, end_month, 1)

                # ✅ Append job duration
                job_periods.append((start_date, end_date))

            except Exception as e:
                print(f"❌ Error parsing dates: {duration} -> {e}")

    # ✅ Compute total experience (avoiding double counting)
    job_periods = sorted(job_periods, key=lambda x: x[0])
    merged_periods = []

    for period in job_periods:
        if not merged_periods or merged_periods[-1][1] < period[0]:  
            merged_periods.append(period)
        else:
            merged_periods[-1] = (merged_periods[-1][0], max(merged_periods[-1][1], period[1]))

    # ✅ Convert merged experience periods to total months
    for start, end in merged_periods:
        total_experience_months += (end.year - start.year) * 12 + (end.month - start.month)

    # ✅ Convert months to years (rounding off)
    total_experience_years = round(total_experience_months / 12, 2)

    return total_experience_years

def score_resume(parsed_data, min_experience):
    """Scores the resume based only on total experience years."""
    if not parsed_data:
        return {"score": 0, "error": "Invalid parsed data"}

    # ✅ Extract Work Experience
    work_experience = parsed_data.get("work_experience", [])
    total_experience = extract_experience_years(work_experience)

    print(f"🟢 Extracted Total Experience: {total_experience} years")  # Debugging print

    # ✅ Calculate Experience Score
    experience_score = 100 if total_experience >= min_experience else (total_experience / min_experience) * 100

    return {
        "score": round(experience_score, 2),
        "details": {
            "total_experience_years": total_experience,
            "required_experience": min_experience,
            "experience_match": experience_score
        }
    }
