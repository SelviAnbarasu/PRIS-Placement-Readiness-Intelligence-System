import json
import os
import re
import tempfile
from io import BytesIO

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether
)
from pypdf import PdfReader
from google import genai

# Local RAG knowledge retriever used to ground learning, roadmap and
# interview recommendations in the curated PRIS knowledge base.
try:
    from rag_knowledge_base import build_rag_context
    RAG_AVAILABLE = True
    RAG_IMPORT_ERROR = None
except Exception as exc:
    build_rag_context = None
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = str(exc)


# ============================================================
# PRIS - Placement Readiness Intelligence System
# SAFE UI VERSION
# No HTML content rendering is used, so raw <div>/<h1> tags
# cannot appear in the dashboard.
# ============================================================

st.set_page_config(
    page_title="PRIS | Placement Readiness Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "student_placement_model.pkl"


# ============================================================
# CSS ONLY
# ============================================================

st.markdown(
"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(45, 117, 220, 0.16), transparent 30%),
        radial-gradient(circle at 90% 0%, rgba(0, 210, 255, 0.09), transparent 25%),
        #06111f;
    color: #f5f7fb;
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c172b 0%, #081221 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #eaf0f8;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111e33, #0b1729);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
}

[data-testid="stMetricLabel"] {
    color: #9aacc4 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(17,30,51,0.95), rgba(10,24,42,0.95));
    border: 1px solid rgba(91, 222, 255, 0.12);
    border-radius: 20px;
}

.stButton > button {
    min-height: 48px;
    border-radius: 12px;
    font-weight: 800;
}

.stDownloadButton > button {
    min-height: 48px;
    border-radius: 12px;
    font-weight: 800;
}

.pris-small {
    color: #8fa3bd;
    font-size: 14px;
}

.pris-blue {
    color: #5be4ff;
}

.pris-green {
    color: #64e7b0;
}

.pris-orange {
    color: #ffc46b;
}

.pris-red {
    color: #ff8190;
}

.pris-footer {
    text-align: center;
    color: #7e91aa;
    padding: 35px 10px 10px;
    font-size: 13px;
}
</style>""",
unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_ml_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"{MODEL_PATH} not found."

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, str(exc)


ml_model, ml_error = load_ml_model()


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low=0, high=100):
    try:
        value = float(value)
    except Exception:
        value = low
    return max(low, min(high, value))


def safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except Exception:
        return default


def clean_json(text):
    if not text:
        return {}

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.I).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    return {}


def create_client(api_key):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# ============================================================
# DEMO DATA
# ============================================================

def demo_result():
    return {
        "candidate": {
            "name": "Demo Student",
            "education": "B.Tech Computer Science",
            "degree": "B.Tech",
            "branch": "CSE",
            "cgpa": 8.1,
            "internships": 1,
            "projects_count": 3,
            "certifications": 2,
            "backlogs": 0,
        },
        "job": {
            "role": "Data Analyst",
            "company": "Target Company",
            "experience": "Fresher / Entry Level",
        },
        "match": {
            "job_match_percentage": 78,
            "skill_match_percentage": 78,
            "keyword_match_score": 74,
            "critical_skill_match_percentage": 70,
            "matched_skills": [
                "Python", "SQL", "Power BI", "Machine Learning", "Data Analysis"
            ],
            "missing_skills": [
                "Advanced Excel", "Statistics", "DAX"
            ],
            "critical_missing_skills": [
                "Advanced Excel", "DAX"
            ],
        },
        "analysis": {
            "summary": (
                "The candidate has a strong foundation in Python, SQL, Power BI "
                "and Machine Learning. The main improvement areas for this role "
                "are advanced Excel, statistics and stronger job-specific keywords."
            ),
            "strengths": [
                "Strong Python and SQL foundation",
                "Practical data and AI project exposure",
                "Power BI knowledge",
                "Relevant academic background",
            ],
            "improvement_areas": [
                "Advanced Excel",
                "Statistics",
                "DAX",
                "Role-specific resume keywords",
            ],
            "project_relevance_score": 82,
            "certification_relevance_score": 76,
            "internship_relevance_score": 72,
            "resume_completeness_score": 84,
            "role_category_match_score": 80,
        },
        "ml_profile": {
            "branch": "CSE",
            "college_tier": "Tier-2",
            "cgpa": 8.1,
            "backlogs": 0,
            "coding_skills": 7,
            "dsa_score": 6,
            "aptitude_score": 72,
            "communication_skills": 7,
            "ml_knowledge": 7,
            "system_design": 4,
            "internships": 1,
            "projects_count": 3,
            "certifications": 2,
            "hackathons": 1,
            "open_source_contributions": 0,
            "extracurriculars": 1,
        },
        "roadmap": {
            "seven_day": [
                {"day": "DAY 1", "title": "Advanced SQL", "details": "Joins, subqueries and window functions."},
                {"day": "DAY 2", "title": "Excel", "details": "Pivot tables, lookup functions and data cleaning."},
                {"day": "DAY 3", "title": "Statistics", "details": "Mean, variance, distributions and hypothesis testing."},
                {"day": "DAY 4", "title": "Power BI", "details": "Data modelling and DAX fundamentals."},
                {"day": "DAY 5", "title": "Portfolio Project", "details": "Build one end-to-end analytics project."},
                {"day": "DAY 6", "title": "Resume Optimization", "details": "Improve role-specific keywords and project bullets."},
                {"day": "DAY 7", "title": "Interview Practice", "details": "Practice SQL, analytics and project questions."},
            ],
            "thirty_day": [
                {"week": "WEEK 1", "title": "SQL + Excel", "details": "Advanced SQL, joins, window functions and Excel dashboards."},
                {"week": "WEEK 2", "title": "Statistics", "details": "Statistics, probability and analytical reasoning."},
                {"week": "WEEK 3", "title": "Power BI + Portfolio", "details": "DAX, modelling and one polished portfolio project."},
                {"week": "WEEK 4", "title": "Interview Preparation", "details": "Mock interviews, resume refinement and job applications."},
            ],
        },
        "interview_questions": [
            "Explain the difference between WHERE and HAVING in SQL.",
            "How would you handle missing values in a dataset?",
            "Explain a data analytics project from your resume.",
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "How would you validate a dashboard before sharing it?",
        ],
        "interview_focus": [
            "SQL joins and window functions",
            "Power BI and DAX",
            "Statistics fundamentals",
            "Project explanation",
        ],
        "rag": {
            "enabled": bool(RAG_AVAILABLE),
            "retrieved_sources": [],
            "skill_gap_sources": [],
            "context_used_for_gemini": False,
            "skill_gap_context_retrieved": False,
            "demo_note": "Demo mode does not call Gemini or perform live RAG retrieval.",
        },
    }


# ============================================================
# LOCAL PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(resume_file):
    """Extract text locally so Gemini receives the actual resume content explicitly.

    This is used as the primary extraction source. If the PDF is image-only and
    contains no extractable text, the Gemini document input is used as fallback.
    """
    try:
        reader = PdfReader(resume_file)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def _branch_supported_alias(branch, resume_text):
    """Validate a branch against the resume instead of trusting an LLM guess."""
    if not branch or not resume_text:
        return False

    b = str(branch).strip().lower()
    aliases = {
        "cse": ["cse", "computer science", "computer science engineering"],
        "it": ["it", "information technology"],
        "ece": ["ece", "electronics and communication", "electronics & communication"],
        "ee": ["ee", "electrical engineering", "electrical & electronics", "eee"],
        "me": ["me", "mechanical engineering", "mechanical"],
        "ce": ["ce", "civil engineering", "civil"],
        "chemical": ["chemical engineering", "chemical"],
    }
    candidates = aliases.get(b, [b])
    text = resume_text.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(x) + r"(?![a-z])", text) for x in candidates)


def _extract_education_branch(resume_text):
    """Extract the academic subject/branch from a degree line without swallowing the college name."""
    if not resume_text:
        return None

    lines = [re.sub(r"\s+", " ", x).strip(" -•") for x in resume_text.splitlines()]
    degree_patterns = [
        r"(?:B\.Sc\.?|BSc|B\.Com\.?|BCom|BCA|BBA|B\.Tech\.?|BTech|B\.E\.?|BE|M\.Sc\.?|MSc|M\.Com\.?|MCom|MCA|MBA)\s+(.+)$",
    ]

    for line in lines:
        for pattern in degree_patterns:
            m = re.search(pattern, line, re.I)
            if not m:
                continue

            remainder = m.group(1).strip()

            # Common resume formats: "B.Com General - Sri Ram College..."
            # or "B.Sc. Mathematics, XYZ University". Stop before the
            # institution/date instead of treating it as part of the branch.
            remainder = re.split(
                r"\s+(?:-|–|—|at|from)\s+|\s*,\s*(?=[A-Z][A-Za-z .&'-]*(?:College|University|Institute|School))|\s+(?=[A-Z][A-Za-z .&'-]*(?:College|University|Institute|School)\b)|\s+(?:20\d{2}|19\d{2})\b",
                remainder,
                maxsplit=1,
                flags=re.I,
            )[0].strip(" ,.-")

            if remainder and len(remainder) <= 60:
                return remainder
    return None


def enrich_obvious_candidate_fields(result, resume_text):
    """Use deterministic resume evidence to prevent hallucinated candidate facts."""
    if not resume_text:
        return result

    candidate = result.setdefault("candidate", {})

    # Explicit CGPA/GPA value. Do not convert percentages or infer a CGPA.
    if candidate.get("cgpa") in (None, ""):
        m = re.search(r"\b(?:CGPA|C\.G\.P\.A\.|GPA)\s*[:\-]?\s*(\d+(?:\.\d+)?)\b", resume_text, re.I)
        if m:
            value = float(m.group(1))
            if 0 <= value <= 10:
                candidate["cgpa"] = value

    # If Gemini proposes a branch that is not supported by the actual resume,
    # replace it with an explicitly visible degree subject when available.
    branch = candidate.get("branch")
    if branch and not _branch_supported_alias(branch, resume_text):
        extracted_branch = _extract_education_branch(resume_text)
        candidate["branch"] = extracted_branch

    # Only accept explicit numeric count statements from the resume itself.
    # Do not let a job description or an LLM default create a count.
    explicit_patterns = {
        "internships": r"\b(\d+)\s+(?:internship|internships)\b",
        "projects_count": r"\b(\d+)\s+(?:projects|project)\b",
        "certifications": r"\b(\d+)\s+(?:certifications|certificates)\b",
        "backlogs": r"\b(\d+)\s+(?:backlogs|backlog)\b",
    }
    for field, pattern in explicit_patterns.items():
        m = re.search(pattern, resume_text, re.I)
        if m:
            candidate[field] = int(m.group(1))

    # If the resume has no explicit evidence for these fields, do not retain a
    # hallucinated zero from Gemini. Missing stays missing.
    evidence_words = {
        "internships": r"\binternships?\b",
        "projects_count": r"\b(?:academic|personal|professional)?\s*projects?\b",
        "certifications": r"\bcertifications?\b|\bcertificates?\b",
        "backlogs": r"\bbacklogs?\b",
    }
    for field, pattern in evidence_words.items():
        if not re.search(pattern, resume_text, re.I) and candidate.get(field) in (0, "0"):
            candidate[field] = None

    # Keep the ML profile synchronized with the factual candidate fields.
    profile = result.setdefault("ml_profile", {})
    for field in ("branch", "cgpa", "internships", "projects_count", "certifications", "backlogs"):
        value = candidate.get(field)
        profile[field] = value if value not in ("",) else None

    return result


# ============================================================
# GEMINI
# ============================================================

def analyze_resume_and_jd(resume_file, jd_text, api_key, model_name):
    client = create_client(api_key)

    if client is None:
        raise ValueError("Gemini API key is required.")

    temp_path = None
    resume_text = ""

    try:
        resume_text = extract_resume_text(resume_file)
    except Exception:
        resume_text = ""

    resume_text_for_prompt = resume_text if resume_text else "[No text could be extracted locally from this PDF. Use the uploaded PDF document itself.]"

    # RAG retrieval happens before the Gemini generation call. The retrieved
    # knowledge is used as grounding context for learning, roadmap and
    # interview recommendations. This keeps RAG local and deterministic and
    # does not require a second Gemini request.
    rag_context = ""
    rag_sources = []
    if RAG_AVAILABLE and build_rag_context is not None:
        try:
            rag_query_text = jd_text
            if resume_text:
                rag_query_text = f"{jd_text}\\n\\nResume context:\\n{resume_text[:6000]}"
            rag_context, rag_sources = build_rag_context(
                jd_text=rag_query_text,
                top_k=4,
            )
        except Exception as exc:
            rag_context = ""
            rag_sources = []

    rag_prompt_context = rag_context if rag_context else "[No RAG context was retrieved for this analysis.]"

    prompt = f"""
You are the AI engine for a Placement Readiness Intelligence System.

IMPORTANT: The following text was extracted directly from the uploaded resume PDF.
Use this resume text as the primary factual source. Read it completely before
creating the JSON. Do not use the job description to invent candidate facts.

===== RESUME TEXT START =====
{resume_text_for_prompt}
===== RESUME TEXT END =====

Analyze the uploaded resume PDF against this Job Description:

{jd_text}

===== RAG KNOWLEDGE CONTEXT START =====
Use the retrieved knowledge below to ground the learning roadmap, improvement
recommendations and interview preparation. Do not invent facts from the context.
Do not use RAG content to invent candidate facts. Candidate facts must come only
from the resume.

{rag_prompt_context}
===== RAG KNOWLEDGE CONTEXT END =====

Return ONLY valid JSON.

Be conservative. Do not invent unsupported facts.

Use this exact structure:

{{
  "candidate": {{
    "name": "",
    "education": "",
    "degree": "",
    "branch": null,
    "cgpa": null,
    "internships": null,
    "projects_count": null,
    "certifications": null,
    "backlogs": null
  }},
  "job": {{
    "role": "",
    "company": "",
    "experience": ""
  }},
  "match": {{
    "job_match_percentage": 0,
    "skill_match_percentage": 0,
    "keyword_match_score": 0,
    "critical_skill_match_percentage": 0,
    "matched_skills": [],
    "missing_skills": [],
    "critical_missing_skills": []
  }},
  "analysis": {{
    "summary": "",
    "strengths": [],
    "improvement_areas": [],
    "project_relevance_score": 0,
    "certification_relevance_score": 0,
    "internship_relevance_score": 0,
    "resume_completeness_score": 0,
    "role_category_match_score": 0
  }},
  "ml_profile": {{
    "branch": null,
    "college_tier": null,
    "cgpa": null,
    "backlogs": null,
    "coding_skills": null,
    "dsa_score": null,
    "aptitude_score": null,
    "communication_skills": null,
    "ml_knowledge": null,
    "system_design": null,
    "internships": null,
    "projects_count": null,
    "certifications": null,
    "hackathons": null,
    "open_source_contributions": null,
    "extracurriculars": null
  }},
  "roadmap": {{
    "seven_day": [
      {{"day": "DAY 1", "title": "", "details": ""}},
      {{"day": "DAY 2", "title": "", "details": ""}},
      {{"day": "DAY 3", "title": "", "details": ""}},
      {{"day": "DAY 4", "title": "", "details": ""}},
      {{"day": "DAY 5", "title": "", "details": ""}},
      {{"day": "DAY 6", "title": "", "details": ""}},
      {{"day": "DAY 7", "title": "", "details": ""}}
    ],
    "thirty_day": [
      {{"week": "WEEK 1", "title": "", "details": ""}},
      {{"week": "WEEK 2", "title": "", "details": ""}},
      {{"week": "WEEK 3", "title": "", "details": ""}},
      {{"week": "WEEK 4", "title": "", "details": ""}}
    ]
  }},
  "interview_questions": [],
  "interview_focus": []
}}

RESUME EXTRACTION CHECKLIST (IMPORTANT):
- Read the ENTIRE uploaded resume PDF before producing the JSON. Do not rely only on the summary or first page.
- Extract factual candidate information from the resume first, especially the Candidate section below.
- If the resume contains a CGPA, percentage, internship entry, project entry, certification entry, or backlog count, copy the factual value into the candidate object.
- Count clearly listed projects one-by-one under Projects/Academic Projects/Personal Projects/Projects. Put that count in candidate.projects_count.
- Count clearly listed internships one-by-one under Internship/Internships/Experience only when the entry is actually an internship. Put that count in candidate.internships.
- Count clearly listed certifications one-by-one under Certifications/Certificates only when they are actual certifications. Put that count in candidate.certifications.
- Count hackathons only when a hackathon/competition participation is explicitly listed.
- Count open-source contributions only when explicit contributions are listed.
- Count extracurricular activities only when explicit extracurricular activities are listed.
- If a section exists but contains no countable item, use 0. If the resume does not contain that information at all, use null.
- Never replace missing information with a generic default such as 0, CSE, Tier-2, 5, or 60 merely to satisfy the schema.
- Do not treat coursework, skills, tools, job-description requirements, or generic statements as projects/certifications/internships.
- Do not infer a college tier from college name unless the resume explicitly states the tier.

ML profile rules:
- First extract the candidate facts in the candidate section. Then COPY those same facts into the corresponding ml_profile fields. Do not leave the ml_profile field null when the candidate field contains a real value.
- Extract values from the resume whenever there is evidence.
- NEVER invent a value just to fill a field.
- Use null when a value is not available or cannot be reasonably inferred from the resume.
- Do not use placeholder defaults such as CSE, Tier-2, 0, 5, or 60 for missing information.
- For example: candidate.cgpa must equal ml_profile.cgpa; candidate.internships must equal ml_profile.internships; candidate.projects_count must equal ml_profile.projects_count; candidate.certifications must equal ml_profile.certifications; candidate.backlogs must equal ml_profile.backlogs; candidate.branch must equal ml_profile.branch.
- branch must preserve the candidate's actual academic branch exactly when stated. Do not convert an unsupported branch such as Mathematics into CSE.
- college_tier must be null unless the resume explicitly states the tier or provides clear evidence that supports it.
- coding_skills, dsa_score, communication_skills, ml_knowledge, system_design: 0-10. These may be estimated ONLY from explicit resume evidence such as listed skills, projects, coursework, certifications or achievements; otherwise use null.
- aptitude_score: 0-100. Use an actual aptitude/test score if stated; otherwise null.
- cgpa: 0-10. Use the actual CGPA/percentage conversion only when the resume provides enough information; otherwise null.
- internships/projects/certifications/hackathons/open_source_contributions/extracurriculars/backlogs: non-negative integers when countable from the resume; otherwise null.
- college_tier: Tier-1, Tier-2, Tier-3, or null.
- For counts, count only clearly identifiable items. Do not count a generic mention as an item.
"""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resume_file.getbuffer())
            temp_path = tmp.name

        # Prefer explicit resume text because it makes factual extraction much
        # more reliable than asking the model to infer structured fields from
        # a document attachment. If the PDF is image-only, fall back to the
        # uploaded document input.
        if resume_text:
            interaction = client.interactions.create(
                model=model_name,
                input=[{"type": "text", "text": prompt}],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                },
            )
        else:
            uploaded_pdf = client.files.upload(file=temp_path)
            interaction = client.interactions.create(
                model=model_name,
                input=[
                    {
                        "type": "document",
                        "uri": uploaded_pdf.uri,
                        "mime_type": uploaded_pdf.mime_type or "application/pdf",
                    },
                    {"type": "text", "text": prompt},
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                },
            )

        result = clean_json(interaction.output_text)

        if not result:
            raise ValueError("Gemini returned invalid JSON.")

        result = enrich_obvious_candidate_fields(result, resume_text)

        # A second retrieval pass is local-only: once Gemini has identified the
        # missing skills, retrieve the most relevant knowledge-base documents for
        # those gaps. No additional LLM/API request is made.
        skill_gap_sources = []
        skill_gap_context = ""
        if RAG_AVAILABLE and build_rag_context is not None:
            try:
                missing_skills = (result.get("match", {}) or {}).get("missing_skills", []) or []
                critical_skills = (result.get("match", {}) or {}).get("critical_missing_skills", []) or []
                target_role = (result.get("job", {}) or {}).get("role", "") or ""
                gap_query = list(dict.fromkeys([str(x) for x in missing_skills + critical_skills if str(x).strip()]))
                skill_gap_context, skill_gap_sources = build_rag_context(
                    jd_text=jd_text,
                    missing_skills=gap_query,
                    target_role=target_role,
                    top_k=4,
                )
            except Exception:
                skill_gap_context = ""
                skill_gap_sources = []

        result["rag"] = {
            "enabled": bool(RAG_AVAILABLE),
            "retrieved_sources": rag_sources,
            "skill_gap_sources": skill_gap_sources,
            "context_used_for_gemini": bool(rag_context),
            "skill_gap_context_retrieved": bool(skill_gap_context),
        }
        return result

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# ML
# ============================================================

def _optional_number(value, low=None, high=None):
    """Return a numeric value or NaN when the source did not provide one."""
    if value is None or value == "":
        return np.nan
    try:
        number = float(value)
    except Exception:
        return np.nan
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _optional_count(value):
    """Return a non-negative integer or NaN when the source did not provide one."""
    if value is None or value == "":
        return np.nan
    try:
        return max(0, int(round(float(value))))
    except Exception:
        return np.nan


def build_ml_profile(result):
    """Build the ML profile from the most reliable extracted candidate facts.

    Direct resume facts in the candidate section take priority for fields such as
    branch, CGPA, internships, projects, certifications and backlogs. This prevents
    a null/placeholder value in ml_profile from overriding a real extracted value.
    """
    candidate = result.get("candidate", {}) or {}
    profile = result.get("ml_profile", {}) or {}
    merged = dict(profile)

    direct_fields = [
        "branch",
        "cgpa",
        "internships",
        "projects_count",
        "certifications",
        "backlogs",
    ]

    for field in direct_fields:
        candidate_value = candidate.get(field)
        if candidate_value is not None and candidate_value != "":
            merged[field] = candidate_value
        elif merged.get(field) is None or merged.get(field) == "":
            merged[field] = None

    return merged


def run_ml_prediction(result):
    if ml_model is None:
        return None

    p = build_ml_profile(result)

    supported_branches = ["CSE", "IT", "ECE", "EE", "ME", "CE", "Chemical"]
    raw_branch = p.get("branch")
    branch = str(raw_branch).strip() if raw_branch not in (None, "") else None
    branch_supported = branch in supported_branches

    raw_tier = p.get("college_tier")
    tier = str(raw_tier).strip() if raw_tier not in (None, "") else None
    tier_supported = tier in ["Tier-1", "Tier-2", "Tier-3"]

    # Unknown/missing values are deliberately kept as missing instead of
    # silently changing them to CSE, Tier-2, or arbitrary numeric defaults.
    row = pd.DataFrame([{
        "branch": branch if branch is not None else np.nan,
        "college_tier": tier if tier_supported else np.nan,
        "cgpa": _optional_number(p.get("cgpa"), 0, 10),
        "backlogs": _optional_count(p.get("backlogs")),
        "coding_skills": _optional_number(p.get("coding_skills"), 0, 10),
        "dsa_score": _optional_number(p.get("dsa_score"), 0, 10),
        "aptitude_score": _optional_number(p.get("aptitude_score"), 0, 100),
        "communication_skills": _optional_number(p.get("communication_skills"), 0, 10),
        "ml_knowledge": _optional_number(p.get("ml_knowledge"), 0, 10),
        "system_design": _optional_number(p.get("system_design"), 0, 10),
        "internships": _optional_count(p.get("internships")),
        "projects_count": _optional_count(p.get("projects_count")),
        "certifications": _optional_count(p.get("certifications")),
        "hackathons": _optional_count(p.get("hackathons")),
        "open_source_contributions": _optional_count(p.get("open_source_contributions")),
        "extracurriculars": _optional_count(p.get("extracurriculars")),
    }])

    try:
        pred = int(ml_model.predict(row)[0])
        not_placed_probability = None
        placed_probability = None
        predicted_class_probability = None

        if hasattr(ml_model, "predict_proba"):
            probabilities = ml_model.predict_proba(row)[0]
            not_placed_probability = float(probabilities[0])
            placed_probability = float(probabilities[1])
            predicted_class_probability = float(probabilities[pred])

        return {
            "prediction": pred,
            # Keep this as predicted-class probability for diagnostics.
            "probability": predicted_class_probability,
            # This is the probability used by the PRIS readiness score.
            "placed_probability": placed_probability,
            "not_placed_probability": not_placed_probability,
            "features": row.iloc[0].to_dict(),
            "feature_sources": {
                "branch": "candidate/resume" if p.get("branch") not in (None, "") else "missing",
                "cgpa": "candidate/resume" if p.get("cgpa") not in (None, "") else "missing",
                "internships": "candidate/resume" if p.get("internships") not in (None, "") else "missing",
                "projects_count": "candidate/resume" if p.get("projects_count") not in (None, "") else "missing",
                "certifications": "candidate/resume" if p.get("certifications") not in (None, "") else "missing",
                "backlogs": "candidate/resume" if p.get("backlogs") not in (None, "") else "missing",
            },
            "branch_supported": branch_supported,
            "branch_value": branch,
            "tier_supported": tier_supported,
            "tier_value": tier,
        }

    except Exception as exc:
        return {"error": str(exc), "features": row.iloc[0].to_dict()}


def calculate_readiness(result, ml_result):
    m = result.get("match", {})
    a = result.get("analysis", {})

    job = clamp(m.get("job_match_percentage", 0))
    skill = clamp(m.get("skill_match_percentage", 0))
    keyword = clamp(m.get("keyword_match_score", 0))
    project = clamp(a.get("project_relevance_score", 0))
    resume = clamp(a.get("resume_completeness_score", 0))

    placed_probability = ml_result.get("placed_probability") if ml_result else None

    if ml_result and not ml_result.get("error") and placed_probability is not None:
        ml = placed_probability * 100
        score = (
            0.40 * job +
            0.20 * skill +
            0.10 * keyword +
            0.10 * project +
            0.05 * resume +
            0.15 * ml
        )
    else:
        score = (
            0.45 * job +
            0.25 * skill +
            0.10 * keyword +
            0.10 * project +
            0.10 * resume
        )

    return round(clamp(score), 1)


def readiness_label(score):
    if score >= 80:
        return "Highly Ready"
    if score >= 65:
        return "Moderately Ready"
    if score >= 50:
        return "Needs Improvement"
    return "Not Ready Yet"


# ============================================================
# PDF
# ============================================================

def create_pdf_report(result, readiness, ml_result):
    """Create the complete PRIS PDF report.

    Includes the dashboard's core analysis plus the 7-day roadmap, 30-day
    improvement plan, and AI interview preparation requested in the project brief.
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title="PRIS Placement Readiness Report",
        author="Placement Readiness Intelligence System",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PRISTitle", parent=styles["Title"], fontSize=20,
        leading=24, alignment=TA_CENTER, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="PRISSubTitle", parent=styles["Normal"], fontSize=9,
        leading=13, alignment=TA_CENTER, spaceAfter=16
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontSize=13,
        leading=16, spaceBefore=10, spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontSize=9,
        leading=13, spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontSize=8,
        leading=11, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name="CardTitle", parent=styles["Heading3"], fontSize=10,
        leading=13, spaceAfter=4
    ))

    story = []

    def safe(value, fallback="N/A"):
        if value is None or str(value).strip() == "":
            return fallback
        return str(value)

    def para(text, style="BodySmall"):
        # Escape only the common XML-sensitive characters because Paragraph
        # interprets them as markup.
        text = safe(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(text, styles[style])

    def bullet_list(items):
        flow = []
        items = items or []
        if not items:
            flow.append(para("None available.", "BodySmall"))
        else:
            for item in items:
                flow.append(para("• " + safe(item), "BodySmall"))
        return flow

    c = result.get("candidate", {}) or {}
    j = result.get("job", {}) or {}
    m = result.get("match", {}) or {}
    a = result.get("analysis", {}) or {}
    roadmap = result.get("roadmap", {}) or {}
    seven_day = roadmap.get("seven_day", []) or []
    thirty_day = roadmap.get("thirty_day", []) or []
    questions = result.get("interview_questions", []) or []
    interview_focus = result.get("interview_focus", []) or []

    # Header
    story.append(Paragraph("PLACEMENT READINESS INTELLIGENCE SYSTEM", styles["PRISTitle"]))
    story.append(Paragraph(
        "AI Resume + Job Description Placement Analyzer",
        styles["PRISSubTitle"]
    ))

    # Candidate + readiness summary
    candidate_data = [
        [para("Candidate", "CardTitle"), para(safe(c.get("name")), "CardTitle")],
        [para("Education", "Small"), para(safe(c.get("education")), "Small")],
        [para("Branch", "Small"), para(safe(c.get("branch")), "Small")],
        [para("CGPA", "Small"), para(safe(c.get("cgpa")), "Small")],
        [para("Target Role", "Small"), para(safe(j.get("role")), "Small")],
    ]
    readiness_data = [
        [para("Readiness Score", "CardTitle"), para(f"{readiness}/100", "CardTitle")],
        [para("Readiness Level", "Small"), para(readiness_label(readiness), "Small")],
        [para("Job Match", "Small"), para(f"{safe(m.get('job_match_percentage'), '0')}%", "Small")],
        [para("Skill Match", "Small"), para(f"{safe(m.get('skill_match_percentage'), '0')}%", "Small")],
    ]
    if ml_result and not ml_result.get("error"):
        placed_p = ml_result.get("placed_probability")
        if placed_p is not None:
            readiness_data.append([
                para("ML Placed Probability", "Small"),
                para(f"{placed_p * 100:.1f}%", "Small")
            ])

    summary_table = Table([candidate_data, readiness_data], colWidths=[250, 250])
    summary_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # AI summary and match
    story.append(Paragraph("AI Analysis", styles["Section"]))
    story.extend(bullet_list([a.get("summary", "No summary available.")]))

    story.append(Paragraph("Matched Skills", styles["Section"]))
    story.extend(bullet_list(m.get("matched_skills", [])))

    story.append(Paragraph("Missing / Priority Skills", styles["Section"]))
    story.extend(bullet_list(m.get("missing_skills", [])))

    # RAG evidence used to ground the recommendations.
    rag = result.get("rag", {}) or {}
    rag_sources_pdf = rag.get("skill_gap_sources", []) or rag.get("retrieved_sources", []) or []
    if rag.get("enabled"):
        story.append(Paragraph("RAG Knowledge Retrieval", styles["Section"]))
        if rag_sources_pdf:
            for item in rag_sources_pdf:
                title = safe(item.get("title"), "Knowledge source")
                score = item.get("score")
                score_text = f" — similarity {float(score):.3f}" if score is not None else ""
                story.append(para(f"• {title}{score_text}", "Small"))
        else:
            story.append(para("RAG was enabled, but no matching knowledge source was retrieved.", "Small"))

    strengths = a.get("strengths", []) or []
    improvements = a.get("improvement_areas", []) or []
    story.append(Paragraph("Strengths & Improvement Areas", styles["Section"]))
    strength_flow = [para("• " + safe(x), "Small") for x in strengths] or [para("None available.", "Small")]
    improvement_flow = [para("• " + safe(x), "Small") for x in improvements] or [para("None available.", "Small")]
    si = Table([
        [para("Strengths", "CardTitle"), para("Priority Improvements", "CardTitle")],
        [strength_flow, improvement_flow],
    ], colWidths=[250, 250], repeatRows=1)
    si.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(si)

    # 7-day roadmap
    story.append(PageBreak())
    story.append(Paragraph("Personalized 7-Day Roadmap", styles["Section"]))
    if seven_day:
        rows = [[para("Day", "CardTitle"), para("Focus", "CardTitle"), para("Action Plan", "CardTitle")]]
        for item in seven_day:
            rows.append([
                para(item.get("day", ""), "Small"),
                para(item.get("title", ""), "Small"),
                para(item.get("details", ""), "Small"),
            ])
        t = Table(rows, colWidths=[65, 145, 290], repeatRows=1)
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
    else:
        story.append(para("No 7-day roadmap was generated."))

    # 30-day roadmap
    story.append(Paragraph("Personalized 30-Day Improvement Plan", styles["Section"]))
    if thirty_day:
        rows = [[para("Week", "CardTitle"), para("Focus", "CardTitle"), para("Action Plan", "CardTitle")]]
        for item in thirty_day:
            rows.append([
                para(item.get("week", ""), "Small"),
                para(item.get("title", ""), "Small"),
                para(item.get("details", ""), "Small"),
            ])
        t = Table(rows, colWidths=[75, 145, 280], repeatRows=1)
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
    else:
        story.append(para("No 30-day improvement plan was generated."))

    # Interview preparation
    story.append(PageBreak())
    story.append(Paragraph("AI Interview Preparation", styles["Section"]))
    story.append(Paragraph("Practice Questions", styles["Section"]))
    if questions:
        for number, question in enumerate(questions, 1):
            story.append(para(f"{number}. {question}"))
    else:
        story.append(para("No interview questions were generated."))

    story.append(Paragraph("Interview Focus", styles["Section"]))
    story.extend(bullet_list(interview_focus))

    # ML result + diagnostic summary
    if ml_result and not ml_result.get("error"):
        story.append(Paragraph("Machine Learning Result", styles["Section"]))

        # The ML model is a supporting probability signal, not the final PRIS
        # readiness decision. The final user-facing status is derived from the
        # combined PRIS readiness score, which includes JD/job/skill alignment.
        final_status = readiness_label(readiness)
        ml_rows = [
            [para("PRIS Final Status", "CardTitle"), para(final_status, "CardTitle")],
        ]
        np_prob = ml_result.get("not_placed_probability")
        placed_prob = ml_result.get("placed_probability")
        if np_prob is not None:
            ml_rows.append([para("Not Placed Probability", "Small"), para(f"{np_prob * 100:.1f}%", "Small")])
        if placed_prob is not None:
            ml_rows.append([para("Placed Probability", "Small"), para(f"{placed_prob * 100:.1f}%", "Small")])
        ml_rows.append([para("Branch", "Small"), para(safe(ml_result.get("branch_value")), "Small")])
        ml_rows.append([para("Training Branch Support", "Small"), para("Yes" if ml_result.get("branch_supported") else "No", "Small")])
        ml_table = Table(ml_rows, colWidths=[250, 250])
        ml_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 8))
        story.append(para(
            "Note: The ML model uses the structured candidate profile and its saved preprocessing pipeline. "
            "Its placed probability is a supporting signal, not the final PRIS readiness decision. "
            "The final status is based on the combined readiness score, including job and skill alignment. "
            "For fields unavailable in the resume, the pipeline's preprocessing handles missing values. "
            "An unsupported academic branch is not replaced with another branch.",
            "Small"
        ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "PRIS • Gemini AI • Machine Learning • NLP • Resume Intelligence • Skill Gap Analysis",
        styles["Small"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎓 PRIS")
    st.caption("Placement Readiness Intelligence System")
    st.divider()

    st.markdown("### ⚙️ AI Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter your Gemini API key",
    )

    model_name = st.selectbox(
        "Choose Gemini Model",
        ["gemini-3.5-flash-lite", "gemini-3.6-flash"],
    )

    demo_mode = st.checkbox("🎯 Demo Mode", value=False)

    st.divider()

    if ml_model is not None:
        st.success("ML model loaded")
    else:
        st.warning("ML model unavailable")

        if ml_error:
            with st.expander("Model diagnostic"):
                st.code(ml_error)

    st.caption(
        "PRIS combines resume intelligence, job matching, "
        "skill-gap analysis, machine learning and career roadmaps."
    )


# ============================================================
# HERO - NATIVE STREAMLIT, NO HTML
# ============================================================

st.info("✦ AI-POWERED PLACEMENT ANALYTICS")

st.title("🎓 Placement Readiness Intelligence System")

st.write(
    "Transform a student's resume and target job description into a "
    "complete placement-readiness intelligence report powered by "
    "Gemini AI, NLP and Machine Learning."
)


# ============================================================
# FEATURE CARDS
# ============================================================

st.markdown("## ✨ What PRIS Does")

f1, f2, f3, f4 = st.columns(4)

with f1:
    with st.container(border=True):
        st.subheader("📄 Resume Intelligence")
        st.caption("Extract candidate profile, education, skills, projects and experience.")

with f2:
    with st.container(border=True):
        st.subheader("🎯 Job Match")
        st.caption("Compare the resume against the target role and measure job fit.")

with f3:
    with st.container(border=True):
        st.subheader("⚠️ Skill Gap")
        st.caption("Identify missing and critical skills that need improvement.")

with f4:
    with st.container(border=True):
        st.subheader("🚀 Career Roadmap")
        st.caption("Generate personalized 7-day and 30-day improvement plans.")


# ============================================================
# INPUT
# ============================================================

st.markdown("## 🔎 Start Your Placement Analysis")
st.caption(
    "Upload the student's resume and paste the company's job description."
)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("📄 Step 1 — Upload Resume")
        st.caption("Upload the student's resume in PDF format.")

        resume_file = st.file_uploader(
            "Resume PDF",
            type=["pdf"],
        )

        if resume_file:
            st.success(f"Uploaded: {resume_file.name}")

with right:
    with st.container(border=True):
        st.subheader("💼 Step 2 — Job Description")
        st.caption("Paste the target company's job description.")

        jd_text = st.text_area(
            "Job Description",
            height=220,
            placeholder="Paste the complete job description here...",
        )

st.write("")

analyze_clicked = st.button(
    "🚀 Analyze Placement Readiness",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_clicked:

    if demo_mode:
        with st.spinner("Preparing PRIS demo report..."):
            st.session_state["pris_result"] = demo_result()

        st.success("Demo analysis completed successfully.")

    else:
        if not resume_file:
            st.error("Please upload a resume PDF.")
            st.stop()

        if not jd_text.strip():
            st.error("Please paste the job description.")
            st.stop()

        if not api_key.strip():
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()

        with st.spinner("PRIS is analyzing the resume and job description..."):
            try:
                st.session_state["pris_result"] = analyze_resume_and_jd(
                    resume_file,
                    jd_text,
                    api_key.strip(),
                    model_name,
                )
                st.success("AI analysis completed successfully.")

            except Exception as exc:
                st.error(f"Gemini analysis failed: {exc}")
                st.stop()


# ============================================================
# RESULTS
# ============================================================

if "pris_result" in st.session_state:

    result = st.session_state["pris_result"]

    ml_result = run_ml_prediction(result)
    readiness = calculate_readiness(result, ml_result)
    label = readiness_label(readiness)

    candidate = result.get("candidate", {})
    job = result.get("job", {})
    match = result.get("match", {})
    analysis = result.get("analysis", {})

    st.divider()
    st.markdown("## 📊 Placement Intelligence Report")

    st.subheader(candidate.get("name", "Candidate"))
    st.write(f"**Target Role:** {job.get('role', 'Not specified')}")

    a, b, c, d = st.columns(4)

    a.metric(
        "🎯 Job Match",
        f"{safe_int(match.get('job_match_percentage', 0))}%",
    )

    b.metric(
        "🧩 Skill Match",
        f"{safe_int(match.get('skill_match_percentage', 0))}%",
    )

    if ml_result and not ml_result.get("error"):
        probability = ml_result.get("placed_probability")
        ml_value = f"{probability * 100:.1f}%" if probability is not None else "N/A"
    else:
        ml_value = "N/A"

    c.metric("🤖 ML Placement Probability", ml_value)
    d.metric("📈 Readiness Score", f"{readiness:.0f}/100")

    st.progress(int(readiness))
    st.success(f"Current readiness level: {label}")

    # Candidate profile
    st.markdown("## 👤 Candidate Intelligence")

    p1, p2 = st.columns([1, 1.6], gap="large")

    with p1:
        with st.container(border=True):
            st.subheader("Candidate Profile")

            st.write(f"**Name:** {candidate.get('name', 'N/A')}")
            st.write(f"**Education:** {candidate.get('education', 'N/A')}")
            st.write(f"**Degree:** {candidate.get('degree', 'N/A')}")
            st.write(f"**Branch:** {candidate.get('branch', 'N/A')}")
            st.write(f"**CGPA:** {candidate.get('cgpa', 'N/A')}")
            st.write(f"**Internships:** {candidate.get('internships', 0)}")
            st.write(f"**Projects:** {candidate.get('projects_count', 0)}")
            st.write(f"**Certifications:** {candidate.get('certifications', 0)}")

    with p2:
        with st.container(border=True):
            st.subheader("🧠 AI Executive Summary")
            st.write(analysis.get("summary", "No summary available."))

    # Skills
    st.markdown("## 🧩 Skill Gap Intelligence")

    s1, s2, s3 = st.columns(3)

    with s1:
        with st.container(border=True):
            st.subheader("Matched Skills")
            skills = match.get("matched_skills", [])
            for skill in skills:
                st.success(str(skill))

    with s2:
        with st.container(border=True):
            st.subheader("Missing Skills")
            skills = match.get("missing_skills", [])
            for skill in skills:
                st.warning(str(skill))

    with s3:
        with st.container(border=True):
            st.subheader("Critical Gaps")
            skills = match.get("critical_missing_skills", [])
            for skill in skills:
                st.error(str(skill))

    # RAG
    st.markdown("## 📚 RAG Knowledge Retrieval")
    rag = result.get("rag", {}) or {}

    if rag.get("enabled"):
        sources = rag.get("skill_gap_sources", []) or rag.get("retrieved_sources", []) or []
        if sources:
            st.caption(
                "PRIS retrieves relevant learning and interview-preparation knowledge "
                "from the local knowledge base and uses it to ground recommendations."
            )
            rcols = st.columns(min(4, len(sources)))
            for i, item in enumerate(sources):
                with rcols[i]:
                    with st.container(border=True):
                        st.subheader(str(item.get("title", "Knowledge Source")))
                        score = item.get("score")
                        if score is not None:
                            st.metric("Similarity", f"{float(score):.3f}")
                        st.caption(str(item.get("source", "")))
        else:
            st.info("RAG is enabled, but no matching knowledge source was retrieved for this profile.")
    else:
        st.warning("RAG module is not available. Ensure rag_knowledge_base.py and rag_knowledge/ are in the PRIS project folder.")
        if RAG_IMPORT_ERROR:
            with st.expander("RAG diagnostic"):
                st.code(RAG_IMPORT_ERROR)

    # Breakdown
    st.markdown("## 📈 Readiness Breakdown")

    breakdown = {
        "Job Match": clamp(match.get("job_match_percentage", 0)),
        "Skill Match": clamp(match.get("skill_match_percentage", 0)),
        "Keywords": clamp(match.get("keyword_match_score", 0)),
        "Projects": clamp(analysis.get("project_relevance_score", 0)),
        "Certifications": clamp(analysis.get("certification_relevance_score", 0)),
        "Internship": clamp(analysis.get("internship_relevance_score", 0)),
        "Resume": clamp(analysis.get("resume_completeness_score", 0)),
        "Role Fit": clamp(analysis.get("role_category_match_score", 0)),
    }

    fig = go.Figure(
        go.Bar(
            x=list(breakdown.keys()),
            y=list(breakdown.values()),
            text=[f"{v:.0f}%" for v in breakdown.values()],
            textposition="outside",
        )
    )

    fig.update_layout(
        height=420,
        yaxis=dict(range=[0, 110], title="Score"),
        xaxis_title="Readiness Dimension",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce6f2"),
        margin=dict(l=20, r=20, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Strengths
    st.markdown("## 💡 Strengths & Improvement Areas")

    q1, q2 = st.columns(2)

    with q1:
        with st.container(border=True):
            st.subheader("Strengths")
            for item in analysis.get("strengths", []):
                st.success(str(item))

    with q2:
        with st.container(border=True):
            st.subheader("Priority Improvements")
            for item in analysis.get("improvement_areas", []):
                st.warning(str(item))

    # 7 day
    st.markdown("## 🗓️ Personalized 7-Day Roadmap")

    for item in result.get("roadmap", {}).get("seven_day", []):
        with st.container(border=True):
            st.caption(item.get("day", ""))
            st.subheader(item.get("title", ""))
            st.write(item.get("details", ""))

    # 30 day
    st.markdown("## 🗓️ Personalized 30-Day Roadmap")

    thirty = result.get("roadmap", {}).get("thirty_day", [])

    if thirty:
        cols = st.columns(min(4, len(thirty)))

        for i, item in enumerate(thirty):
            with cols[i]:
                with st.container(border=True):
                    st.caption(item.get("week", ""))
                    st.subheader(item.get("title", ""))
                    st.write(item.get("details", ""))

    # Interview
    st.markdown("## 🎤 AI Interview Preparation")

    i1, i2 = st.columns(2)

    with i1:
        with st.container(border=True):
            st.subheader("Practice Questions")

            questions = result.get("interview_questions", [])

            for number, question in enumerate(questions, 1):
                st.write(f"**{number}.** {question}")

    with i2:
        with st.container(border=True):
            st.subheader("Interview Focus")

            for item in result.get("interview_focus", []):
                st.info(str(item))

    # ML
    st.markdown("## 🤖 Machine Learning Result")

    if ml_result and not ml_result.get("error"):

        p = ml_result.get("probability")
        final_status = readiness_label(readiness)

        with st.container(border=True):
            st.subheader("Placement Readiness Decision")
            st.write(f"**PRIS Final Status:** {final_status}")
            st.caption(
                "The final status is based on the combined PRIS readiness score. "
                "The ML placement probability is shown separately as a supporting signal; "
                "it does not override JD/skill alignment."
            )

            np_prob = ml_result.get("not_placed_probability")
            placed_prob = ml_result.get("placed_probability")

            if np_prob is not None and placed_prob is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(
                        "Not Placed Probability",
                        f"{np_prob * 100:.1f}%"
                    )
                with c2:
                    st.metric(
                        "Placed Probability",
                        f"{placed_prob * 100:.1f}%"
                    )

            skill_match_value = clamp(match.get("skill_match_percentage", 0))
            if skill_match_value < 50:
                st.warning(
                    f"Skill Match is {skill_match_value:.0f}%. The candidate has significant JD skill gaps, "
                    "so the PRIS final status is not determined by the ML placed probability alone."
                )

            with st.expander("🔎 ML Input Features — Diagnostic"):
                features = ml_result.get("features", {})
                if features:
                    diagnostic_df = pd.DataFrame(
                        {
                            "Feature": list(features.keys()),
                            "Value": list(features.values()),
                        }
                    )
                    st.dataframe(
                        diagnostic_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Direct candidate fields (branch, CGPA, internships, projects, "
                        "certifications and backlogs) are mapped into the ML profile before prediction."
                    )

                    if ml_result.get("branch_value") and not ml_result.get("branch_supported"):
                        st.warning(
                            f"Resume branch '{ml_result.get('branch_value')}' is not represented in the training dataset. "
                            "It was not replaced with another branch. The saved model may therefore have limited reliability for this profile."
                        )

                    if not ml_result.get("tier_supported"):
                        st.info(
                            "College tier was not available with sufficient evidence, so the saved preprocessing pipeline will impute it."
                        )
                else:
                    st.info("No ML feature values were returned.")

            st.caption(
                "The ML model uses the structured candidate profile from the resume. "
                "Its placed probability is a supporting signal only; PRIS does not use "
                "the raw predicted class as the final readiness decision."
            )

    else:
        st.warning(
            "The dashboard is working, but the saved ML model is not "
            "compatible with the current scikit-learn environment."
        )

    # PDF
    st.markdown("## 📄 Placement Readiness Report")

    try:
        pdf_bytes = create_pdf_report(
            result,
            readiness,
            ml_result,
        )

        st.download_button(
            "📄 Download Placement Readiness Report",
            data=pdf_bytes,
            file_name="PRIS_Placement_Readiness_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"PDF creation failed: {exc}")

else:

    st.divider()

    with st.container(border=True):
        st.markdown("## 🚀 Ready to Analyze?")
        st.write(
            "Upload a resume, paste a job description and click "
            "**Analyze Placement Readiness** to generate the complete "
            "PRIS intelligence report."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="pris-footer">
        🎓 Placement Readiness Intelligence System<br>
        Gemini AI • Machine Learning • NLP • RAG • Resume Intelligence •
        Skill Gap Analysis • Career Roadmap<br><br>
        GUVI Data Science & Generative AI Final Project
    </div>
    """,
    unsafe_allow_html=True,
)
