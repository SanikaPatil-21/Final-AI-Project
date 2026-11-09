
import re
import io

import fitz
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Skill Dictionary
# -----------------------------
SKILLS = [
    "python", "java", "c++", "javascript", "sql",
    "html", "css", "react", "node.js",
    "machine learning", "deep learning",
    "artificial intelligence", "data science",
    "data analysis", "natural language processing",
    "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy",
    "excel", "power bi", "tableau",
    "statistics", "data visualization",
    "flask", "django", "streamlit",
    "git", "github", "docker",
    "aws", "azure", "google cloud",
    "communication", "teamwork",
    "problem solving", "project management",
    "leadership", "critical thinking"
]


# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_pdf_text(pdf_bytes):
    text_parts = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            text_parts.append(page.get_text())

    return "\n".join(text_parts).strip()


# -----------------------------
# Detect Skills
# -----------------------------
def detect_skills(text):
    normalized_text = text.lower()
    found_skills = []

    for skill in SKILLS:
        if skill in ("c++",):
            pattern = r"(?<!\w)c\+\+(?!\w)"
        else:
            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

        if re.search(pattern, normalized_text):
            found_skills.append(skill)

    return sorted(found_skills)


# -----------------------------
# Calculate Text Similarity
# -----------------------------
def calculate_similarity(resume_text, job_text):
    documents = [resume_text, job_text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        vectors = vectorizer.fit_transform(documents)
        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(float(score) * 100, 2)

    except ValueError:
        return 0.0


# -----------------------------
# Main Application
# -----------------------------
st.title("📄 AI Resume Analyzer")
st.subheader("Resume Analysis and Job Matching System")

st.write(
    "Upload a resume, enter a job description, and analyze "
    "their text similarity and detected skills."
)

st.info(
    "The matching score measures text similarity only. "
    "It does not predict hiring success or candidate suitability."
)


# Resume Upload
uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)


# Job Description
job_description = st.text_area(
    "Paste the Job Description",
    height=220,
    placeholder=(
        "Example: We are looking for a Python developer "
        "with knowledge of SQL, machine learning, "
        "data analysis, and communication."
    )
)


# Analyze Button
if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:
        st.error("Please upload a PDF resume.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    else:
        try:
            pdf_bytes = uploaded_file.getvalue()
            resume_text = extract_pdf_text(pdf_bytes)

            if not resume_text:
                st.error(
                    "No readable text was found in this PDF. "
                    "It may be a scanned image. Try a text-based PDF."
                )
                st.stop()

            if len(job_description.strip()) < 10:
                st.error(
                    "Please enter a more detailed job description."
                )
                st.stop()

            # Perform Analysis
            resume_skills = detect_skills(resume_text)
            job_skills = detect_skills(job_description)

            matched_skills = sorted(
                set(resume_skills) & set(job_skills)
            )

            missing_skills = sorted(
                set(job_skills) - set(resume_skills)
            )

            match_score = calculate_similarity(
                resume_text,
                job_description
            )

            # Display Results
            st.divider()
            st.header("📊 Analysis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Text Similarity",
                f"{match_score}%"
            )

            col2.metric(
                "Matched Skills",
                len(matched_skills)
            )

            col3.metric(
                "Detected Job Skills",
                len(job_skills)
            )

            st.progress(match_score / 100)

            st.caption(
                "A higher score indicates greater textual similarity, "
                "not a probability of being hired."
            )

            # Skill Comparison
            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("✅ Matched Skills")

                if matched_skills:
                    st.write(", ".join(matched_skills))
                else:
                    st.write("No matching skills detected.")

            with right_col:
                st.subheader("📌 Job Skills Not Detected")

                if missing_skills:
                    st.write(", ".join(missing_skills))
                else:
                    st.write(
                        "All detected job-description skills "
                        "were also detected in the resume."
                    )

            st.subheader("🧠 Skills Detected in Resume")

            if resume_skills:
                st.write(", ".join(resume_skills))
            else:
                st.write("No skills from the current dictionary detected.")

            # Analysis Table
            all_job_skills = sorted(set(job_skills))

            results_df = pd.DataFrame({
                "Job Skill": all_job_skills,
                "Detected in Resume": [
                    "Yes" if skill in resume_skills else "No"
                    for skill in all_job_skills
                ]
            })

            st.subheader("📋 Skill Analysis Table")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            # Download Results
            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Skill Analysis (CSV)",
                data=csv_data,
                file_name="resume_analysis.csv",
                mime="text/csv"
            )

            report = f"""
AI RESUME ANALYZER REPORT

Text Similarity Score: {match_score}%

Matched Skills:
{", ".join(matched_skills) if matched_skills else "None detected"}

Job Skills Not Detected in Resume:
{", ".join(missing_skills) if missing_skills else "None"}

Skills Detected in Resume:
{", ".join(resume_skills) if resume_skills else "None"}

Note:
The similarity score is a text comparison measure, not a
prediction of hiring success. Undetected skills may still be
possessed by the candidate.
"""

            st.download_button(
                label="📥 Download Analysis Report (TXT)",
                data=report,
                file_name="resume_analysis_report.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error(
                "An error occurred while analyzing the resume."
            )
            st.exception(error)


# Footer
st.divider()

st.caption(
    "AI Resume Analyzer | Educational Project | "
    "Python, Streamlit, PyMuPDF and Scikit-learn"
)
