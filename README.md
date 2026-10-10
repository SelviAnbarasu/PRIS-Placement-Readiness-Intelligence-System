# 🎯 Placement Readiness Intelligence System (PRIS)

### AI-Powered Resume & Job Description Based Placement Readiness Analyzer

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)](https://scikit-learn.org/)
[![Generative AI](https://img.shields.io/badge/Generative%20AI-Gemini-purple)](https://ai.google.dev/)
[![RAG](https://img.shields.io/badge/RAG-TF--IDF%20%2B%20Cosine%20Similarity-green)](#rag-implementation)
[![Interface](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io/)

## 📌 Project Overview

The **Placement Readiness Intelligence System (PRIS)** is an AI-powered application that helps candidates understand their readiness for a target job by analyzing their resume against a job description.

It combines Machine Learning, Generative AI, and Retrieval-Augmented Generation (RAG) to identify matching skills, highlight skill gaps, and provide personalized improvement recommendations.

## ✨ Key Features

- 📄 Resume PDF text extraction
- 🤖 Resume and Job Description analysis using Gemini AI
- 🎯 Matched and missing skill identification
- 🔍 Skill gap analysis
- 🧠 Machine Learning-based placement prediction signal
- 📚 RAG-based knowledge retrieval
- 🗺️ Personalized 7-day and 30-day improvement plans
- 💬 Interview preparation recommendations
- 📊 Interactive Streamlit dashboard
- 📑 Downloadable PDF report

## 🏗️ System Architecture

```text
Resume PDF + Job Description
            |
            v
    Python Resume Processing
            |
            v
       Gemini AI Analysis
            |
            v
   Matched & Missing Skills
            |
            v
      Local RAG Retrieval
   (TF-IDF + Cosine Similarity)
            |
            v
  Learning Recommendations
            |
            v
    Machine Learning Model
            |
            v
  Final Readiness Score
            |
            v
   Streamlit Dashboard
            |
            v
      PDF Report
```

## 🐍 Python and Machine Learning Workflow

1. Loaded and explored the student placement dataset using Pandas.
2. Performed data preprocessing and prepared numerical and categorical features.
3. Split the dataset into 80% training data and 20% testing data.
4. Trained and compared four Machine Learning algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
5. Evaluated models using Accuracy, Precision, Recall, and F1 Score.
6. Selected Gradient Boosting based on the model comparison.
7. Saved the trained model using Joblib and integrated it into the application.

**Important:** The trained model provides a placement prediction signal. The final PRIS readiness score also considers job alignment, skill matching, and other relevant analysis features.

## 🤖 Generative AI Integration

Gemini AI analyzes the candidate's resume and target job description to identify relevant skills, missing skills, candidate strengths, and areas for improvement.

The resulting analysis supports personalized recommendations and interview preparation.

## 📚 RAG Implementation

PRIS uses a local knowledge base with **TF-IDF vectorization and cosine similarity** to retrieve relevant knowledge for a candidate's target role and skill gaps.

The retrieved context is provided to Gemini to support learning recommendations, improvement roadmaps, and interview preparation.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Pandas & NumPy | Data handling and numerical operations |
| Scikit-learn | Data preprocessing and Machine Learning |
| Gradient Boosting | Placement prediction signal |
| Joblib | Saving and loading the trained model |
| Gemini API | Resume and Job Description analysis |
| TF-IDF & Cosine Similarity | Local RAG retrieval |
| PyPDF | Resume PDF text extraction |
| Streamlit | Interactive web application |
| ReportLab | PDF report generation |

## 📁 Project Structure

```text
PRIS/
├── app.py
├── rag_knowledge_base.py
├── rag_knowledge/
│   ├── audit_compliance.md
│   ├── cash_flow_management.md
│   ├── excel_finance.md
│   ├── financial_analysis.md
│   ├── placement_readiness.md
│   ├── power_bi.md
│   ├── python_data_analysis.md
│   ├── resume_interview.md
│   └── sql_data_analytics.md
├── retrain_pris_model.py
├── student_placement_model.pkl
├── requirements.txt
├── requirements_rag.txt
└── README.md
```

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/SelviAnbarasu/PRIS-Placement-Readiness-Intelligence-System.git
cd PRIS-Placement-Readiness-Intelligence-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Enter your Gemini API key in the application sidebar when prompted.

**Security:** Never commit API keys, passwords, or other secrets to GitHub.

## 📈 Model Evaluation

The project compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using:

- Accuracy
- Precision
- Recall
- F1 Score

The initial Gradient Boosting evaluation achieved an F1 score of approximately **0.81**. Model performance should be interpreted alongside class-wise precision and recall, particularly because the dataset contains more placed than non-placed students.

## 🎯 Expected Outcome

PRIS helps candidates understand their alignment with a target job, identify missing skills, and receive actionable recommendations to improve their placement readiness.

## 👩‍💻 Author

**Selvi Anbarasu**

Advanced Data Science & Generative AI Project

[GitHub Repository](https://github.com/SelviAnbarasu/PRIS-Placement-Readiness-Intelligence-System)

---

*Built with Python, Machine Learning, Generative AI, RAG, and Streamlit.*
