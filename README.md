
# Resume Classification Project

This project builds an NLP-based Resume Classification System that categorizes resumes into predefined job categories and deploys the trained model using Streamlit

---

## Project Structure

```
Resume-Classification-Project/
│
├── models/
│   ├── label_encoder.pkl
│   └── resume_classifier_nb_pipeline.pkl
│
├── data/
│   ├── Test_Resumes/
│   ├── Unmodified_Resume_Docx/
│   ├── Resume_Docx/
│   ├── Cleaned_Resumes.csv
│   └── Raw_Resume.csv
│
├── notebooks/
│   ├── modeling.ipynb
│   └── analysis.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── Business_Objective.docx
```

## Setup & Execution (Windows - CMD)

1. Open Command Prompt and navigate to project folder:
   Ex:- cd "D:\Data Science Excelr\Resume Classification"

2. Create Virtual Environment for Once:
   python -m venv venv

3. Activate the virtual environment:
   venv\Scripts\activate

4. Upgrade pip:
   python -m pip install --upgrade pip

5. Install project dependencies:
   pip install -r requirements.txt

6. Run the Flask app:
   streamlit run app.py

7. Open your browser and go to the link shown (usually (http://192.168.88.16:8501)):
   Press CTRL+C

8. Deactivate the virtual environment:
   deactivate

# Resume Classification Project – Comprehensive Pipeline

## Phase 1: Data Analysis – Resume Classification

### 1.1 Project Setup

#### 1.1.1 Environment Configuration
- Imported core libraries: pandas, numpy.
- NLP libraries: nltk, spacy.
- Visualization: matplotlib, seaborn, wordcloud.
- Feature engineering: CountVectorizer, TfidfVectorizer.
- Downloaded NLTK stopwords and loaded SpaCy `en_core_web_sm`.

---

### 1.2 Data Standardization

#### 1.2.1 Resume Formats
- Source formats: PDF, DOC, DOCX.
- Standardized into uniform text structure for consistent extraction.

#### 1.2.2 Text Extraction
- Extracted paragraph content and table data (skills often stored in tables).
- Combined into a single structured dataset.
- Assigned corresponding job category labels.

---

### 1.3 Initial Data Inspection

#### 1.3.1 Dataset Checks
- Verified dataset size and category distribution.
- Checked duplicates and missing values.
- Measured resume length and token statistics.

---

### 1.4 Text Preprocessing & Cleaning

#### 1.4.1 Noise Removal
- Removed URLs, HTML tags, emails, years, months, numeric-only tokens.
- Removed special characters and formatting artifacts.

#### 1.4.2 Custom Stopword Strategy
- Combined NLTK stopwords with resume-specific generic words (experience, project, summary, responsible, etc.).
- Removed business-fluff tokens.

#### 1.4.3 Linguistic Filtering (SpaCy)
- Removed pronouns, determiners, conjunctions, particles.
- Removed named entities (DATE, TIME, LOCATION, etc.).
- Retained meaningful technical tokens.

#### 1.4.4 Regex-Based Technical Token Filtering
- Preserved technical patterns (C++, C#, PL/SQL, REST APIs, hyphenated/underscored terms).
- Removed fluff patterns and action verbs (ending in “ing”, “ed”).
- Final output: normalized technical-token-focused resume text.

---

### 1.5 Exploratory Data Visualization

This section analyzes patterns in the cleaned dataset.

#### 1.5.1 Category Distribution Analysis
- Count plot of resumes per category.
- Pie chart showing percentage distribution.
- Verified class balance across job roles.

#### 1.5.2 Resume Length Comparison
- Computed resume length using character count.
- Boxplot of resume length by category.
- Observed variation in resume size across roles.

---

### 1.6 Keyword & N-gram Analysis

#### 1.6.1 Top Keywords per Category
- Used CountVectorizer to extract top words per job role.
- Visualized category-specific dominant technical terms using bar plots.

#### 1.6.2 WordCloud Visualization
- Generated global WordCloud of cleaned resume text.
- Generated category-wise WordClouds to visualize skill prominence.
- Identified visually dominant technologies per job role.

#### 1.6.3 N-gram Analysis
- Extracted bi-grams and tri-grams using CountVectorizer.
- Identified common multi-word skill patterns (e.g., “application engine”, “react js”, etc.).
- Visualized top n-grams using bar plots.

---

### 1.7 TF-IDF Feature Engineering & Analysis

#### 1.7.1 Category-wise TF-IDF Analysis
- Applied TfidfVectorizer (1–3 grams, max_features=3000, min_df=10, max_df=0.8).
- Computed average TF-IDF scores per category.
- Identified discriminative words per job role.

#### 1.7.2 TF-IDF Visualization
- Plotted top TF-IDF words per category using bar plots.
- Generated WordClouds weighted by TF-IDF scores.
- Extracted top global TF-IDF terms across all resumes.

---

### Outcome of Phase 1

The analysis phase provides:

- Cleaned and normalized resume corpus.
- Balanced and validated category distribution.
- Insight into keyword dominance per job role.
- Identification of meaningful bi-grams and tri-grams.
- Discriminative TF-IDF features for model training.
- A structured, model-ready dataset.

## Phase 2 – Model Training & Evaluation

**Prerequisite:** `analysis.ipynb` must be completed and `Cleaned_Resumes.csv` must exist.

---

### 1. Data Loading & Validation

- Loaded preprocessed dataset: `../data/Cleaned_Resumes.csv`
- Verified shape, columns, missing values, duplicates.
- Checked category distribution to confirm balanced classes.
- Target column: `Category`
- Feature column: `Resume_Details`

---

### 2. Target Encoding

- Applied `LabelEncoder` to convert categorical job roles into numeric labels.
- Stored mapping for inverse transformation during prediction.

---

### 3. Train–Test Split

- Split data into 80% training and 20% testing.
- Used `stratify=y` to preserve class distribution.
- Fixed `random_state=42` for reproducibility.

---

### 4. Feature Extraction – TF-IDF

`TfidfVectorizer` configuration:
- `max_features=1500`
- `ngram_range=(1,2)`
- `sublinear_tf=True`
- `min_df=2`
- `max_df=0.8`
- `stop_words='english'`

Integrated into sklearn `Pipeline` for clean training workflow.

---

### 5. Models Trained & Evaluated

All models were trained using TF-IDF + classifier inside a Pipeline and evaluated using:
- Training Accuracy
- Testing Accuracy
- 5-Fold Cross-Validation
- Classification Report
- Confusion Matrix

#### Models Implemented

1. AdaBoost (Base + GridSearch tuned)
2. Gradient Boosting (GridSearch tuned)
3. Decision Tree (Base + GridSearch tuned)
4. Random Forest
5. Logistic Regression
6. Multinomial Naive Bayes (GridSearch tuned)
7. Linear SVM (GridSearch tuned)

---

### 6. Naive Bayes Performance

- Achieved 100% Training Accuracy
- Achieved 100% Testing Accuracy
- 5-Fold CV Mean Accuracy = 1.00
- CV Std = 0.00

Reason: Dataset contains strongly role-specific technical vocabularies with minimal overlap, making classes highly separable in TF-IDF space.

---

### 7. Data Leakage Check (Label Shuffle Test)

- Randomly shuffled target labels.
- Re-trained best model.
- Accuracy dropped to near-random (~0.30 for 4 classes).

Conclusion: No data leakage; model performance is due to genuine discriminative features.

---

### 8. Model Comparison

Compared models using:
- Train Accuracy
- Test Accuracy
- CV Mean Accuracy
- CV Standard Deviation

Observation:
- Naive Bayes and Linear SVM achieved identical perfect scores.
- Naive Bayes selected as final model due to lower complexity, faster inference, and suitability for TF-IDF text classification.

---

### 9. Model Persistence

Saved artifacts:

- `resume_classifier_nb_pipeline.pkl` (TF-IDF + Naive Bayes Pipeline)
- `label_encoder.pkl` (Category Encoder)

Used `joblib` for serialization.

---

### 10. Prediction on New Resumes

#### Supported Formats
- PDF
- DOC
- DOCX

#### Prediction Workflow
1. Convert PDF/DOC → DOCX if required.
2. Extract paragraph + table text.
3. Apply full custom preprocessing pipeline.
4. Transform using saved TF-IDF.
5. Predict using trained Naive Bayes.
6. Decode label using `LabelEncoder`.

#### Output
- Predicted Category
- Raw Extracted Text
- Cleaned Technical Token Text

---

### Phase 2 Outcome

- Multiple ML models trained and tuned.
- Verified absence of data leakage.
- Selected Naive Bayes as final production model.
- Saved deployment-ready pipeline.
- Implemented end-to-end prediction logic for unseen resumes.

## Phase 3 – Model Deployment (Streamlit Application)

This phase deploys the trained **Naive Bayes + TF-IDF pipeline** as an interactive Streamlit web application for real-time resume classification.

---

### 1. Application Overview

The Streamlit app allows users to:

- Upload one or multiple resumes (PDF, DOC, DOCX)
- Automatically extract text (paragraphs + tables)
- Apply the same preprocessing pipeline used during training
- Predict job category
- Extract experience
- Extract technical skills
- Download structured results as CSV

The deployed model uses:
- `resume_classifier_nb_pipeline.pkl`
- `label_encoder.pkl`

---

### 2. Resume Processing Pipeline (Deployment Flow)

Raw Resume → Format Conversion → Text Extraction → Custom NLP Cleaning → TF-IDF → Naive Bayes → Label Decoding → Output Display

---

### 3. File Handling & Text Extraction

#### 3.1 Supported Formats
- PDF
- DOC
- DOCX

#### 3.2 Conversion Logic
- PDF → DOCX (via pdf2docx)
- DOC → DOCX (via win32 Word automation)
- DOCX → processed directly

#### 3.3 Text Extraction
- Extracts paragraph text.
- Extracts table content (important for skills sections).
- Merges all content into a single raw text string.

---

### 4. Custom Resume Preprocessing (Production Version)

The exact preprocessing logic used in training is reused during deployment to avoid feature mismatch.

#### 4.1 Text Normalization
- Remove URLs, HTML tags, special characters.
- Normalize separators.
- Remove standalone numbers and dates.
- Replace hyphenated words with underscore format.

#### 4.2 Entity & POS Filtering (SpaCy)
- Remove pronouns, determiners, conjunctions.
- Remove named entities (DATE, TIME, GPE, LOC, etc.).
- Remove cardinal and quantity tokens.

#### 4.3 Stopword Strategy
- NLTK stopwords
- Resume-specific generic words
- Custom business-fluff words

#### 4.4 Regex-Based Filtering
- Preserve technical patterns (C++, C#, PL/SQL, hyphen/underscore tokens).
- Remove action verbs (ending in *ing*, *ed*).
- Remove resume boilerplate (summary, profile, responsibilities, etc.).

Final output: technical-token-focused cleaned resume text.

---

### 5. Job Category Prediction

- Cleaned text is passed into the saved pipeline.
- TF-IDF transforms text into feature space.
- Naive Bayes predicts encoded label.
- LabelEncoder converts numeric label back to original job category.

Output: Predicted Job Role (Peoplesoft / React Developer / SQL Developer / Workday).

---

### 6. Experience Extraction

Two methods used:

#### 6.1 Direct Pattern Detection
Detects:
- “5 years”
- “3+ years”
- “2.5 years”
- “8 months”

#### 6.2 Date Range Calculation
- Detects patterns like: `MM/YYYY – Present`
- Computes total duration dynamically.
- Returns consolidated experience summary in years/months.

---

### 7. Skill Extraction

#### 7.1 Predefined Skill Dictionary Includes
- Programming Languages
- Frontend Frameworks
- Backend Technologies
- Databases
- PeopleSoft Tools
- Workday Modules
- DevOps Tools
- Cloud Platforms

#### 7.2 Extraction Logic
- Word-boundary regex matching.
- Case-insensitive detection.
- Removes duplicates.
- Returns structured list of matched skills.

---

### 8. Batch Processing

- Accepts multiple resume uploads.
- Processes each file independently.
- Stores results in a structured DataFrame.
- Displays summary table in UI.

---

### 9. Output & Export

For each resume, the app displays:

- File Name
- Predicted Category
- Extracted Experience
- Extracted Skills

Additionally:
- Saves results to `resume_classification_results.csv`
- Provides downloadable CSV button inside the app

---

### Phase 3 Outcome

- Fully deployed ML application.
- End-to-end automated resume screening.
- Consistent preprocessing between training and production.
- Supports real-world resume formats.
- Enables scalable, batch resume classification with structured output.
