# 🧠 Hate Speech Detector

## 📌 Project Overview

In Kenya's highly active digital space, Twitter has become a central platform for political discourse—ranging from public opinion to targeted harassment. Political figures frequently face hate speech that not only affects reputations but also threatens national unity. This project develops a supervised machine learning model using Natural Language Processing (NLP) to detect hate speech in tweets aimed at Kenyan politicians. The ultimate goal is to support content moderation, improve digital safety, and contribute to civic integrity.

---

## 💼 Business Understanding

### ✅ Problem Statement

Online hate speech targeting Kenyan politicians has become alarmingly frequent. These messages can:

- Incite ethnic tension or violence  
- Harm the mental health of individuals  
- Erode public trust in political institutions  
- Spread rapidly before moderators can respond

Manual moderation is slow, inconsistent, and unable to scale. An automated hate speech detection system can enable early intervention and better resource allocation for moderation teams.

### 🎯 Project Goals

- Automatically classify tweets as **hate** or **neutral**
- Uncover **linguistic trends** in political hate speech
- Provide **actionable insights** for moderation teams, policymakers, and civil rights organizations

### 👥 Key Stakeholders

- Electoral bodies (IEBC, NCIC)  
- NGOs (Ushahidi, Amnesty Kenya)  
- Government communication units  
- Social media platforms (Twitter/X Kenya)  
- Researchers and policy analysts  

### 📏 Success Metrics

**Model Evaluation Metrics:**

- Accuracy  
- Precision and Recall (for "hate" class)  
- F1 Score  
- Confusion Matrix  

**Business Impact Metrics:**

- Time saved in manual moderation  
- Speed of hate speech flagging  
- Cross-group fairness in model predictions  
- Interpretability via tools like SHAP or LIME

---

## 🎯 Project Objectives

1. Preprocess and explore real-world Twitter data
2. Apply NLP techniques to extract meaningful features
3. Train and evaluate supervised ML models (e.g., Logistic Regression, SVM, BERT)
4. Visualize trends in hate speech and sentiment over time
5. Recommend how the model can be applied in moderation workflows or policy monitoring
6. Lay groundwork for real-time, multilingual hate detection systems

---

## 📊 Data Understanding

### 📅 Source & Collection Method

The dataset was collected via Twitter scraping using [Tweepy](https://www.tweepy.org/) and [Twint's TwSearchExporter](https://github.com/twintproject/twint). It focuses on tweets referencing high-profile Kenyan politicians such as:

- President William Ruto  
- Deputy President Rigathi Gachagua  
- Interior Cabinet Secretary Kithure Kindiki  
- Other notable political figures

The time frame spans politically charged periods like elections, protests, or government announcements.

### 📂 Dataset Structure

Total Records: **11,317 tweets**

| Column Name       | Description                                                                 |
|-------------------|------------------------------------------------------------------------------|
| `Tweet ID`        | Unique identifier for each tweet                                             |
| `Likes`           | Count of likes as a proxy for popularity                                     |
| `Retweets`        | Count of how many times the tweet was shared                                 |
| `Total Replies`   | Number of replies (engagement/controversy proxy)                             |
| `Texts`           | Raw tweet content (main input for NLP)                                       |
| `Created At`      | Timestamp of tweet publication for trend/time analysis                       |

---

## 🛠️ Tools & Libraries Used

The following Python libraries were used for data processing, visualization, feature engineering, text analysis, and modeling:

### 📊 Data Manipulation & Visualization
- `pandas`, `numpy`: Data handling and numerical operations
- `matplotlib.pyplot`, `seaborn`: Static visualizations
- `plotly.express`, `plotly.graph_objects`, `plotly.figure_factory`: Interactive visualizations

### 🧠 Feature Engineering & Utilities
- `re`, `calendar`, `dateutil.easter`: Text and date processing
- `spacy`: Advanced NLP and tokenization
- `rapidfuzz`: Fast string matching for fuzzy joins

### 🧼 Text Preprocessing
- `nltk`: Tokenization, stopword removal, and lemmatization
  - `stopwords`, `word_tokenize`, `WordNetLemmatizer`
- `string`: Handling punctuation
- `sklearn.feature_extraction.text.TfidfVectorizer`: Vectorizing text data

### 🤖 Modeling & Evaluation
- `torch`: Deep learning framework (PyTorch)
- `sklearn`: Model building, preprocessing, and evaluation
  - `train_test_split`, `LabelEncoder`, `metrics` (accuracy, F1, confusion matrix, etc.)
- `transformers` (from HuggingFace):
  - `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, `DataCollatorWithPadding`
- `datasets`: HuggingFace Datasets library for efficient model input

### 🚫 Warning Management
- `warnings`: Used to suppress unnecessary warnings during runtime

---

### 🏷️ Label Classes

The target variable is binary:

- `Hate` – Direct hate speech or dehumanizing statements  
- `None-hate` – Informative, benign, or opinionated but respectful content

---

## 🔧 Tools and Technologies

- Python (Pandas, Scikit-learn, NLTK, SpaCy)
- Jupyter Notebook
- NLP Models: TF-IDF, Logistic Regression, SVM, BERT
- Visualizations: Matplotlib, Seaborn, Plotly
- Explainability: SHAP, LIME

---

## 🧼 Data Cleaning

- - Removed duplicates and irrelevant columns
- Handled missing values in `location` and `text`
- Cleaned text (lowercased, removed URLs, mentions, emojis, etc.)

---

## 📈 Data Visualization and Analysis

### 📊 Tweet Count by Sentiment(label)

This chart shows the distribution of tweet count segmented by sentiment labels — **Not_hate** and **Hate Speech**. It provides an overview of class distribution, which is important for understanding dataset balance and guiding model evaluation.


- **Not_hate** tweets dominate during regular days.
- **Hate speech** tweets tend to cluster around certain controversial dates.

![Tweet Count by Sentiment]![image](https://github.com/user-attachments/assets/4c8ddf4b-4f70-486c-8dd5-0f357924a690)

![Tweet Volume by Politician and Sentiment]![image](https://github.com/user-attachments/assets/7638dbd8-e98b-4a3c-87c2-391e1e079f10)

![Average Engagement Score by Year (2016–2025)]![image](https://github.com/user-attachments/assets/57550f75-fe70-443f-aa42-a0ca7b695336)



---

## 🤝 Contribution

This project supports online safety efforts by helping moderation teams, NGOs, and civic tech initiatives flag and respond to political hate speech in real-time.

---

