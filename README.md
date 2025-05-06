#  Hate Speech Detector

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

### N-Gram Frequency Analysis

This section presents an overview of the top N-gram frequencies (from unigrams to 4-grams) extracted from a textual dataset, highlighting key themes and entities.

### Top 20 1-Grams (Unigrams)

![Top 1-Grams]![image](https://github.com/user-attachments/assets/a1d76125-8fa3-4e90-8dbd-3447dea98e9e)


**Observations:**
- The most frequent unigrams are common English stopwords such as **"the"**, **"to"**, **"and"**, **"is"**, and **"of"**.
- Among named entities, **"ruto"** appears with notable frequency, suggesting a high presence of Kenya's President **William Ruto** in the text.
- Other pronouns like **"he"**, **"you"**, **"we"**, and **"I"** also rank highly, indicating a conversational or narrative-heavy dataset.

---

### Top 20 2-Grams (Bigrams)

![Top 2-Grams]![image](https://github.com/user-attachments/assets/5168441c-8e41-4237-b7ee-0d66b8b3fe71)


**Observations:**
- Frequent bigrams include expected phrase structures like **"of the"**, **"in the"**, and **"is a"**.
- Key named entities become clearer: **"william ruto"**, **"riggy g"**, **"rachel ruto"**, **"raila odinga"**, and **"martha koome"**—all prominent figures in Kenyan politics.
- Shows a strong political discourse embedded in the dataset.

---

### Top 20 3-Grams (Trigrams)

![Top 3-Grams]![image](https://github.com/user-attachments/assets/ae9abd5e-473f-4330-a3b0-198aa03667f7)


**Observations:**
- Named trigrams such as **"president william ruto"**, **"mama rachel ruto"**, and **"cs aden duale"** confirm the political orientation of the dataset.
- Political sentiment and action-oriented phrases like **"ruto must go"**, **"reject finance bill"**, and **"is our business"** begin to surface.
- This level offers insight into not just entities but their roles or associated events.

---

### Top 20 4-Grams

![Top 4-Grams]![image](https://github.com/user-attachments/assets/fe53a41d-f738-4860-82cc-b6ddaf12859a)


**Observations:**
- Clear, coherent phrases emerge such as:
  - **"we are african and"**, **"africa is our business"**
  - **"reject the finance bill"**
  - **"first lady rachel ruto"**, **"chief justice martha koome"**
- Highlights both socio-political narratives and government-related discourse.
- Phrases suggest emphasis on **identity, governance, and resistance** themes.

---

> ### Summary

>This N-gram analysis reveals a **politically charged dataset** centered on Kenyan political figures, governance, and current socio-political themes. The higher-order N-grams (3- and 4-grams) provide the clearest insight into **named entities**, **political discourse**, and **sentiment** (e.g., protest language, national identity assertions).
---

### 📊 Distribution of Labels (label)

This chart shows the distribution of tweet count segmented by sentiment labels — **Not_hate** and **Hate Speech**. It provides an overview of class distribution, which is important for understanding dataset balance and guiding model evaluation.


- **Not_hate** tweets dominate during regular days.
- **Hate speech** tweets tend to cluster around certain controversial dates.

![Distribution of Labels]![image](https://github.com/user-attachments/assets/34abf4d4-ca70-443f-8476-dfa25b8142a6)

![Engagement Score vs Hour of Day]![image](https://github.com/user-attachments/assets/a6a4bac8-ff37-481b-b62b-33c8bb266eec)

![Label vs Holiday]![image](https://github.com/user-attachments/assets/3a3ed535-bbe0-4af2-a8fa-9071abf6ad82)

![Tweet Volume by Politician and Sentiment]![image](https://github.com/user-attachments/assets/7638dbd8-e98b-4a3c-87c2-391e1e079f10)

![Average Engagement Score by Year (2016–2025)]![image]![image](https://github.com/user-attachments/assets/296402c7-04f7-4418-bf52-715c335a783d)

![Average Engagement score by Holiday]!![image](https://github.com/user-attachments/assets/0501a83d-b05f-4ae7-a5f9-e2e1de15c054)

![Engagement Breakdown Likes / Retweets / Replies vs Label]![image](https://github.com/user-attachments/assets/c3393a73-601e-4060-b642-c642ed22567e)

![Word count vs text length per label category]![image](https://github.com/user-attachments/assets/848da21e-4e4e-459f-99c1-4882fd6787f6)

**![Distribution of Word count vs text length per label category]![image](https://github.com/user-attachments/assets/e90107d8-1de5-44ef-98f1-26b655c0474c)

**---
![Correlation Heatmap of Likes, Retweets & Replies]![image](https://github.com/user-attachments/assets/7a5f4648-b64a-4551-90b2-9d82bfbb1989)

![PairPlot of numeric features]![image](https://github.com/user-attachments/assets/984557b1-41a4-4fda-a17d-9d812cbcb7cc)

---

## Machine Learning Models
> **Machine Learning (ML)** is a branch of Artificial Intelligence (AI) that enables computers to learn from data and improve their performance over time without being explicitly programmed. In ML, algorithms identify patterns in data, make predictions, and adapt to new information.

**Base models**
The pipeline contains 6 default models ie :
- Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- Neural Network
---
### Training the models
- 'Training Logistic Regression...'
- 'Training Linear SVM...'
- 'Training Random Forest...'
- 'Training Gradient Boosting...'
- 'Training Neural Network...'
> {'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
 'Linear SVM': LinearSVC(class_weight='balanced', random_state=42),
 'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
 'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)}
---
### Evaluate the base models
- 'Evaluating Logistic Regression...'
- 'Evaluating Linear SVM...'
- 'Evaluating Random Forest...'
- 'Evaluating Gradient Boosting...'
- 'Evaluating Neural Network...'

  - Best model: Logistic Regression with F1 score: 0.6609
  - Model: Logistic Regression
  - Accuracy: 0.6742
  - F1 Score: 0.6609
  ![image](https://github.com/user-attachments/assets/f58d5685-3719-40da-891c-c4838c496c04)

  - Model: Linear SVM
  - Accuracy: 0.6726
  - F1 Score: 0.6592
    
  ![image](https://github.com/user-attachments/assets/d704cf14-d425-4bd8-8329-57688e685fec)

   - Model: Random Forest
  - Accuracy: 0.6224
  - F1 Score: 0.6000
    
  ![image](https://github.com/user-attachments/assets/c7791ae2-fd5f-4e52-8f97-9a7e645db806)

  - Model: Gradient Boosting
  - Accuracy: 0.6370
  - F1 Score: 0.6293
    
  ![image](https://github.com/user-attachments/assets/98066089-e403-4541-a344-8cad8aa3f4ce)

  - Model: Neural Network
  - Accuracy: 0.6408
  - F1 Score: 0.6211
    
  ![image](https://github.com/user-attachments/assets/807889e9-4cc2-45f8-9a54-ca414c6a7e84)

> The best base model on F1_score is Logistic classifier with F1 score: 0.6609. However the f1_score is average meaning the model was classifying fairly okay. The best way forward is to perform a hyperparameter search on the base models to get the best performing parameters for the models ensuring the best F1 score.

---
### Hyperparameter tuning
Above we can see the models performed averagely from the average accuracy and F1 score. We attempt to tune the hyperparameters to get the parameters that will give us the best F1 score. The default cross validation search method for our pipeline is a GridsearchCV.

#### Support Vector Machine

- **Best parameters for Linear SVM: {'C': 0.1, 'class_weight': 'balanced', 'max_iter': 1000}**
- 'Best f1_macro score': 0.6564
- 'Best model Accuracy': 0.6793
> The best parameters for the Linear Support vector machine still gives an average F1 score and accuracy but it is not higher than the best base model.

#### Logistic Classifier
- **Best parameters for Logistic Regression: {'C': 1, 'max_iter': 100, 'solver': 'liblinear'}**
- 'Best f1_macro score': 0.6562
- 'Best model Accuracy': 0.6780
> This is the best performing model overall with an f1 score of 65.81%. This is however not a very reliable model with an accuracy of 65.72%. We try to ensemble the models to see if performance will increase

---

### Ensembling the best performimg models
We attempt to ensemble the best performing models to see if they will perform better by covering up each others weaknesses.

--- Ensemble Evaluation ---
- 'Ensemble Accuracy': 0.6586141131595677
- 'Ensemble Macro F1 Score': 0.6509557693261269
- 'Ensemble Classification Report': {'Hate': {'precision': 0.5433017591339648, 'recall': 0.668053244592346, 'f1-score': 0.5992537313432836, 'support': 1202.0}, 'Not_Hate': {'precision': 0.7607913669064749, 'recall': 0.6527777777777778, 'f1-score': 0.7026578073089701, 'support': 1944.0}, 'accuracy': 0.6586141131595677, 'macro avg': {'precision': 0.6520465630202199, 'recall': 0.6604155111850619, 'f1-score': 0.6509557693261269, 'support': 3146.0}, 'weighted avg': {'precision': 0.6776945746170416, 'recall': 0.6586141131595677, 'f1-score': 0.6631499562883868, 'support': 3146.0}}

> The ensembled model did not perform better than the logistic regression on f1 score hence not suitable for our problem.
![image](https://github.com/user-attachments/assets/0abae943-a627-45da-a7e5-c1bb05733023)

---
### Deep Learning Models
- Since our traditional machine learning models are underperforming—particularly in correctly classifying the minority class—we will now explore deep learning approaches using Hugging Face Transformers. Transformers are state-of-the-art models in Natural Language Processing (NLP) that have demonstrated superior performance in a variety of text classification tasks, including those involving imbalanced and multilingual data.

> Why Use Hugging Face Transformers?

**The main advantages of using Hugging Face transformers for this project include:**
- 'Multilingual Understanding': Our dataset contains tweets in various languages, including English and local dialects such as Sheng (a Swahili-English hybrid spoken in Kenya).
Traditional models struggle with such linguistic diversity, but transformers like XLM-RoBERTa are trained on hundreds of languages and are capable of understanding context in multilingual settings.
- 'Contextual Embeddings': Unlike classical methods like TF-IDF, which treat words independently, transformers generate contextual embeddings. This means that the meaning of a word is understood in relation to surrounding words, which is critical for nuanced tasks like hate speech detection.
- 'Transfer Learning': Transformers are pre-trained on massive corpora and can be fine-tuned on smaller datasets. This makes them ideal for our use case, where the data size (~9700 tweets) may not be sufficient to train deep models from scratch.

**Microsoft/deberta-v3-base**
- The microsoft/deberta-v3-base is a 'pretrained transformer-based large language model from Microsoft', part of the DeBERTa (Decoding-enhanced BERT with disentangled attention) family. It is a highly optimized alternative to BERT and RoBERTa, designed to improve both accuracy and efficiency in NLP tasks.

**Key Features of DeBERTa V3**
- 'Disentangled Attention Mechanism' - Separates content and position embeddings during attention, improving contextual understanding.
- 'Enhanced Mask Decoder' - Improves the masked language modeling objective by using a more refined decoding strategy during pretraining.
- 'Better Generalization' - Achieves higher accuracy than BERT and RoBERTa on various NLP benchmarks.
- 'Efficient' - It provides better performance with fewer parameters compared to older models.

**[1458/1458 20:08, Epoch 6/6]**

| Step | Loss     | Precision | Recall   | F1 Score |
|------|----------|-----------|----------|----------|
| 500  | 0.476400 | 0.560481  | 0.704791 | 0.695103 |
| 1000 | 0.318600 | 0.774572  | 0.668212 | 0.663774 |
![image](https://github.com/user-attachments/assets/2e58f25b-9b91-4a4a-9995-52a306589c9c)

📊 Train Accuracy: 0.7970
📊 Train F1 Score: 0.7907
✅ Validation Accuracy: 0.7048
✅ Validation F1 Score: 0.6951
[[511 262]
 [311 857]]

> This model has an accuracy score of 70.48% and an F1 score of 69.41%. This is a good model. It is classyfying quite better than the first model due to a higher f1 score and the higher accuracy. We save this model in preparation for deployment.

> **deberta_base_hatespeech_5 was the best model hence it is saved for the next step ie Deployment**
---
# Conclusion and Recommendations

## 📌 Conclusion

### 🔍 Model Performance

**DeBERTa Model** clearly outperforms the **Logistic Classifier** across all major metrics:

- ✅ Higher accuracy and F1 score on validation.
- ✅ Better confusion matrix with fewer misclassifications.
- ✅ Lower overfitting risk (closer train and validation scores).

**Logistic Classifier** likely underfits or is not well-optimized, especially struggling with the **"Hate"** classification due to high false negatives.

---

### 🕒 Time of Day Trend

- **Midday (10 AM – 2 PM)** is the most active window for both *Hate* and *Not_Hate* tweets.
- Strategic timing for:
  - 🔹 Moderation
  - 🔹 Counter-speech
  - 🔹 Promotional posting

**Evening resurgence (7–9 PM)** of Hate tweets may indicate:
- Reactions to daily events.
- Increased posting after work hours.

---

### 📆 Yearly Tweet Trend

- Engagement follows a **cyclical pattern**, peaking in select years — possibly aligned with national or global events.
- Peaks may correspond to:
  - 🗳️ Election periods
  - 📰 Socio-political developments

- **Sharp declines** after peak years (e.g., 2018 → 2019, 2024 → 2025) suggest reduced discourse or event-driven fatigue.

---

### 🧑‍💼 Politician Engagement Score

#### 🚀 Top Performing Politicians
- **Betty Maina** and **John Kiarie** exceed **10,000** average engagement scores.
- Indicates viral content or high public interest (controversy or trending issues).

#### ⚖️ Moderate Engagement Group
- **Alice Nganga**, **Noordin Haji**, and **Didmus Barasa**: Scores between **1,500–2,500**.

#### ⬇️ Lower Engagement Politicians
- **Martha Koome**, **Japheth Koome**, **Kalonzo Musyoka**, and **Johnson Sakaja**: Around **1,000 or below**.

> The inclusion of "unknown" implies some tweets couldn't be attributed to a specific politician but still saw moderate engagement.

---

### 🗣️ Top Discussed Politicians

| Politician         | Tweet Volume | Hate Tweet % | Insight                                                       |
|--------------------|--------------|--------------|---------------------------------------------------------------|
| **William Ruto**   | > 1,500      | ~45%         | Highly polarized public sentiment                             |
| **Raila Odinga**   | ~540         | Balanced     | High volume, mix of hate and non-hate                         |
| **Oscar Sudi**     | High         | >50%         | Strong public backlash or controversy                         |
| **Rigathi Gachagua**| High        | >50%         | Similar trend to Oscar Sudi                                   |
| **Fred Matiang’i** | Moderate     | Lower        | Sustained public interest                                     |
| **Kithure Kindiki**| Moderate     | Lower        | Sustained public interest                                     |
| **Rachel Ruto**    | Low          | Low          | Minimal public discourse                                      |
| **Martha Koome**   | Low          | Low          | Minimal public discourse                                      |

---

## 💡 Recommendations

### 🧠 Model Utilization and Enhancement

- **✅ Prioritize DeBERTa for Deployment**: Given superior accuracy, F1 score, and balanced confusion matrix.
- **🔍 Monitor Overfitting Trends**: Ensure stability with future data, drift, or evolving hate speech patterns.

---

### 🛡️ Moderation and Platform Strategy

- **📈 Focus Moderation During Peak Hours (10 AM – 2 PM)**:
  - Deploy more resources
  - Automate flagging tools
  - Introduce timely interventions

- **🕑 Counter-Speech Timing**:
  - Schedule awareness campaigns or positive content
  - Ideal windows: **10 AM–2 PM** and **7–9 PM**

---

### 🗓️ Event-Driven Planning

- **🧭 Prepare for Election and Crisis Peaks**:
  - Use past data to predict future flashpoints
  - Scale moderation and monitoring accordingly

- **🧹 Post-Event Cooldown Monitoring**:
  - Leverage low-activity periods to refine models
  - Reassess baseline hate speech levels

---

### 📊 Engagement Strategy

- **Engagement ≠ Volume**: High engagement may reflect strong (positive/negative) sentiment.
- Politicians with high engagement should evaluate the **quality** of interactions, not just quantity.
- Lower engagement profiles could:
  - Reassess communication strategy
  - Boost public engagement with targeted content

---

### 🧾 Top Discussed Politicians Strategy

- Profiles like **William Ruto**, **Raila Odinga**, and **Oscar Sudi** dominate discussions, including hate speech.
- Hate speech is disproportionately concentrated on a few figures — indicating:
  - 🧨 Targeted harassment
  - ⚖️ Divisive public perception

> Understanding the *context* (e.g., campaigns, scandals) is key to strategic intervention.

---

## 🚀 Deployment

Deployment is the process of making your application live and accessible. It includes:

- 📦 Packaging
- 🧪 Testing
- ⚙️ Configuring the software for production

## 🚀 Deployment Instructions

### 📁 Step 1: Create the FastAPI App

Create a file named `app.py` — this is your application entry point.

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./final_model/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

label_map = {0: "not_hatespeech", 1: "hatespeech"}

@app.post("/Hatespeech_detection-csv")
async def Hatespeech_detection_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print("📄 CSV Loaded:", df.head())  # Debug print

        if "text" not in df.columns:
            return JSONResponse(content={"error": "CSV must contain a 'text' column."}, status_code=400)

        inputs = tokenizer(df["text"].tolist(), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        df["Detection_result"] = [label_map[p] for p in preds]

        print("✅ Predictions:", df["Detection_result"].value_counts().to_dict())  # Debug print
        return df[["text", "Detection_result"]].to_dict(orient="records")

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
```

---

### ▶️ Step 2: Run the App

Use Uvicorn to start the FastAPI server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

### 📫 Access the API

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Endpoint**: `POST /Hatespeech_detection-csv`

Upload a CSV with a `text` column to receive predictions.

Example CSV format:

```csv
text
"This is an example tweet."
"Another sample input for hate speech detection."
```
---

### 📦 Dependencies

Install required packages with:

```bash
pip install fastapi uvicorn transformers pandas torch
```
---

## 🤝 Contribution

This project supports online safety efforts by helping moderation teams, NGOs, and civic tech initiatives flag and respond to political hate speech in real-time.

---

