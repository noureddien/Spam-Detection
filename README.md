# SMS Spam Detection System

An end-to-end Machine Learning pipeline and web application that classifies text messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and Multinomial Naive Bayes model.
<img width="1310" height="469" alt="image" src="https://github.com/user-attachments/assets/d30232e6-0dd2-40a9-9ca4-8c9c5ffe7639" />
<img width="1339" height="869" alt="image" src="https://github.com/user-attachments/assets/c7060966-4394-4dc6-829a-63dd2fe8e6c1" />

---

## Project Structure
```text
├── weights/
│   ├── best_model.pkl          # Serialized Multinomial Naive Bayes classifier
│   └── vectorizer.pkl          # Serialized TF-IDF Vectorizer
├── models/
│   ├── preprocess.py           # Text cleaning, tokenization, & stemming pipeline
│   └── predict.py              # Local inference test script
├── notebook/
│   └── eda_and_training.ipynb  # Data exploration, model training, & evaluation
├── app.py                      # Streamlit web application interface
├── requirements.txt            # Python dependencies (conda)
└── piprequirements.txt         # Python dependencies (pip)
```
## Setup & Installation

1. **Clone repo:**

   git clone <https://github.com/noureddien/Spam-Detection.git>

2. **Install dependencies:**

    `(conda)` conda install --file requirements.txt | OR | 
    `(pip)`   pip install -r piprequirements.txt

3. **Run local inference test:**

    python models/predict.py

## Web Interface (Streamlit)

This project includes an interactive web application built with Streamlit for real-time spam classification and model performance monitoring.

### Features
* **Real-time Classification:** Input custom SMS or email text to receive instant "Spam" or "Ham" predictions.
* **Model Dashboard:** View evaluation metrics (Accuracy: 97.49%, Precision: 98.17%, Recall: 81.68%, ROC-AUC: 0.9901).
* **Data Visualizations:** Interactive dataset class distribution pie chart, saved confusion matrix, and message length distribution analysis.


## Features & Implementation

1. **Text Preprocessing** (models/preprocess.py): Converts input text to lowercase, tokenizes it using NLTK, removes non-alphanumeric noise/stopwords/punctuation, and applies Porter Stemming.

2. **Feature Extraction**: Transforms preprocessed text into TF-IDF vector representations.

3. **Model Inference**: Classifies vectorized input in real-time to minimize false positives.


 ## Model Evaluation

|            Model            |   Accuracy    |   Precision   |   Recall   |  F1-Score  |    Status    |
| :-------------------------: | :-----------: | :-----------: | :--------: | :--------: | :----------: |
| **Multinomial Naive Bayes** |   **0.9749**  | **0.9817**    | **0.8168** | **0.8917** | **Selected** |
|   Logistic Regression       |     0.9555    |    1.0000     |   0.6489   |   0.7871   |   Rejected   |
|   Support Vector Machine    |     0.9758    |   0.9732      |   0.8321   |   0.8971   |   Alternate  |

