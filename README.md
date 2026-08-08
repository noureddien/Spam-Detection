# SMS Spam Detection System

An end-to-end Machine Learning pipeline and web application that classifies text messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and Multinomial Naive Bayes model.

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

