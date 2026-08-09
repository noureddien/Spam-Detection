import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    
    # Keep alphanumeric words, remove punctuation only
    y = [i for i in words if i.isalnum()]
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)