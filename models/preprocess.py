import re
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    
    # Replace URLs, phone numbers, and $ before filtering
    text = re.sub(r'http\S+|www\.\S+', 'linktoken', text)
    text = re.sub(r'\b\d{7,11}\b', 'phonetoken', text)
    text = re.sub(r'\$\d+', 'moneytoken', text)
    
    words = nltk.word_tokenize(text)
    
    # Keep alphanumeric words + token placeholders
    y = [i for i in words if i.isalnum()]
    y = [i for i in words if i.isalnum() or i in ['$']]
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)