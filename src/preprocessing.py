#Handles text cleaning and normalization.

# src/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and preprocess complaint text.

    Steps:
    - Lowercase
    - Remove special characters & numbers
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


#Purpose: Prepare text for NLP models consistently.
