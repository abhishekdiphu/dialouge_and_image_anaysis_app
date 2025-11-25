"""
This module provides a function to analyze the sentiment of a given text using
an external sentiment analysis API powered by IBM Watson.

Functions:
    sentiment_analyzer(text_to_analyse): Analyzes the sentiment of the given text
    using the Watson Sentiment Prediction API and returns the sentiment label and score.

Usage:
    - Import the module and call the `sentiment_analyzer` function with a text string.
    - The function will return a dictionary containing the sentiment label
     ('positive', 'neutral', 'negative') and a sentiment score (a float value).

Example:
    result = sentiment_analyzer("I love this product!")
    print(result)  # Output: {'label': 'positive', 'score': 0.95}
"""

import json
import requests
import transformers



# sentiment_analysis.py
from transformers import pipeline

# Load Hugging Face BERT model once (for efficiency)
sentiment_model = pipeline("sentiment-analysis")

def analyze_text(text: str):
    """Analyze sentiment and return label + confidence."""
    if not text.strip():
        return {"error": "Empty text provided"}
    
    result = sentiment_model(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }

