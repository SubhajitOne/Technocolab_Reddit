import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
sid = SentimentIntensityAnalyzer()
def main():
    
    st.title("Predicting the Popularity of Reddit Posts")
    f=open("model.pkl", "rb")
    model = pickle.load(f)
    v= pickle.load(f)
    st.header('Input Comment')
    text = st.text_input("Input text")
    pred = model.predict(v.transform([text]))    
    st.header('Output')
    st.write('The score is:', str(pred))

    st.write('The Polarity scores are:','\n',sid.polarity_scores(text)) 
    st.write('---')



    
    
          
if __name__ == '__main__':
    main()
