"""
# My first app
Here's our first attempt at using data to create a table:
"""



import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google_play_scraper import Sort, reviews_all


def scrape_google_play(country, start_date, end_date):
    # Konfigurasi scraping
    reviews = reviews_all(
        'id.qoin.korlantas.user',
        # keyword,
        lang='id',
        country=country,
        sort=Sort.MOST_RELEVANT,
        filter_score_with=None,
        # continuation_token=None,
        # pagination=True
    )

    # Mengubah hasil scraping ke dalam dataframe
    df = pd.DataFrame(reviews)

    

    # Mengubah format kolom tanggal menjadi datetime
    df['at'] = pd.to_datetime(df['at']).dt.date

    # Memfilter data berdasarkan tanggal
    mask = (df['at'] >= start_date) & (df['at'] <= end_date)
    df = df.loc[mask]

    # Mengambil kolom teks ulasan dan rating
    df = df[['content', 'score']]

    sentimen = []
    for index, row in df.iterrows():
        if row['score'] > 3 :
            sentimen.append(1)
        elif row['score'] == 3:
            sentimen.append(0)
        else:
            sentimen.append(-1)
    df['sentiment'] = sentimen
    st.dataframe(df) 
    # return df


def predict_sentiment(text, model):
    # Melakukan preprocessing teks
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    

    


# Muat model random forest dari file
model = joblib.load('35-rf.pkl')

# Tampilan aplikasi Streamlit
st.title('Analisis Sentimen Ulasan Aplikasi di Playstore')

# Input form
# keyword = st.text_input('Kata Kunci Aplikasi')
country = st.selectbox('Negara', ['id', 'us', 'gb'])
start_date = st.date_input('Tanggal Awal')
end_date = st.date_input('Tanggal Akhir')

if st.button('Analyze'):
    # result = predict_sentiment
    # Memanggil fungsi untuk scraping data

    df = scrape_google_play(country, start_date, end_date)

    
