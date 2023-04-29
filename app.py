"""
# My first app
Here's our first attempt at using data to create a table:
"""

# import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit as st
# import seaborn as sns



  
# st.header('Single File Upload')
# # uploaded_file = st.file_uploader('Upload a file')

# df = st.file_uploader("upload file", type={"csv", "txt"})
# if df is not None:
#     opini_df = pd.read_csv(df)
# st.write(opini_df)
# # df = pd.read_csv(uploaded_file)
# # st.write(df)

# # val_count  = df['sentiment'].value_counts()
# df_new = opini_df[['Year', 'Month', 'sentiment']]
# fig = plt.figure(figsize=(10,5))
# result = df_new.groupby(['sentiment']).size()
 
# # plot the result
# sns.barplot(x = result.index, y = result.values)
# st.pyplot(fig)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import re
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier

# Mengimpor dataset ulasan aplikasi Play Store
# df = pd.read_csv('outputt.csv')
# # Memilih data ulasan yang menggunakan bahasa Indonesia
# df = df[df['language']=='id']
# # Membersihkan dataset
# df.dropna(inplace=True)
# df.drop_duplicates(subset=['Translated_Review'], inplace=True)
# df.reset_index(drop=True, inplace=True)

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
# from sastrawi.stemmer import StemmerFactory
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

    st.dataframe(df) 

    # Mengubah format kolom tanggal menjadi datetime
    df['at'] = pd.to_datetime(df['at']).dt.date

    # Memfilter data berdasarkan tanggal
    mask = (df['at'] >= start_date) & (df['at'] <= end_date)
    df = df.loc[mask]

    # Mengambil kolom teks ulasan dan rating
    df = df[['content', 'score']]

    return df


def predict_sentiment(text, model):
    # Melakukan preprocessing teks
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # # Mengubah teks menjadi vektor TF-IDF
    # vectorizer = TfidfVectorizer(stop_words='indonesian')
    # text_vec = vectorizer.fit_transform([text])

    
     # Melakukan stemming menggunakan Sastrawi
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    # text = stemmer.stem(text)

    # # Mengubah teks menjadi vektor TF-IDF
    # vectorizer = TfidfVectorizer(stop_words='indonesian')
    # text_vec = vectorizer.fit_transform([text])

    # Mengubah teks menjadi vektor TF-IDF
    vectorizer = TfidfVectorizer(stop_words=None)
    text_vec = vectorizer.fit_transform([text])

    # Melakukan stemming pada teks
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_text = stemmer.stem(text)

    # Melakukan prediksi dengan model random forest
    prediction = model.predict(text_vec)

    return prediction[0]

# Muat model random forest dari file
model = joblib.load('modell.pkl')

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
    # df = scrape_google_play(keyword, country, start_date, end_date)
    df = scrape_google_play(country, start_date, end_date)
# Menambahkan kolom sentimen pada dataframe
    df['sentiment'] = df['content'].apply(lambda x: predict_sentiment(x, model))

    st.write('Hasil Analisis Sentimen:')
    st.write(df['sentiment'].value_counts())
    # Memeriksa apakah ada ulasan dalam
    # df = predict_sentiment(text, model)
    
