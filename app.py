"""
# My first app
Here's our first attempt at using data to create a table:
"""



import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix# Performance Metrics  
from wordcloud import WordCloud,STOPWORDS
from nltk import SnowballStemmer

from sklearn.model_selection import train_test_split # Split Data 
from imblearn.over_sampling import SMOTE # Handling Imbalanced

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

    df['content'] = df['content'].str.replace('https\S+', ' ', case=False)
    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].str.replace('@\S+', ' ', case=False)
    df['content'] = df['content'].str.replace('#\S+', ' ', case=False)
    df['content'] = df['content'].str.replace("\'\w+", ' ', case=False)
    df['content'] = df['content'].str.replace("[^\w\s]", ' ', case=False)
    df['content'] = df['content'].str.replace("\s(2)", ' ', case=False)
    regexp = RegexpTokenizer('\w+')
    df['content_token']=df['content'].apply(regexp.tokenize)
    #
    stopwords = nltk.corpus.stopwords.words("indonesian")
    # Remove stopwords
    df['content_token'] = df['content_token'].apply(lambda x: [item for item in x if item not in stopwords])
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['stemmed'] = df['content_token'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
    st.dataframe(df) 

    #wordcloud positif
    df_p=df[df['sentiment']==1]
    all_words_lem = ' '.join([word for word in df_p['content']])
    wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_lem)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    #wordcloud negatif
    df_p=df[df['sentiment']==-1]
    all_words_lem = ' '.join([word for word in df_p['content']])
    wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_lem)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    #wordcloud netral
    df_p=df[df['sentiment']==0]
    all_words_lem = ' '.join([word for word in df_p['content']])
    wordcloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(all_words_lem)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    # return df

    X = df['content']
    y = df['sentiment']
    tfid = TfidfVectorizer()
    X_final = tfid.fit_transform(X)

    smote = SMOTE()
    x_sm,y_sm = smote.fit_resample(X_final,y)

    X_train , X_test , y_train , y_test = train_test_split(x_sm , y_sm , test_size=0.1,random_state=3)

    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train,y_train)
    RandomForestClassifier()

    random_forest_classifier_prediction =  random_forest_classifier.predict(X_test)
    accuracy_score(random_forest_classifier_prediction,y_test)
    st.write(accuracy_score)
    
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

    
