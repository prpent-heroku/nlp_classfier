# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:36:17 2022

@author: velua

Editors:Pravallika
"""
# !pip install textblob
#!pip install wordcloud


import streamlit as st 
import joblib,os
import spacy
import pandas as pd
spacy.cli.download("en")
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import emoji
from textblob import TextBlob
import re

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
 # wordcloud function
def show_wordcloud(data, title = None):
    from wordcloud import WordCloud,STOPWORDS
    plt.figure(figsize=(12,12))
    wc = WordCloud(max_words=1000, 
               min_font_size=10,
               height=600,
               width=1600,
               background_color='black',
               contour_color='black',
               colormap='plasma',
               repeat=False,
               stopwords=STOPWORDS).generate(' '.join(data))

    plt.title("42 News Categories' Wordcloud", size=15, weight='bold')
    plt.imshow(wc, interpolation= "bilinear")
    plt.axis('off')
    st.pyplot(plt)
    
def plot_by_category(data):
    plt.figure(figsize=(12,8))
    count = data.value_counts()
    sns.barplot(x=count.index, y=count)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
def Visualization():
    st.info("Dataset Information:")
    News = pd.read_json("News_Category_Dataset_v3.json",lines=True)
    st.write(News.head())
    st.write(News.shape)
    News['Headline_Combined']= News["headline"] +"" +News["short_description"]
    plotChoice = st.sidebar.selectbox("Select the plot you want to see",["-- Choose One --","Number of headlines for each category","Top 10 News Categories","Display Word Cloud With Headlines"])
    if plotChoice=="Number of headlines for each category":
        st.info("Number of headlines for each category")
        plot_by_category(News['category'])
    else:
        st.write("under development")
    #show_wordcloud(News['Headline_Combined'])
    
   
def clean_news(text):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())    


def home_page_module():
    st.title("News Headlines Classification")
    image = Image.open("Projectimage.jpeg")
    st.image( image , caption='** Classify a News** ' )
    st.subheader("About Application:")
    st.write("The objective of this project is to create a web application that can consume news headline sentences and provide a classification label to understand what the topic of the news article is and suggest an overall article sentiment.")
    
    st.write("Usefullness: News is all around us through the internet or Television keeping us up to date on the daily events as they happen. There are over 5000 news articles published just in the United States every day [1]. The amount of News articles available to consumers can be overwhelming and requires a process to classify them, this helps the reader to focus on the articles that are of interest to them. This process filters the news articles to an extent but in addition to news topic classification, we would like to provide a suggested sentiment on a news article such that the consumer has an additional piece of information before reading the article and potentially decide on whether to read the article.")
   
    st.write("Build By Fall 2022 NLP Team: Agalya Velusamy, Pravallika Pentapati and Srikanth Bolishetty")
    #st.image(show_wordcloud(text))
# def normalize_document(doc):
#  # lower case and remove special characters\whitespaces
#      doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
#      doc = doc.lower()
#      doc = doc.strip()
#      stop_words = stopwords.words('english')
#      tokens = nltk.word_tokenize(doc)
#      # remove stopwords
#      filtered_tokens = [token for token in tokens if token not in stop_words]
#      # remvoe punctuation
#      filtered_tokens = [word.lower() for word in filtered_tokens if word.isalpha()]
#      # Lemmatization
#      lemmatizer = WordNetLemmatizer()
#      filtered_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
#      doc = ' '.join(filtered_tokens)
     
     #return doc
def nlp_task():
    # Applying TFIDF
    # vectorizer = TfidfVectorizer(ngram_range = (3,3))
    # tfidf_model= vectorizer.fit_transform(text)
    # scores = (tfidf_model.toarray())
    # features = (vectorizer.get_feature_names())
    st.title("Natural Language Processing")
    raw_text = st.text_area("Enter News Here","Type Here")
    nlp_task = ["Select NLP Task","Tokenization","Lemmatization","Named Entity Recognition","Parts Of Speech Tags","Sentiment Analysis"]
    task_choice = st.selectbox("Choose NLP Task",nlp_task)
    if st.button("Analyze"):
         st.info("Original Text::\n{}".format(raw_text))

         docx = nlp(raw_text)
         if task_choice == 'Tokenization':
             st.caption("Result:")
             result = [token.text for token in docx ]
             st.text(result)
         elif task_choice == 'Lemmatization':
             result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
             st.text(result)
         elif task_choice == 'Named Entity Recognition':
             for entity in docx.ents:
                 result=[f"{entity.text:-<{20}}{entity.label_:-<{20}}{str(spacy.explain(entity.label_))}"] 
                 st.text(result)
         elif task_choice == 'Parts Of Speech Tags':
               result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]
               st.json(result)  
         elif task_choice=='Sentiment Analysis':
             analysis = TextBlob(clean_news(raw_text))
             result = analysis.sentiment.polarity
             if result > 0.0:
                 custom_emoji = ':smile:'
                 st.write("smile",emoji.emojize(custom_emoji))
             elif result < 0.0:
                 custom_emoji = ':disappointed:'
                 st.write("sad",emoji.emojize(custom_emoji))
             else:
                 st.write("neutral",emoji.emojize(':expressionless:'))
             st.info("Polarity Score is:: {}".format(result))
                
                
            
     
         


    
  
def  prediction_task():
     st.info("News Category Prediction")
     user_headline=st.text_area('Enter your news headline to predict news category')
     if(st.button('Predict')):
         st.success("In Process")      
def main():
    if 'loggedIn' not in st.session_state:
        st.session_state.loggedIn = False
    
    menu = ["Home","NLP Task" ,"Prediction Task","Statistical Plots"]
    choice = st.sidebar.selectbox("Menu",menu )

    ## HOME PAGE
    if choice == "Home":
        home_page_module()
    elif choice =="NLP Task":
        nlp_task()
    elif choice=="Prediction Task": 
        prediction_task()
#     elif choice=="Statistical Plots":
# #        Visualization()
    else:
        st.write("Under Development")
if __name__ == '__main__':
    main()        
