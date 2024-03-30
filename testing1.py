from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

# Set up Streamlit sidebar options with emojis
rad=st.sidebar.radio("Exploration",["🏠 Welcome","🚀 Explore Spam or Ham","😄 Dive into Sentiments","😰 Stress Evaluator","🚫 Hate and Offensive Detector","😏 Sarcasm Sleuth"])

# Home Page
# Home Page
if rad == "🏠 Welcome":
    st.title("Text Analysis Toolbox: Spam to Sarcasm Detection")
    st.image("Page12.jpg", width=500)
    #st.image("Page2.jpg", width=200)
    st.markdown(
        """
        <style>
            .app-title {
                color: #428af5;
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .home-section {
                background-color: #428af5;
                padding: 30px;
                border-radius: 15px;
                margin-top: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .home-description {
                font-size: 20px;
                margin-bottom: 30px;
            }
            .options-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .option-card {
                background-color: #67C4D0;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s;
                cursor: pointer;
            }
            .option-card:hover {
                transform: scale(1.05);
            }
            .option-title {
                color: #020A0B;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 5px;  /* Added margin-bottom for spacing */
            }
            .option-text {
                font-size: 16px;
                line-height: 1.6;
            }
            .image-row {
                display: flex;
                justify-content: space-around;
                margin-top: 30px;
            }
            .image-box {
                width: 200px;
                height: 200px;
                overflow: hidden;
                border-radius: 8px;
            }
        </style>
        """
        , unsafe_allow_html=True
    )


    st.markdown("<div class='app-title'>Explore the Power of Text Analysis</div>", unsafe_allow_html=True)
    st.markdown(
    """
    <div class='home-description' style='text-align: center; font-size: 22px; color: #800080; font-family: "Arial", sans-serif;'>
        <p><strong>Welcome to the Text Analysis Hub!</strong> 🚀 Uncover the magic hidden in words. Select an option at left side to begin your linguistic exploration!</p>
    </div>
    """,
    unsafe_allow_html=True
)
    col1, col2 = st.columns(2)
    col1.image("Page3.jpg", width=200, caption='Stress', use_column_width=True)
    col2.image("Page2.jpg", width=200, caption='Spam', use_column_width=True)
    col1.image("Page1.jpg", width=200, caption='Sentimental', use_column_width=True)
    col2.image("Page4.jpg", width=200, caption='Hate Content', use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    #st.markdown("</div>", unsafe_allow_html=True)
   
    options = [
        {"title": "🚀 Explore Spam or Ham", "text": "Detect whether a text is spam or ham."},
        {"title": "😄 Dive into Sentiments", "text": "Detect the sentiment of the text."},
        {"title": "😰 Stress Evaluator", "text": "Detect the stress in the text."},
        {"title": "🚫 Hate and Offensive Detector", "text": "Detect hate and offensive content in the text."},
        {"title": "😏 Sarcasm Sleuth", "text": "Detect whether the text is sarcastic or not."}
    ]

    st.markdown("<div class='options-container'>", unsafe_allow_html=True)
    for option in options:
        st.markdown(f"<div class='option-card'>"
                    f"<div class='option-title'>{option['title']}</div>"
                    f"<div class='option-text'>{option['text']}</div>"
                    f"</div>", 
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



# Function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Spam Detection Prediction
tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt2=tfidf1.fit_transform(txt1)
    return txt2.toarray()

df1=pd.read_csv("Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=LogisticRegression()
model1.fit(x_train1,y_train1)

#Spam Detection Analysis Page
if rad=="🚀 Explore Spam or Ham":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent1=st.text_area("Enter The Text")
    transformed_sent1=transform_text(sent1)
    vector_sent1=tfidf1.transform([transformed_sent1])
    prediction1=model1.predict(vector_sent1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text!!")
        elif prediction1=="ham":
            st.success("Ham Text!!")

#Sentiment Analysis Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Sentiment Analysis Page
if rad=="😄 Dive into Sentiments":
    st.header("Detect The Sentiment Of The Text!!")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2==0:
            st.warning("Negetive Text!!")
        elif prediction2==1:
            st.success("Positive Text!!")

#Stress Detection Prediction
tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()

df3=pd.read_csv("Stress Detection.csv")
df3=df3.drop(["subreddit","post_id","sentence_range","syntax_fk_grade"],axis=1)
df3.columns=["Text","Sentiment","Stress Level"]
x=transform3(df3["Text"])
y=df3["Stress Level"].to_numpy()
x_train3,x_test3,y_train3,y_test3=train_test_split(x,y,test_size=0.1,random_state=0)
model3=DecisionTreeRegressor(max_leaf_nodes=2000)
model3.fit(x_train3,y_train3)

#Stress Detection Page
if rad=="😰 Stress Evaluator":
    st.header("Detect The Stress In The Text!!")
    sent3=st.text_area("Enter The Text")
    transformed_sent3=transform_text(sent3)
    vector_sent3=tfidf3.transform([transformed_sent3])
    prediction3=model3.predict(vector_sent3)[0]

    if st.button("Predict"):
        if prediction3>=0:
            st.warning("Stressful Text!!")
        elif prediction3<0:
            st.success("Not A Stressful Text!!")

#Hate & Offensive Content Prediction
tfidf4=TfidfVectorizer(stop_words=sw,max_features=20)
def transform4(txt1):
    txt2=tfidf4.fit_transform(txt1)
    return txt2.toarray()

df4=pd.read_csv("Hate Content Detection.csv")
df4=df4.drop(["Unnamed: 0","count","neither"],axis=1)
df4.columns=["Hate Level","Offensive Level","Class Level","Text"]
x=transform4(df4["Text"])
y=df4["Class Level"]
x_train4,x_test4,y_train4,y_test4=train_test_split(x,y,test_size=0.1,random_state=0)
model4=RandomForestClassifier()
model4.fit(x_train4,y_train4)

#Hate & Offensive Content Page
if rad=="🚫 Hate and Offensive Detector":
    st.header("Detect The Hate & Offensive Content In The Text!!")
    sent4=st.text_area("Enter The Text")
    transformed_sent4=transform_text(sent4)
    vector_sent4=tfidf4.transform([transformed_sent4])
    prediction4=model4.predict(vector_sent4)[0]

    if st.button("Predict"):
        if prediction4==0:
            st.exception("Highly Offensive Text!!")
        elif prediction4==1:
            st.warning("Offensive Text!!")
        elif prediction4==2:
            st.success("Non Offensive Text!!")

#Sarcasm Detection Prediction
tfidf5=TfidfVectorizer(stop_words=sw,max_features=20)
def transform5(txt1):
    txt2=tfidf5.fit_transform(txt1)
    return txt2.toarray()

df5=pd.read_csv("Sarcasm Detection.csv")
df5.columns=["Text","Label"]
x=transform5(df5["Text"])
y=df5["Label"]
x_train5,x_test5,y_train5,y_test5=train_test_split(x,y,test_size=0.1,random_state=0)
model5=LogisticRegression()
model5.fit(x_train5,y_train5) 

#Sarcasm Detection Page
if rad=="😏 Sarcasm Sleuth":
    st.header("Detect Whether The Text Is Sarcastic Or Not!!")
    sent5=st.text_area("Enter The Text")
    transformed_sent5=transform_text(sent5)
    vector_sent5=tfidf5.transform([transformed_sent5])
    prediction5=model5.predict(vector_sent5)[0]

    if st.button("Predict"):
        if prediction5==1:
            st.exception("Sarcastic Text!!")
        elif prediction5==0:
            st.success("Non Sarcastic Text!!")
            



