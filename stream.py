import streamlit as st
import pickle as pkl
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download("stopwords")
cv=pkl.load(open("cv.pkl","rb"))
models={    
    "DT":pkl.load(open("DT.pkl","rb")),
    "RF":pkl.load(open("RF.pkl","rb")),
    "KNN":pkl.load(open("KNN.pkl","rb")),
    "SVM":pkl.load(open("SVM.pkl","rb")),
    "LG":pkl.load(open("LG.pkl","rb")),
    "GB":pkl.load(open("GB.pkl","rb")),     
}
st.markdown("<div style='background-color: lightgray; padding:20px'/>",unsafe_allow_html=True)
st.title("RESTAURANT REVIEW")
st.markdown("<div style='background-color: lightgray; padding:20px'/>",unsafe_allow_html=True)
st.write("<font color='red'><b> Positive or Negative</b></font>",unsafe_allow_html=True)
modelname=st.selectbox("Choose The Model",list(models.keys()))
model = models[modelname]
user=st.text_area("Enter your review")
if st.button("Predict"):
    if user.strip()==" ":
        st.warning("please enter your review")
    else: 
        ps = PorterStemmer()
        review = re.sub("[^a-zA-Z]"," ", user)
        review = review.lower()
        review = review.split()
        all_stopwords = stopwords.words("english")
        all_stopwords.remove("not")
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = " ".join(review)
        review_vector=cv.transform([review]).toarray()
        pred=model.predict(review_vector)[0]
        if pred==1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")
   
        
