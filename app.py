import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict(text):
    vec = vectorizer.transform().toarray()
    return model.predict(vec)[text][0]

st.title("Sentiment Analysis App")

user_input = st.text_input("Enter text")

if st.button("Predict"):
    result = predict(user_input)
    st.write("Sentiment:", result)

# Run: streamlit run app.py