import streamlit as st
import pickle
import spacy

# Load spaCy model
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# App title
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and check whether it is Positive or Negative."
)

def preprocess(text):

    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    return " ".join(tokens)

review = st.text_area("Enter Movie Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        clean_review = preprocess(review)

        review_vector = tfidf.transform([clean_review])

        prediction = model.predict(review_vector)[0]

        if prediction == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")