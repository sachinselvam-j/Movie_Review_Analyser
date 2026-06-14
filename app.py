import streamlit as st
import pickle
import spacy

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:900;
    color:#FF4B4B;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#B0B0B0;
    margin-bottom:30px;
}

.card{
    background-color:#1E293B;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(255,255,255,0.05);
}

.feature-box{
    background-color:#1E293B;
    padding:20px;
    border-radius:15px;
}

.metric-container{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ---------------------------------------------------
# PREPROCESS FUNCTION
# ---------------------------------------------------

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

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze movie reviews using NLP, TF-IDF and Machine Learning</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    review = st.text_area(
        "✍️ Enter Movie Review",
        height=250,
        placeholder="Example: This movie was fantastic. Great acting and amazing story..."
    )

    predict = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="feature-box">
    <h3>📊 Features</h3>

    ✅ spaCy NLP Processing

    ✅ TF-IDF Vectorization

    ✅ Naive Bayes Classification

    ✅ Sentiment Prediction

    ✅ Confidence Score
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict:

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        clean_review = preprocess(review)

        review_vector = tfidf.transform([clean_review])

        prediction = model.predict(review_vector)[0]

        confidence = model.predict_proba(review_vector).max() * 100

        st.divider()

        if prediction == "positive":

            st.success("😊 Positive Review")

        else:

            st.error("😞 Negative Review")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

# ---------------------------------------------------
# PROJECT INFO
# ---------------------------------------------------

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Algorithm", "Naive Bayes")

with c2:
    st.metric("Vectorizer", "TF-IDF")

with c3:
    st.metric("NLP Library", "spaCy")