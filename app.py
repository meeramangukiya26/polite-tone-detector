import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Politeness Detection System",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* Main Container */
.main-box {
    background-color: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-top: 30px;
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 40px;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 14px;
    border: 2px solid #d1d5db;
    padding: 15px;
    font-size: 18px;
    background-color: #f9fafb;
}

/* Button */
.stButton button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(to right, #2563eb, #4f46e5);
    color: white;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

/* Button Hover */
.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #1d4ed8, #4338ca);
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 35px;
    color: #6b7280;
    font-size: 14px;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 25px;
}

.polite {
    background-color: #dcfce7;
    color: #166534;
}

.nonpolite {
    background-color: #fee2e2;
    color: #991b1b;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">🧠 AI Politeness Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Natural Language Processing based application to identify polite and non-polite communication</div>',
    unsafe_allow_html=True
)

# ---------------- MAIN BOX ----------------
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Input
user_input = st.text_area(
    "Enter Your Sentence",
    placeholder="Example: Could you please send me the report?"
)

# Analyze Button
if st.button("Analyze Sentence Tone"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a sentence.")

    else:

        # Transform Text
        text = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(text)

        # Result
        if prediction[0] == "polite":

            st.markdown(
                '<div class="result-box polite">✅ Polite Tone Detected</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-box nonpolite">❌ Non-Polite Tone Detected</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer">Powered by NLP • TF-IDF • Logistic Regression • Streamlit</div>',
    unsafe_allow_html=True
)