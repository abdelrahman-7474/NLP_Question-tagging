import streamlit as st
import re
from bs4 import BeautifulSoup
import joblib

programming_terms = {
    'javascript', 'java', 'c#', 'php', 'android', 'jquery',
    'python', 'html', 'c++', 'ios', 'mysql', 'css', 'sql',
    'asp.net', 'objective-c', 'ruby-on-rails', '.net', 'c',
    'iphone', 'angularjs', 'arrays', 'sql-server', 'json',
    'ruby', 'r', 'ajax', 'regex', 'xml', 'node.js',
    'asp.net-mvc', 'linux', 'django', 'wpf', 'database', 'swift'
}

def remove_punctuation_preserve_programming_terms(text):
    words = text.lower().split()
    cleaned = [
        word if word in programming_terms else re.sub(r'[^\w\s]', '', word)
        for word in words
    ]
    return ' '.join(filter(None, cleaned))

def remove_html_tags(html):
    return BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)

# --- Cached Loaders ---
@st.cache_resource
def load_model():
    return joblib.load("logistic_ovr_model_51test.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def load_mlb():
    return joblib.load("mlb_transformer.pkl")

model = load_model()
loaded_tfidf = load_vectorizer()
mlb = load_mlb()

st.title(" Welcome to Question Tagging")

user_title = st.text_area(" Enter your title here:")
user_question = st.text_area(" Enter your question here:")

# st.sidebar.title("Settings")
# option = st.sidebar.selectbox("Choose a task", ["top 25", "top 100", "top with answers"])

if st.button("🔍 Tag it!"):
    if not user_question.strip():
        st.warning("Please enter your question.")
    else:
        full_text = f"{user_title} {user_question}" if user_title else user_question

        with st.spinner("Tagging your question..."):
            cleaned_html = remove_html_tags(full_text)
            cleaned_text = remove_punctuation_preserve_programming_terms(cleaned_html)
            X_test_tfidf = loaded_tfidf.transform([cleaned_text])
            y_pred_proba = model.predict_proba(X_test_tfidf)
            top_5_indices = y_pred_proba[0].argsort()[-5:][::-1]
            top_5_tags = [mlb.classes_[i] for i in top_5_indices if y_pred_proba[0][i] > 0.05]

        if top_5_tags:
            st.success(" Top 5 predicted tags:")
            st.code(' | '.join(top_5_tags))
        else:
            st.info(" No tags were confidently predicted.")

