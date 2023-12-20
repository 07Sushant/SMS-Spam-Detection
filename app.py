import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = y[:]

    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]

    y = [ps.stem(i) for i in text]

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI with custom HTML/CSS
st.markdown(
    """
    <style>
        body {
            background-color: rgba(255, 255, 255, 0.3);
            color: rgba(0, 0, 0, 0.8);
            font-family: Arial, sans-serif;
        }
        .stTextInput {
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: rgba(0, 123, 255, 0.8);
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: rgba(0, 123, 255, 1);
        }
        .stMarkdown {
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message", height=100)

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
    else:
        st.warning("Please enter a message for prediction.")
