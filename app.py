import streamlit as st
import pickle
import string
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
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
# Assuming your files are in the same directory as the script
vectorizer_path = 'vectorizer.pkl'
model_path = 'model.pkl'

with open(vectorizer_path, 'rb') as vectorizer_file, open(model_path, 'rb') as model_file:
    tfidf = pickle.load(vectorizer_file)
    model = pickle.load(model_file)

# Streamlit UI with custom HTML/CSS
st.markdown(
    """
    <style>
        /* Your CSS styles */
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("SMS Classifier")

input_sms = st.text_area("Enter the message", height=100)

if st.button('Check'):
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
