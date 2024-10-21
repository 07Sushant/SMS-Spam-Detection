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

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load models
vectorizer_path = 'vectorizer.pkl'
model_path = 'model.pkl'

# Loading vectorizer and model with error handling
try:
    with open(vectorizer_path, 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)
        # Check if the vectorizer is fitted
        if not hasattr(tfidf, 'vocabulary_'):
            raise ValueError("The loaded vectorizer is not fitted. Please fit the vectorizer before saving.")
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI with background image
st.markdown(
    """
    <style>
        body {
            background-image: url('https://your_website.com/your_image_path.jpg'); /* Replace with your image URL */
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI components
st.title("SMS Classifier")

input_sms = st.text_area("Enter the message", height=100)

if st.button('Check'):
    if input_sms:
        try:
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
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.warning("Please enter a message for prediction.")
