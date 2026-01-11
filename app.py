import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only needed once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep important negation words
important_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                   'nowhere', 'hardly', 'barely', 'scarcely', 'none'}
sentiment_stopwords = stop_words - important_words

# Load models and vectorizer
@st.cache_resource
def load_models():
    """Load trained models and vectorizer"""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('naive_bayes_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        
        with open('svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        # Check if SVM has predict_proba (should be CalibratedClassifierCV)
        if not hasattr(svm_model, 'predict_proba'):
            st.warning("⚠️ SVM model doesn't have probability estimation. Please save a calibrated SVM model.")
            st.info("Use: `CalibratedClassifierCV(svm_model, cv=3)` before saving.")
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return vectorizer, nb_model, svm_model, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure all model files (.pkl) are in the same directory as app.py")
        return None, None, None, None

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in sentiment_stopwords]
    
    return ' '.join(tokens)

def predict_sentiment(text, vectorizer, nb_model, svm_model, label_encoder):
    """
    Predict sentiment for input text
    """
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Transform to TF-IDF features
    features = vectorizer.transform([cleaned_text])
    
    # Predict with Naive Bayes
    nb_pred = nb_model.predict(features)[0]
    nb_proba = nb_model.predict_proba(features)[0]
    
    # Predict with SVM
    svm_pred = svm_model.predict(features)[0]
    svm_proba = svm_model.predict_proba(features)[0]  # Now SVM has probabilities!
    
    # Convert predictions to labels
    nb_sentiment = label_encoder.inverse_transform([nb_pred])[0]
    svm_sentiment = label_encoder.inverse_transform([svm_pred])[0]
    
    return {
        'cleaned_text': cleaned_text,
        'naive_bayes': {
            'sentiment': nb_sentiment,
            'confidence': max(nb_proba) * 100,
            'probabilities': {
                label_encoder.classes_[i]: prob * 100 
                for i, prob in enumerate(nb_proba)
            }
        },
        'svm': {
            'sentiment': svm_sentiment,
            'confidence': max(svm_proba) * 100,
            'probabilities': {
                label_encoder.classes_[i]: prob * 100 
                for i, prob in enumerate(svm_proba)
            }
        }
    }

# ============================================================================
# STREAMLIT APP UI
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon="😊",
        layout="wide"
    )
    
    # Title and description
    st.title("🎭 Product Review Sentiment Analysis")
    st.markdown("""
    This application predicts the sentiment of product reviews using machine learning.
    
    **Models Used:**
    - 🔵 Naive Bayes Classifier
    - 🟢 Support Vector Machine (SVM)
    
    **Sentiment Classes:** Positive 😊 | Negative 😞 | Neutral 😐
    """)
    
    st.divider()
    
    # Load models
    with st.spinner("Loading models..."):
        vectorizer, nb_model, svm_model, label_encoder = load_models()
    
    if vectorizer is None:
        st.stop()
    
    st.success("✓ Models loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Project:** Sentiment Analysis on Product Reviews
        
        **Dataset:** 171,379 e-commerce product reviews
        
        **Features:** TF-IDF with 5,000 features and bigrams
        
        **Performance:**
        - Naive Bayes: ~84% accuracy
        - SVM: ~88% accuracy
        
        **Author:** Your Name
        **Course:** CCS3153 Natural Language Processing
        **Date:** January 2026
        """)
        
        st.divider()
        
        st.header("📝 Example Reviews")
        st.markdown("""
        **Positive:**
        - "Excellent product! Highly recommend!"
        - "Amazing quality, worth every penny"
        
        **Negative:**
        - "Terrible quality, waste of money"
        - "Very disappointed, not as described"
        
        **Neutral:**
        - "It's okay, nothing special"
        - "Average product, decent price"
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Review")
        
        # Text input
        user_input = st.text_area(
            "Type or paste a product review:",
            height=150,
            placeholder="Example: This product is absolutely amazing! Great quality and fast delivery."
        )
        
        # Predict button
        predict_button = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)
        
        # Sample reviews quick select
        st.markdown("**Or try these examples:**")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("😊 Positive Example"):
                user_input = "Excellent product! The quality is outstanding and it exceeded my expectations. Highly recommend!"
                predict_button = True
        
        with col_ex2:
            if st.button("😞 Negative Example"):
                user_input = "Terrible quality! Complete waste of money. Very disappointed with this purchase."
                predict_button = True
        
        with col_ex3:
            if st.button("😐 Neutral Example"):
                user_input = "It's okay. Nothing special, but it works as described. Average quality for the price."
                predict_button = True
    
    with col2:
        st.header("📊 Quick Stats")
        st.metric("Models Loaded", "2/2")
        st.metric("Vocabulary Size", "5,000 terms")
        st.metric("Feature Type", "TF-IDF + Bigrams")
    
    # Prediction results
    if predict_button and user_input.strip():
        st.divider()
        st.header("🎯 Analysis Results")
        
        with st.spinner("Analyzing sentiment..."):
            results = predict_sentiment(
                user_input, 
                vectorizer, 
                nb_model, 
                svm_model, 
                label_encoder
            )
        
        # Display original and cleaned text
        with st.expander("🔍 View Preprocessing Steps", expanded=False):
            st.markdown("**Original Text:**")
            st.info(user_input)
            st.markdown("**Cleaned Text (after preprocessing):**")
            st.code(results['cleaned_text'])
        
        # Results display
        col_nb, col_svm = st.columns(2)
        
        with col_nb:
            st.subheader("🔵 Naive Bayes Prediction")
            
            sentiment = results['naive_bayes']['sentiment']
            confidence = results['naive_bayes']['confidence']
            
            # Sentiment emoji
            emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
            
            st.markdown(f"### {emoji_map.get(sentiment, '😐')} {sentiment.upper()}")
            st.metric("Confidence", f"{confidence:.2f}%")
            
            # Probability distribution
            st.markdown("**Probability Distribution:**")
            probs = results['naive_bayes']['probabilities']
            
            for label, prob in probs.items():
                st.progress(prob / 100, text=f"{label.capitalize()}: {prob:.2f}%")
        
        with col_svm:
            st.subheader("🟢 SVM Prediction")
            
            sentiment_svm = results['svm']['sentiment']
            confidence_svm = results['svm']['confidence']
            
            # Sentiment emoji
            st.markdown(f"### {emoji_map.get(sentiment_svm, '😐')} {sentiment_svm.upper()}")
            st.metric("Confidence", f"{confidence_svm:.2f}%")
            
            # Probability distribution (same as Naive Bayes!)
            st.markdown("**Probability Distribution:**")
            probs_svm = results['svm']['probabilities']
            
            for label, prob in probs_svm.items():
                st.progress(prob / 100, text=f"{label.capitalize()}: {prob:.2f}%")
        
        # Agreement indicator
        st.divider()
        if sentiment == sentiment_svm:
            st.success(f"✅ **Both models agree:** {sentiment.upper()}")
        else:
            st.warning(f"⚠️ **Models disagree:** Naive Bayes says {sentiment.upper()}, SVM says {sentiment_svm.upper()}")
        
        # Download results
        st.divider()
        st.markdown("### 💾 Export Results")
        
        result_text = f"""
Sentiment Analysis Results
==========================

Original Text:
{user_input}

Cleaned Text:
{results['cleaned_text']}

Naive Bayes Prediction: {sentiment.upper()}
Confidence: {confidence:.2f}%

SVM Prediction: {sentiment_svm.upper()}
Confidence: {confidence_svm:.2f}%

Naive Bayes Probabilities:
{chr(10).join([f'  {k}: {v:.2f}%' for k, v in probs.items()])}

SVM Probabilities:
{chr(10).join([f'  {k}: {v:.2f}%' for k, v in probs_svm.items()])}

Models Agreement: {'YES - Both predict ' + sentiment.upper() if sentiment == sentiment_svm else 'NO - Disagreement'}
"""
        
        st.download_button(
            label="📥 Download Results as TXT",
            data=result_text,
            file_name="sentiment_analysis_results.txt",
            mime="text/plain"
        )
    
    elif predict_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Built with ❤️ using Streamlit | CCS3153 NLP Assignment | © 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()