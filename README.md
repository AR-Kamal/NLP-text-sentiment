# Sentiment Analysis Web Application 🎭

A beautiful web application for predicting sentiment of product reviews using Machine Learning.

## Features ✨

- 🔵 **Naive Bayes Classifier** with probability scores
- 🟢 **Support Vector Machine (SVM)** predictions
- 📊 Real-time sentiment analysis
- 🎨 Beautiful, user-friendly interface
- 📥 Export results functionality
- 📝 Example reviews for quick testing

## Installation 🚀

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Model Files

Make sure you have these files in the same directory as `app.py`:
- `tfidf_vectorizer.pkl`
- `naive_bayes_model.pkl`
- `svm_model.pkl`
- `label_encoder.pkl`

These files are generated from your Jupyter notebook when you save the models.

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Usage 📝

1. **Enter a review** in the text area
2. **Click "Analyze Sentiment"** button
3. **View predictions** from both models
4. **Compare results** between Naive Bayes and SVM
5. **Download results** as a text file

## Quick Examples 🎯

Try these example reviews:

**Positive:**
- "Excellent product! Highly recommend!"
- "Amazing quality, worth every penny"

**Negative:**
- "Terrible quality, waste of money"
- "Very disappointed, not as described"

**Neutral:**
- "It's okay, nothing special"
- "Average product, decent price"

## How It Works 🔧

1. **Text Preprocessing:**
   - Lowercase conversion
   - Remove URLs, HTML tags, punctuation
   - Tokenization
   - Stopword removal (keeping negations)
   - Lemmatization

2. **Feature Extraction:**
   - TF-IDF vectorization (5,000 features)
   - Bigrams for context

3. **Prediction:**
   - Naive Bayes: Probabilistic classification
   - SVM: Geometric classification

4. **Output:**
   - Sentiment class (Positive/Negative/Neutral)
   - Confidence scores
   - Probability distributions

## Model Performance 📊

| Model | Accuracy | Best For |
|-------|----------|----------|
| Naive Bayes | ~84% | Speed, interpretability |
| SVM | ~88% | Overall accuracy |

## Troubleshooting 🔧

### Error: "Model file not found"
- Ensure all `.pkl` files are in the same directory as `app.py`
- Check that you've saved the models in your Jupyter notebook

### Error: "No module named 'streamlit'"
```bash
pip install streamlit
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

## File Structure 📁

```
project/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── tfidf_vectorizer.pkl       # TF-IDF model
├── naive_bayes_model.pkl      # Naive Bayes model
├── svm_model.pkl              # SVM model
└── label_encoder.pkl          # Label encoder
```

## Deployment 🌐

### Deploy to Streamlit Cloud (Free!)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Deploy to Heroku

1. Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## Screenshots 📸

*Add screenshots of your app here after running it!*

## Assignment Details 📚

**Course:** CCS3153 Natural Language Processing  
**Institution:** AlBukhari International University  
**Semester:** 1 2024/2025  
**Date:** January 2026  

## Author ✍️

**Your Name**  
**Student ID**  
**Email**  

## License 📄

This project is created for educational purposes as part of the NLP course assignment.

## Acknowledgments 🙏

- Dataset: E-commerce product reviews (171,379 reviews)
- Framework: Streamlit
- ML Libraries: scikit-learn, NLTK
- Models: Naive Bayes, SVM

---

**Enjoy analyzing sentiments!** 😊😞😐
