from flask import Flask, request, render_template_string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained SVM model
with open('svm_tfidf_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
          <head>
            <title>Sentiment Analysis</title>
          </head>
          <body>
            <h1>Sentiment Analysis</h1>
            <form method="POST" action="/predict">
              <input type="text" name="text" placeholder="Enter text..." value="{{ text }}">
              <button type="submit">Predict</button>
            </form>
            {% if prediction %}
              <h3>Prediction: {{ prediction }}</h3>
            {% endif %}
          </body>
        </html>
    ''', text="")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Preprocess the text input using the TF-IDF vectorizer
    text_vector = tfidf_vectorizer.transform([text])
    sentiment_text = ''
    # Perform sentiment classification using the SVM model
    sentiment = svm_model.predict(text_vector)[0]
    if sentiment == 0:
        sentiment_text = 'negative'
    else:
        sentiment_text = 'positive'
    return render_template_string('''
        <!DOCTYPE html>
        <html>
          <head>
            <title>Sentiment Analysis</title>
          </head>
          <body>
            <h1>Sentiment Analysis</h1>
            <form method="POST" action="/predict">
              <input type="text" name="text" placeholder="Enter text..." value="{{ text }}">
              <button type="submit">Predict</button>
            </form>
            <h3>Prediction: {{ prediction }}</h3>
          </body>
        </html>
    ''', text=text, prediction=sentiment_text)

if __name__ == '__main__':
    app.run(debug=True, port=8888)
