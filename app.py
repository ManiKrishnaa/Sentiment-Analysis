from flask import Flask, request, render_template
import pickle
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

with open("sentiment_analysis_encoder.pkl", "rb") as f:
    model,encoder = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", message=None, prediction=None)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        message = request.form["msg"]
        if message.strip() == "":
            return render_template("index.html", message=message, prediction="Please enter a message.")

        processed_message = preprocess(message)
        message_vectorized = tfidf_vectorizer.transform([processed_message])
        message_vectorized_dense = message_vectorized.toarray()
        prediction = model.predict(message_vectorized_dense)[0]
        predicted_label = encoder.inverse_transform([prediction])[0]
        return render_template("index.html", message=message, prediction=predicted_label)
    return render_template("index.html", message=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
