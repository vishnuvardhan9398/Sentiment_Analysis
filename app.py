from flask import Flask, request, render_template_string
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# -------------------------
# LOAD YOUR TRAINED MODEL
# -------------------------
# Save your model and vectorizer before using this
# Example:
# pickle.dump(model, open("model.pkl","wb"))
# pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------
# NLP PROCESSING
# -------------------------
stop_words = set(stopwords.words('english'))
stop_words.discard('not')   # keep "not"
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def process_text(text):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    stemmed = [stemmer.stem(w) for w in filtered]
    return " ".join(stemmed)

# -------------------------
# UI TEMPLATE (HTML + CSS + JS)
# -------------------------
html_page = """
<!DOCTYPE html>
<html>
<head>
<title>Sentiment Analyzer 😎</title>
<style>
body {
    font-family: Arial;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    padding: 50px;
}

.container {
    background: rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 15px;
    width: 50%;
    margin: auto;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

textarea {
    width: 90%;
    height: 120px;
    border-radius: 10px;
    padding: 10px;
    border: none;
    outline: none;
    font-size: 16px;
}

button {
    background: #ff7eb3;
    border: none;
    padding: 12px 25px;
    margin-top: 15px;
    border-radius: 10px;
    font-size: 18px;
    color: white;
    cursor: pointer;
}

button:hover {
    background: #ff4e91;
}

.result {
    margin-top: 20px;
    font-size: 22px;
    font-weight: bold;
}
</style>
</head>

<body>

<div class="container">
    <h1>🎬 Sentiment Analyzer 💬</h1>
    <p>Enter your review below 👇</p>

    <form method="POST">
        <textarea name="text" placeholder="Type your review here..."></textarea><br>
        <button type="submit">Analyze 🚀</button>
    </form>

    {% if result %}
        <div class="result">
            {{ result }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        user_input = request.form["text"]

        cleaned = clean_text(user_input)
        processed = process_text(cleaned)

        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            result = "😊 Positive Review"
        else:
            result = "😡 Negative Review"

    return render_template_string(html_page, result=result)

# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)