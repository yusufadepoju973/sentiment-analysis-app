from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the Hugging Face sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.form["text"]

    try:
        # Perform sentiment analysis
        result = sentiment_analyzer(input_text)[0]
        sentiment = result["label"]  # Sentiment label (e.g., POSITIVE, NEGATIVE)

        # Map Hugging Face labels to your desired format
        sentiment_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive",
            "NEGATIVE": "Negative",
            "POSITIVE": "Positive",
            "NEUTRAL": "Neutral",
        }
        sentiment = sentiment_mapping.get(sentiment.upper(), sentiment)

        return render_template("result.html", text=input_text, sentiment=sentiment)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

