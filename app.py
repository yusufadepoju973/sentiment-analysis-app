from flask import Flask, request, render_template
from transformers import pipeline
import logging

app = Flask(__name__)

# Load the Hugging Face sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/test', methods=['GET'])
def test():
    return "App is running!"




# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input text from the form
        input_text = request.form.get("text", "").strip()
        logging.info(f"Received input: {input_text}")

        if not input_text:
            return "Error: No input text provided", 400

        # Perform sentiment analysis
        result = sentiment_analyzer(input_text)[0]
        logging.info(f"Sentiment analysis result: {result}")

        # Extract sentiment
        sentiment = result["label"]  # Sentiment label (e.g., POSITIVE, NEGATIVE)
        sentiment_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive",
            "NEGATIVE": "Negative",
            "POSITIVE": "Positive",
            "NEUTRAL": "Neutral",
        }
        sentiment = sentiment_mapping.get(sentiment.upper(), sentiment)

        # Render result.html
        return render_template("result.html", text=input_text, sentiment=sentiment)

    except Exception as e:
        logging.error(f"Error in /predict route: {e}", exc_info=True)
        return "Internal Server Error", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

