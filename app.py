from flask import Flask, request, render_template
import subprocess
import json

app = Flask(__name__)

OLLAMA_MODEL_NAME = "phi3"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.form["text"]
    prompt = f"Analyze the sentiment of the following text and classify it as Negative, Neutral, or Positive: '{input_text}'"

    try:
        # Run Ollama using subprocess
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL_NAME],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )

        # Extract response from subprocess output
        sentiment = result.stdout.strip()
        return render_template("result.html", text=input_text, sentiment=sentiment)

    except subprocess.CalledProcessError as e:
        return f"An error occurred while running Ollama: {e.stderr}"

if __name__ == "__main__":
    app.run(debug=True)
