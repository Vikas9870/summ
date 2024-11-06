from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains (or configure for specific domains if needed)
CORS(app)

# Initialize the Hugging Face summarization pipeline
summarizer = pipeline("summarization")

# Define the summarization route
@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text data from the incoming request
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Summarize the text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    
    # Return the summary as a JSON response
    return jsonify({"summary": summary[0]["summary_text"]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

