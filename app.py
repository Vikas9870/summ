import gradio as gr
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains
CORS(app)

# Initialize the model pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Define the function for summarization
def summarize_text(text):
    result = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return result[0]["summary_text"]

# Create the Gradio interface
iface = gr.Interface(fn=summarize_text, inputs="text", outputs="text")

# Mount Gradio interface in the Flask app
iface.mount(app)

# Define a POST route to summarize text
@app.route("/summarize", methods=["POST"])
def summarize():
    # Get the text from the request
    data = request.get_json()
    text = data.get("text", "")
    
    # Call the summarization function
    summary = summarize_text(text)
    
    # Return the summary in the response
    return jsonify({"summary": summary})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False)

