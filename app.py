from flask import Flask, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
from flask_cors import CORS
# Download Punkt model if not available
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

app = Flask(__name__)
CORS(app)
@app.route('/summarize', methods=['POST'])
def summarize():
  """
  Summarizes the provided text data in a POST request.

  Returns:
      JSON response containing the summary.
  """
  data = request.get_json()
  if not data or 'text' not in data:
    return jsonify({'error': 'Missing text data'}), 400

  text = data['text']
  num_sentences = int(data.get('num_sentences', 3))  # Default to 3 sentences

  parser = PlaintextParser.from_string(text, Tokenizer("english"))
  summarizer = LexRankSummarizer()
  summary = summarizer(parser.document, num_sentences)
  summary_text = ' '.join([str(sentence) for sentence in summary])

  return jsonify({'summary': summary_text})

if __name__ == '__main__':
  app.run(debug=True)
