from flask import Flask, render_template, request, jsonify
from sentiment_analysis import clean, analyzer 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned_text = clean(text)
    print(f"Cleaned Text: {cleaned_text}")

    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    print(f"Sentiment Scores: {sentiment_scores}") 

    sentiment = 'positive' if sentiment_scores['compound'] >= 0.05 else (
        'negative' if sentiment_scores['compound'] <= -0.05 else 'neutral'
    )

    return jsonify({
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'scores': sentiment_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
