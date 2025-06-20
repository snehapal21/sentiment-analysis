# 💬 Sentiment Analysis Web App with Machine Learning

This project is a complete sentiment analysis system that processes text data, performs extensive NLP preprocessing, trains multiple machine learning models, and allows users to analyze sentiment via a web interface.

---

##  Features

- Text data cleaning with NLTK
- Sentiment analysis using VADER
- ML classifiers: Logistic Regression, Random Forest, SVM, Naive Bayes, and Passive Aggressive
- Hyperparameter tuning with RandomizedSearchCV
- Data visualization with Seaborn, Matplotlib & Plotly
- Flask-powered web app for real-time predictions

---

##  Tech Stack

- **Python**
- **Flask** for web app
- **NLTK** for text processing
- **Scikit-learn** for ML models
- **Plotly**, **Seaborn**, **Matplotlib** for data viz
- **Pandas**, **NumPy** for data handling

---

## 🗂️ Project Structure

```bash
📦 sentiment-analysis
├── app.py                   # Flask app for web interface
├── sentiment_analysis.py    # Main ML/NLP script
├── sentimentdataset.csv     # Dataset
├── requirements.txt         # Dependencies
├── templates/
│   └── index.html           # Webpage layout
│
├── static/
│   └── style.css            # CSS for styling
│
└── README.md
