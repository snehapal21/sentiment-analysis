import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
import plotly.express as px
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud

init(autoreset=True)

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("sentimentdataset.csv")

df.head()

df.columns

print(df.columns)

for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")

df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Hashtags', 'Day', 'Hour', 'Sentiment'], errors='ignore')

for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")

df['Platform'].value_counts()

df['Country'].value_counts()

if 'Country' in df.columns:
    df['Country'] = df['Country'].str.strip()

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Day_of_Week'] = df['Timestamp'].dt.day_name()
else:
    print("Warning: 'Timestamp' column is missing.")

if 'Month' in df.columns and pd.api.types.is_numeric_dtype(df['Month']):
    month_mapping = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }
    df['Month'] = df['Month'].map(month_mapping)
    df['Month'] = df['Month'].astype('object')
else:
    print("Warning: 'Month' column is missing or not numeric.")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'<.*?>+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)  
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    text = " ".join(text.split())
    tokens = word_tokenize(text)

    cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]

    cleaned_text = ' '.join(cleaned_tokens)
    print(f"Cleaned Text: {cleaned_text}")  

    return cleaned_text

__all__ = ['clean', 'analyzer']

analyzer = SentimentIntensityAnalyzer()

if __name__ == "__main__":
    df["Clean_Text"] = df["Text"].apply(clean)

    specified_columns = ['Platform', 'Country', 'Year', 'Month', 'Day_of_Week']

    for col in specified_columns:
        total_unique_values = df[col].nunique()
        print(f'Total unique values for {col}: {total_unique_values}')

        top_values = df[col].value_counts()

        colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTBLACK_EX, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX]

        for i, (value, count) in enumerate(top_values.items()):
            color = colors[i % len(colors)]
            print(f'{color}{value}: {count}{Fore.RESET}')

        print('\n' + '=' * 30 + '\n')

    df1 = df.copy()

    df1['Vader_Score'] = df1['Clean_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
    df1['Sentiment'] = df1['Vader_Score'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))

    print(df1[['Clean_Text', 'Vader_Score', 'Sentiment']].head())

    colors = ['#66b3ff', '#99ff99', '#ffcc99']

    explode = (0.1, 0, 0)

    sentiment_counts = df1.groupby("Sentiment").size()

    fig, ax = plt.subplots()

    wedges, texts, autotexts = ax.pie(
        x=sentiment_counts,
        labels=sentiment_counts.index,
        autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})',
        wedgeprops=dict(width=0.7),
        textprops=dict(size=10, color="r"),
        pctdistance=0.7,
        colors=colors,
        explode=explode,
        shadow=True)

    center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
    fig.gca().add_artist(center_circle)

    ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

    ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax.axis('equal')

    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Year', hue='Sentiment', data=df1, palette='Paired')
    plt.title('Relationship between Years and Sentiment')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Month', hue='Sentiment', data=df1, palette='Paired')
    plt.title('Relationship between Month and Sentiment')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Platform', hue='Sentiment', data=df1, palette='Paired')
    plt.title('Relationship between Platform and Sentiment')
    plt.xlabel('Platform')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
    top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
    top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

    fig = px.bar(top_words_df,
                x="count",
                y="Common_words",
                title='Common Words in Text Data',
                orientation='h',
                width=700,
                height=700,
                color='Common_words')

    fig.show()

    Positive_sent = df1[df1['Sentiment'] == 'positive']
    Negative_sent = df1[df1['Sentiment'] == 'negative']
    Neutral_sent = df1[df1['Sentiment'] == 'neutral']

    top = Counter([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])
    temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
    temp_positive.style.background_gradient(cmap='Greens')

    top = Counter([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
    temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
    temp_positive.style.background_gradient(cmap='Blues')

    top = Counter([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
    temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
    temp_positive.style.background_gradient(cmap='Reds')

    df2 = df1.copy()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import confusion_matrix

    X = df2['Clean_Text'].values
    y = df2['Sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    logistic_classifier = LogisticRegression(max_iter=50, random_state=42)
    logistic_classifier.fit(X_train_tfidf, y_train)

    y_pred_logistic = logistic_classifier.predict(X_test_tfidf)
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    classification_rep_logistic = classification_report(y_test, y_pred_logistic)

    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy_logistic}")
    print("Classification Report:\n", classification_rep_logistic)

    random_forest_classifier = RandomForestClassifier(random_state=42)
    random_forest_classifier.fit(X_train_tfidf, y_train)


    y_pred_rf = random_forest_classifier.predict(X_test_tfidf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    classification_rep_rf = classification_report(y_test, y_pred_rf)

    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_rf}")
    print("Classification Report:\n", classification_rep_rf)

    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train_tfidf, y_train)

    y_pred_svm = svm_classifier.predict(X_test_tfidf)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    classification_rep_svm = classification_report(y_test, y_pred_svm)

    print("Support Vector Machine Results:")
    print(f"Accuracy: {accuracy_svm}")
    print("Classification Report:\n", classification_rep_svm)

    pac_classifier = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    pac_classifier.fit(X_train_tfidf, y_train)

    y_pred = pac_classifier.predict(X_test_tfidf)
    accuracy_test = accuracy_score(y_test, y_pred)
    classification_rep_test = classification_report(y_test, y_pred)

    print("Test Set Results:")
    print(f"Accuracy: {accuracy_test}")
    print("Classification Report:\n", classification_rep_test)

    param_dist = {
        'C': [0.1, 0.5, 1.0],
        'fit_intercept': [True, False],
        'shuffle': [True, False],
        'verbose': [0, 1],
    }

    pac_classifier = PassiveAggressiveClassifier(random_state=42)

    randomized_search = RandomizedSearchCV(pac_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    randomized_search.fit(X_train_tfidf, y_train)

    best_params_randomized = randomized_search.best_params_
    best_params_randomized

    best_pac_classifier_randomized = PassiveAggressiveClassifier(random_state=42, **best_params_randomized)
    best_pac_classifier_randomized.fit(X_train_tfidf, y_train)

    y_pred_best_pac_randomized = best_pac_classifier_randomized.predict(X_test_tfidf)

    accuracy_best_pac_randomized = accuracy_score(y_test, y_pred_best_pac_randomized)
    classification_rep_best_pac_randomized = classification_report(y_test, y_pred_best_pac_randomized)
    conf_matrix_test = confusion_matrix(y_test, y_pred_best_pac_randomized)

    print("Best PassiveAggressiveClassifier Model (RandomizedSearchCV):")
    print(f"Best Hyperparameters: {best_params_randomized}")
    print(f"Accuracy: {accuracy_best_pac_randomized}")
    print("Classification Report:\n", classification_rep_best_pac_randomized)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Greys', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.title('Confusion Matrix - Hyperparameters')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()