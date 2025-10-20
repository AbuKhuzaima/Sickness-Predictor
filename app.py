from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)


df = pd.read_csv('disease.csv')


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['disease'], test_size=0.2, random_state=42)


model = make_pipeline(TfidfVectorizer(), SVC())
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = model.predict([text])[0]
        return render_template('result.html', prediction=prediction)

@app.route('/calculate_mse', methods=['GET'])
def calculate_mse():
    # Predict labels for the test set
    y_pred = model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    return f'Mean Squared Error (MSE): {mse}'

if __name__ == '__main__':
    app.run(debug=True)
