from flask import Flask, request, jsonify, render_template
from main_model import tree, model_accuracy

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', accuracy=round(model_accuracy * 100, 2))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = [
        float(data['Pregnancies']),
        float(data['Glucose']),
        float(data['BloodPressure']),
        float(data['SkinThickness']),
        float(data['Insulin']),
        float(data['BMI']),
        float(data['DiabetesPedigreeFunction']),
        float(data['Age'])
    ]
    prediction = tree.predict(input_features)
    result = "Disease Detected" if prediction == 1 else "No Disease Detected"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
