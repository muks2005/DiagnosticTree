Awesome! I see that you’ve built not just the model but also a **full web app** (with a **dark mode toggle**, clean UI, and form inputs for predictions). 🔥  
Let's write you a complete, professional `README.md` based on everything you've shown:

---

# DiagnosticTree – Disease Prediction Using Decision Tree

![App Screenshot](./static/screenshot.png)

## Overview

**DiagnosticTree** is a machine learning project that implements a **custom Decision Tree Classifier** to predict the presence of a disease based on patient data. The model is integrated into a clean and interactive web application that allows users to input clinical parameters and receive real-time predictions.

The project demonstrates end-to-end development — from **model building** to **deployment-ready UI** — focused on transparency, simplicity, and accuracy.

---

## Features

- ✅ **Custom Decision Tree Implementation** (no external ML libraries)
- ✅ **75% Model Accuracy** on unseen test data
- ✅ **Web Interface** for real-time prediction
- ✅ **Dark/Light Mode Toggle** for better user experience
- ✅ **Data Visualization** (Correlation Heatmap & Confusion Matrix)
- ✅ **Responsive Input Form** for medical parameters

---

## Tech Stack

- **Python** (Model + Backend)
- **Flask** (Web Framework)
- **HTML/CSS/JavaScript** (Frontend)
- **pandas, NumPy** (Data Handling)
- **seaborn, matplotlib** (Data Visualization)

---

## Dataset

The dataset consists of medical diagnostic features such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The target label `Outcome` indicates the presence (`1`) or absence (`0`) of the disease.

---

## Model Details

- The **Decision Tree** is implemented from scratch.
- **Gini Impurity** is used to determine the best splits.
- **Max Depth** is set to 5 to prevent overfitting.
- Model achieved **75% accuracy** on the test dataset.
- **Confusion Matrix** provides insight into true/false predictions.

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/muks2005/DiagnosticTree.git
   cd DiagnosticTree
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```


## Future Improvements

- Add support for model retraining with user-uploaded data
- Expand model to support multiple diseases
- Integrate more advanced classifiers (Random Forest, XGBoost)
- Improve input validation and error handling

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Made  by [Muks2005](https://github.com/muks2005)**





