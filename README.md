
# Clickbait Detection System

## 📄 Overview

The Clickbait Detection System is an AI-driven tool designed to classify headlines as clickbait or non-clickbait. Leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques, this system analyzes textual features to determine the likelihood of a headline being clickbait.

![Clickbait Detection Architecture](https://github.com/vansh050/Clickbait_Detection_System/blob/main/images/architecture.png)

*Figure 1: System Architecture*

---

## ⚙️ Features

* **Headline Classification**: Distinguishes between clickbait and non-clickbait headlines.
* **Feature Extraction**: Utilizes NLP techniques to extract key features from headlines.
* **Model Training**: Implements ML algorithms for accurate classification.
* **Evaluation Metrics**: Provides performance metrics to assess model effectiveness.

---

## 🛠️ Technologies Used

* **Programming Languages**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn, NLTK, TensorFlow/Keras
* **Tools**: Google Colab, Jupyter Notebook

---

## 📁 Project Structure

```
Clickbait_Detection_System/
│
├── data/                # Dataset files
│   ├── headlines.csv    # Sample headlines dataset
│
├── notebooks/           # Jupyter Notebooks
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── src/                 # Source code
│   ├── feature_extraction.py
│   ├── model.py
│   └── utils.py
│
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vansh050/Clickbait_Detection_System.git
cd Clickbait_Detection_System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks

* **Data Preprocessing**: `notebooks/data_preprocessing.ipynb`
* **Model Training**: `notebooks/model_training.ipynb`
* **Evaluation**: `notebooks/evaluation.ipynb`

---

## 📊 Model Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Below is an example of the evaluation results:

![Model Performance Metrics](https://github.com/vansh050/Clickbait_Detection_System/blob/main/images/performance_metrics.png)

*Figure 2: Model Performance Metrics*

---

## 🧪 Example Usage

```python
from src.feature_extraction import extract_features
from src.model import load_model, predict

# Sample headline
headline = "You won't believe what happened next!"

# Extract features
features = extract_features(headline)

# Load pre-trained model
model = load_model('model.pkl')

# Predict clickbait probability
prediction = predict(model, features)
print(f"Clickbait Probability: {prediction}")
```

---

## 📈 Results

The model achieves the following performance metrics:

* **Accuracy**: 92%
* **Precision**: 90%
* **Recall**: 93%
* **F1-Score**: 91%

![ROC Curve](https://github.com/vansh050/Clickbait_Detection_System/blob/main/images/roc_curve.png)

*Figure 3: ROC Curve*

---

## 🔄 Future Improvements

* Incorporate deep learning models like LSTM or BERT for better accuracy.
* Expand the dataset to include more diverse headlines.
* Implement a web interface for real-time clickbait detection.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**created by Vansh Jaiswal**
