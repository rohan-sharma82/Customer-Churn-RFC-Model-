# 🔮 Customer Churn Prediction ML Model

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg" alt="ML Model">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" alt="Status">
</div>

## 📊 Overview

This project implements a **Random Forest Classifier** to predict customer churn risk for an online trading platform. The model categorizes customers into **High**, **Medium**, or **Low** churn risk categories, helping businesses proactively retain valuable customers.

## 🎯 Key Features

- **Multi-class Classification**: Predicts churn risk in 3 categories (High/Medium/Low)
- **Robust Preprocessing**: Handles missing values and categorical encoding automatically
- **Feature Scaling**: Standardizes numerical features for optimal model performance
- **Model Validation**: Includes cross-validation and overfitting detection
- **Visualization**: Comprehensive confusion matrix and performance metrics

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm scikitplot
```

### Usage
```python
# Load the saved model
import pickle
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(new_customer_data)
```

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 94.0% |
| **Train Accuracy** | 93.5% |
| **Cross-validation (5-fold)** | 92.9% ± 1.4% |
| **Precision (Macro Avg)** | 95% |
| **Recall (Macro Avg)** | 90% |
| **F1-Score (Macro Avg)** | 92% |

### Class-wise Performance
| Churn Risk | Precision | Recall | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| **Low (0)** | 90% | 99% | 94% | 251 |
| **Medium (1)** | 99% | 100% | 99% | 170 |
| **High (2)** | 97% | 70% | 81% | 96 |

✅ **No Overfitting Detected**: Train accuracy (93.5%) vs Test accuracy (94.0%)

## 🔧 Technical Architecture

### Data Preprocessing Pipeline
```
Raw Data → Missing Value Imputation → One-Hot Encoding → Feature Scaling → Model Training
```

### Features Used
- **Categorical**: Gender, Status, Homeowner
- **Numerical**: All numeric columns (auto-detected)
- **Target**: Churn Risk (High/Medium/Low)

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Estimators**: 60 trees
- **Max Depth**: 3
- **Random State**: 12 (for reproducibility)

## 📁 Project Structure

```
├── churn.csv                    # Dataset
├── Classification_RF_Model.ipynb # Main notebook
├── churn_prediction.py          # Python script version
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🎨 Visualizations

The project includes:
- **Churn Risk Distribution**: Bar chart showing class balance
- **Confusion Matrix**: Heatmap with prediction accuracy breakdown
- **Performance Metrics**: Detailed classification report

## 🔍 Key Insights

- Automated feature detection separates categorical and numerical columns
- Pipeline architecture ensures consistent preprocessing for train/test data
- Model achieves strong performance with cross-validation scores
- Overfitting detection implemented to ensure model generalization

## 🛠️ How It Works

1. **Data Loading**: Reads customer data from CSV file
2. **Preprocessing**: 
   - Removes ID column
   - Handles missing values in categorical columns
   - Applies one-hot encoding to categorical features
   - Standardizes numerical features
3. **Model Training**: Uses Random Forest with optimized hyperparameters
4. **Evaluation**: Comprehensive metrics including confusion matrix and cross-validation
5. **Model Saving**: Exports trained model using pickle

## 📊 Sample Output

```
Accuracy of Random Forest Classifier: 0.9400

Classification Report:
                 precision    recall  f1-score   support
Low Risk (0)         0.90      0.99      0.94       251
Medium Risk (1)      0.99      1.00      0.99       170  
High Risk (2)        0.97      0.70      0.81        96

Cross-validation scores: [0.945, 0.910, 0.926, 0.939, 0.926]
Mean CV score: 0.929
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Dataset: Customer churn data from online trading platform
- Built with: scikit-learn, pandas, numpy, matplotlib, seaborn
- Visualization: scikitplot for enhanced model evaluation plots

## 📞 Contact

Feel free to reach out if you have questions or suggestions!

---

<div align="center">
  <b>⭐ Star this repo if you found it helpful! ⭐</b>
</div>
