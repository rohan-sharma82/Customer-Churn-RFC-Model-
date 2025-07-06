# 🔮 Customer Churn Prediction ML Model

<div align="center">
  
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg" alt="ML Model">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
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
| **Accuracy** | 85%+ |
| **Cross-validation** | 5-fold CV implemented |
| **Overfitting Check** | Train vs Test accuracy monitored |

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
├── random_forest_model.pkl      # Trained model
├── Classification_RF_Model.ipynb # Main notebook
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
Accuracy of Random Forest Classifier: 0.8547
Cross-validation scores: [0.84, 0.86, 0.85, 0.87, 0.83]
Mean CV score: 0.85
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

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
