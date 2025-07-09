# 🚗 UAE Used Car Price Prediction Project

This project focuses on analyzing used car data in the UAE and predicting car prices using various machine learning models. The goal is to build a reliable regression model that can help users estimate fair market prices for used cars.

---

## 📌 Project Structure

```bash
UAE_USED_CAR_ANALYSIS-PROJECT/
│
├── data/                    # Contains raw dataset (CSV files)
├── notebooks/               # Jupyter notebooks for EDA, preprocessing, modeling
├── src/                     # Python scripts for modular code structure
│   ├── components/          # Data transformation, model training, evaluation
│   ├── pipeline/            # Prediction pipeline and data flow
│   ├── utils.py             # Helper functions for loading/saving models
│   └── logger.py            # Logging setup
├── artifacts/               # Saved models, encoders, scaler, and test data
├── templates/               # HTML files for Flask UI (if applicable)
├── app.py                   # Flask API for prediction (optional)
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
└── setup.py                 # Setup for packaging
````

---

## 📊 Dataset Description

* **Source**: [Kaggle - UAE Used Cars Analysis Dataset](https://www.kaggle.com/datasets/mohamedsaad254/uae-used-cars-analysis-full-project-v1-0)
* **Attributes include**:

  * `Make`, `Model`, `Year`, `Milage`, `Transmission`, `Fuel_Type`, `Color`, `Location`, etc.
  * Target Variable: **Price**

---

## ⚙️ Tools & Technologies

* **Languages**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn, CatBoost, XGBoost, Matplotlib, Seaborn
* **ML Models Used**:

  * Linear Regression
  * Random Forest Regressor
  * XGBoost Regressor
  * CatBoost Regressor (Best Model)
* **Other Tools**:

  * Jupyter Notebook
  * Flask (for deployment)
  * VS Code
  * Git & GitHub

---

## 📈 Workflow

1. **Data Preprocessing**

   * Null value handling
   * Encoding categorical features using OneHotEncoder & BinaryEncoder
   * Feature scaling with StandardScaler

2. **Exploratory Data Analysis (EDA)**

   * Visualizing distributions, correlations, and trends

3. **Model Training & Evaluation**

   * Trained multiple regression models
   * Tuned hyperparameters
   * Evaluated using R² Score, MAE, RMSE
   * **Best Best model: **KNeighborsRegressor** with **R2 score:** **0.8288822627425911**

4. **Model Deployment**

   * Flask app for interactive predictions (optional)

---

## 📊 Results

* 📌 Final selected model: **KNeighborsRegressor**
* ✅ Accuracy (R² Score): **82.88**
* 🏆 Best performance in predicting prices of used cars in UAE

---

## 🧠 Future Improvements

* Add advanced hyperparameter tuning (GridSearchCV, Optuna)
* Use deep learning (Neural Network Regression)
* Integrate with a web dashboard (e.g., Streamlit)
* Improve UI/UX of prediction interface

---

## 🧑‍💻 Author

**Vansh Kaushik Singh**

* 📧 Email: [vanshkaushiksingh@gmail.com](mailto:vanshkaushiksingh@gmail.com)
* 🔗 GitHub: [@VANSHKAUSHIKSINGH](https://github.com/VANSHKAUSHIKSINGH)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).