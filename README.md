🚗 UAE Used Car Price Prediction Project
🔗 View Dataset on Kaggle
📌 Table of Contents
📘 Project Overview

🎯 Objective

📊 Dataset Description

🔍 Exploratory Data Analysis

⚙️ Machine Learning Workflow

🏆 Model Performance

💡 Key Insights

🧠 Tech Stack Used

📁 Project Structure

🚀 How to Run the Project

📌 Future Improvements

📬 Contact

📘 Project Overview
This project aims to predict used car prices in the UAE using machine learning techniques. It leverages a real-world dataset with features such as brand, model, year, mileage, fuel type, and transmission.
The goal is to build a predictive model to support car buyers, sellers, and dealerships in making fair, data-driven pricing decisions.

🎯 Objective
Clean and preprocess the dataset

Analyze patterns in features like brand, mileage, year, and price

Train and evaluate regression models to predict car prices

Select the best model based on accuracy and error metrics

📊 Dataset Description
Feature	                     Description
Brand	                        Manufacturer of the vehicle (e.g. Toyota)
Model	                        Model name of the car
Year                        	Manufacturing year
Mileage                    	Kilometers driven
Transmission               	Automatic / Manual
Fuel Type	                  Gasoline / Diesel / Hybrid / Electric
Body Type	                  Sedan, SUV, Hatchback, etc.
Color                         Cars Color
Location                   	Location of the car in the UAE
Price (Target)	               Resale price of the car in AED

🔍 Exploratory Data Analysis
Key insights:

Toyota, Nissan, and BMW are the most common brands

Newer cars with lower mileage are priced significantly higher

Automatic transmission dominates listings

SUVs and Sedans have a higher average price


⚙️ Machine Learning Workflow

✅ Preprocessing Steps

Handled missing values and outliers

Used OneHotEncoder and BinaryEncoder for categorical data

Applied StandardScaler to normalize mileage and numerical fields

🤖 Models Trained

K-Nearest Neighbors (KNN)

Random Forest Regressor

XGBoost Regressor

CatBoost Regressor (Best Model)

🏆 Model Performance
Model	          R² Score	   MAE	    RMSE
KNN	          0.4609	      0.55	    0.71
Random Forest	 0.6145	      0.46	    0.60
XGBoost	       0.6091	      0.47	    0.61
CatBoost	   ✅ 0.6342	     ✅0.45	✅ 0.59

📉 CatBoost outperformed all other models in R² score, MAE, and RMSE, making it the most reliable choice.

📊 (Insert model performance bar charts here)

💡 Key Insights
CatBoost Regressor was the best-performing model on the dataset

Preprocessing and encoding techniques directly impacted model accuracy

Features like Model, Year, Mileage, Brand, and Transmission are highly influential

The model could serve as the core engine behind a used car price estimator app in the UAE market

🧠 Tech Stack Used

Category	                 Tools / Libraries
Language	                 Python
Data Analysis	           Pandas, NumPy
Visualization	           Matplotlib, Seaborn
Machine Learning	        Scikit-learn, XGBoost, CatBoost
Feature Engg.	           OneHotEncoder, BinaryEncoder, StandardScaler
Deployment	              Flask (in progress)
Version Control	        Git, GitHub

📁 Project Structure

`
UAE_USED_CAR_ANALYSIS-PROJECT/
│
├── artifacts/               # Saved data & models
├── data/                    # Raw dataset
├── images/                  # EDA & model visuals
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── components/          # Training & evaluation modules
│   ├── pipeline/            # Train & predict pipelines
│   └── utils.py             # Helper functions
├── webapp/                  # Flask app (under development)
├── README.md                # Project summary
├── requirements.txt         # Package requirements
└── setup.py                 # Setup file
🚀 How to Run the Project
`


1️⃣ Clone the Repository
`
git clone https://github.com/VANSHKAUSHIKSINGH/UAE_USED_CAR_ANALYSIS-PROJECT.git
cd UAE_USED_CAR_ANALYSIS-PROJECT
`

2️⃣ Install Required Libraries

`
pip install -r requirements.txt
`


3️⃣ Train the Model
`
python src/pipeline/train_pipeline.py
`

4️⃣ Make Predictions

`
python src/pipeline/predict_pipeline.py
`
📌 Future Improvements
🎯 Finalize and deploy web app using Streamlit or Flask

🔍 Use SHAP or LIME for feature importance interpretation

🧪 Try ensemble stacking for even better results

📱 Deploy API for real-time price prediction

📬 Contact
👤 Vansh Kaushik
📧 Vansh.k0907@gmail.com