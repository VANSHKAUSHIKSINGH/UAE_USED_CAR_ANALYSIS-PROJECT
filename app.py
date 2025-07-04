from flask import Flask, request, render_template, jsonify
import pandas as pd
from src.pipeline.Predict_pipeline import CustomData, PredictPipeline
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Load your dataset to extract dropdown options
df = pd.read_csv("notebook/data/uae_used_cars_10k.csv")

application = Flask(__name__)
app = application

# Extract unique options for dropdowns
make_options = sorted(df['Make'].dropna().unique().tolist())
body_type_options = sorted(df['Body_Type'].dropna().unique().tolist())
fuel_type_options = sorted(df['Fuel_Type'].dropna().unique().tolist())
transmission_options = sorted(df['Transmission'].dropna().unique().tolist())
location_options = sorted(df['Location'].dropna().unique().tolist())
color_options = sorted(df['Color'].dropna().unique().tolist())

@app.route('/')
def index():
    return render_template(
        'home.html',
        make_options=make_options,
        body_type_options=body_type_options,
        fuel_type_options=fuel_type_options,
        transmission_options=transmission_options,
        location_options=location_options,
        color_options=color_options,
        results=None
    )

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            data = CustomData(
                Make=request.form.get('Make'),
                Model=request.form.get('Model'),
                Mileage=float(request.form.get('Mileage')),
                Year=int(request.form.get('Year')),
                Body_Type=request.form.get('Body_Type'),
                Fuel_Type=request.form.get('Fuel_Type'),
                Transmission=request.form.get('Transmission'),
                Location=request.form.get('Location'),
                Color=request.form.get('Color'),
                Cylinders=int(request.form.get('Cylinders'))
            )
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:")
            print(pred_df)

            predict_pipeline = PredictPipeline()
            print("Predicting...")
            results = predict_pipeline.predict(pred_df)
            print("Prediction done!")

            return render_template(
                'home.html',
                make_options=make_options,
                body_type_options=body_type_options,
                fuel_type_options=fuel_type_options,
                transmission_options=transmission_options,
                location_options=location_options,
                color_options=color_options,
                results=round(results[0], 2)
            )
        except Exception as e:
            return f"Error occurred: {e}"
    else:
        return render_template(
            'home.html',
            make_options=make_options,
            body_type_options=body_type_options,
            fuel_type_options=fuel_type_options,
            transmission_options=transmission_options,
            location_options=location_options,
            color_options=color_options,
            results=None
        )

@app.route('/get_models/<make>')
def get_models(make):
    try:
        filtered_models = df[df['Make'] == make]['Model'].dropna().unique().tolist()
        filtered_models.sort()
        return jsonify(filtered_models)
    except Exception as e:
        return jsonify([])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
