from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import uvicorn
import pandas as pd
import os

app = Flask(__name__)

#This API will take the json input of the 'month' and 'year' and output the prediction in json format. Based on the challenge description, I understand that the predictions should be provided specifically for the year 2021, with the month input expected as an integer (e.g., 1 for January).

#Input Example : curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"year":2021,"month":3}'

#Output: { "prediction": 20.08856773376465 }

current_dir = os.getcwd()

model_path = os.path.join(current_dir, 'model.pkl')
with open(model_path, 'rb') as f:
     model = pickle.load(f)

dataset_preprocessed = os.path.join(current_dir, 'datasets/model_prediction_df.csv')
df = pd.read_csv(dataset_preprocessed)
print(dataset_preprocessed)

dataset_preprocessed_2021 = os.path.join(current_dir, 'datasets/df_2021.csv')
df_2021 = pd.read_csv(dataset_preprocessed_2021)

@app.route('/')
def home():
    return render_template_string("""
    <h2>Welcome to the Prediction API</h2>
    """)
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    print(data)
    
    year = data['year']
    month = data['month']

    print(year)
    print(month)

    last_12_values_list = df['WERT'].tail(12).tolist()
    predictions = []

    import numpy as np

    for _ in range(12):
        X_pred = np.array(last_12_values_list[-12:]).reshape(1, -1)
        next_pred = model.predict(X_pred)[0]
        predictions.append(next_pred)
        last_12_values_list.append(next_pred)

    print('predictions', predictions)

    df_2021['Predictions'] = predictions

    df_2021['MONAT'] = df_2021['MONAT'].str.split("-").str[1].astype(int)

    print(df_2021)

    filtered = df_2021[(df_2021['JAHR'] == year) & (df_2021['MONAT'] == month)]
    
    if not filtered.empty:
        prediction = filtered['Predictions'].values[0]
        return jsonify({"prediction": float(prediction)})
    else:
        return {"error": "No prediction found for the given year and month."}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use PORT env variable
    app.run(host='0.0.0.0', port=port)