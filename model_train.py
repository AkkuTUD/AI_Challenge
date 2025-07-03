import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

# Plotting
def plotting(df, filename):
    
    plt.figure(figsize=(14,8))

    df_total = df[df['AUSPRAEGUNG'] == 'insgesamt']
    pivot_df = df_total.pivot(index='MONAT', columns='MONATSZAHL', values='WERT')

    plt.figure(figsize=(14, 7))
    for category in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[category], label=category)

    plt.title('Historical Number of Accidents per Category (Total Accident Type)')
    plt.xlabel('DATE')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

#Forecasting using XGBoost Algorithm
def train_model(df, df_2021):
    for lag in range(1, 13):
        df[f'lag_{lag}'] = df['WERT'].shift(lag)

    cwd = os.getcwd()
    file = os.path.join(cwd, 'datasets/model_prediction_df.csv')
    df.to_csv(file, index=True)

    if os.path.isfile(file):
        print(f"File '{file}' saved successfully.")
    else:
        print(f"Failed to save the file '{file}'.")

    train_data = df.dropna().reset_index(drop=True)
    X_train = train_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', 'lag_11', 'lag_12']]
    y_train = train_data['WERT']


    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
    }

    model = xgb.XGBRegressor(objective='reg:squarederror', param_grid=param_grid)
    model.fit(X_train, y_train)

    save_dir = os.getcwd()
    file_path = os.path.join(save_dir, 'model.pkl')
    

    joblib.dump(model, file_path)

    last_12_values = df.tail(12)['WERT'].values

    # Create input for prediction
    X_pred = last_12_values.reshape(1, -1)
    predicted_value = model.predict(X_pred)[0]
    print(predicted_value)

    actual_value = df_2021['WERT'].values[0]

    return actual_value, predicted_value, model

# Calculate errors
def calculate_errors(actual_value, predicted_value):

    mse = mean_squared_error([actual_value], [predicted_value])
    mae = mean_absolute_error([actual_value], [predicted_value])
    rmse = np.sqrt(mse)

    return rmse, mae, mse



def main():
    #Load the dataset

    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'datasets/monatszahlen2505_verkehrsunfaelle_06_06_25.csv')

    print(file_path)
    df = pd.read_csv(file_path)

    df = df.dropna()
    df = df.drop(df[df['MONAT']=='Summe'].index)
    df['MONAT'] = df['MONAT'].str[:4] + '-' + df['MONAT'].str[4:6]

    df = df.sort_values('MONAT').reset_index(drop=True)

    #Plotting
    plot = plotting(df, filename='Historical_data_Plot.png')

    df = df[df['AUSPRAEGUNG'] == 'insgesamt']
    df = df[df['MONATSZAHL'] == 'Alkoholunfälle']

    df_2021 = df[df['JAHR'] == 2021]

    keep_cols = ['MONATSZAHL','AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']

    df_2021 = df_2021[keep_cols]

    cwd = os.getcwd()
    file_2021 = os.path.join(cwd, 'datasets/df_2021.csv')
    df_2021.to_csv(file_2021, index=True)
    
    df = df[df['JAHR'] <= 2020]

    # Drop all columns except keep_cols
    df = df[keep_cols]

    df = df.set_index(['MONAT'])
    
    actual_value, predicted_value, model = train_model(df, df_2021)
    rmse, mae, mse = calculate_errors(actual_value, predicted_value)

    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


if __name__ == "__main__":
    main()