import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_save_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # Dataframe with column names
    column_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(x_train, columns=column_names)
    df['MEDV'] = y_train

    # Call visualization
    visualize_data(df)

    # Normalize data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Save mean and std for later use
    joblib.dump((mean, std), 'mean_std.pkl')

    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    # Compile model with built-in Mean Squared Error loss
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

    # Train model
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

    # Save the trained model in the new .keras format
    model.save('boston_housing_model.keras')

    # Evaluate model
    test_loss, test_mae = model.evaluate(x_test, y_test)
    print(f'Test MAE: {test_mae}')

def visualize_data(df):
    # Plot histograms for each feature
    df.hist(bins=20, figsize=(20, 15))
    plt.show()

    # Plot heatmap of the correlation matrix
    plt.figure(figsize=(16, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()

def load_model_and_predict(features):
    # Load the trained model saved in .keras format
    model = tf.keras.models.load_model('boston_housing_model.keras')

    # Load mean and std
    mean, std = joblib.load('mean_std.pkl')

    # Predict house price
    predicted_price = predict_house_price(model, mean, std, features)
    print(f'Predicted house price: ${predicted_price * 1000:.2f}')

# Function to predict house price
def predict_house_price(model, mean, std, features):
    features = np.array(features).reshape(1, -1)
    normalized_features = normalize_data(features, mean, std)
    prediction = model.predict(normalized_features)
    return prediction[0][0]

# Function to normalize input data
def normalize_data(data, mean, std):
    return (data - mean) / std

if __name__ == "__main__":
    train_and_save_model()

    # Example input data for prediction
    example_features = [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]
    load_model_and_predict(example_features)
    print("Completed")

# Column descriptions
column_names = [
    'Per capita crime rate by town (CRIM)',
    'Proportion of residential land zoned for lots over 25,000 sq. ft. (ZN)',
    'Proportion of non-retail business acres per town (INDUS)',
    'Charles River dummy variable (1 if tract bounds river; 0 otherwise) (CHAS)',
    'Nitric oxides concentration (parts per 10 million) (NOX)',
    'Average number of rooms per dwelling (RM)',
    'Proportion of owner-occupied units built prior to 1940 (AGE)',
    'Weighted distances to five Boston employment centers (DIS)',
    'Index of accessibility to radial highways (RAD)',
    'Full-value property tax rate per $10,000 (TAX)',
    'Pupil-teacher ratio by town (PTRATIO)',
    '1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town (B)',
    'Percentage of lower status of the population (LSTAT)'
]
for i in column_names:
    print(i)
