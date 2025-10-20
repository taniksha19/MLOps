import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == '__main__': 
    print("Loading diabetes dataset...")
    # Load the Diabetes dataset
    X, y = datasets.load_diabetes(return_X_y=True)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Save the scaler
    joblib.dump(sc, 'scaler_diabetes.joblib')
    
    # Build a simple REGRESSION model (10 inputs, 1 output)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(1, activation='linear') # 'linear' for regression
    ])

    # Compile for regression: use 'mean_squared_error' as the loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    model.save('my_model_diabetes.keras')
    print("Diabetes Regression Model and scaler were trained and saved")