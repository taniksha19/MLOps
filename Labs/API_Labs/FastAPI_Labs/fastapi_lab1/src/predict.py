import joblib

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/iris_model_dt.pkl")
    y_pred = model.predict(X)
    return y_pred

def predict_rf(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/iris_model_rf.pkl")
    y_pred = model.predict(X)
    return y_pred
