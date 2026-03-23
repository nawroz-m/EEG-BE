import pandas as pd
def predict_sample(model=None, X=None):
    """
    This method predict X using the model and return
    - `y_pred, y_prob, prob_df`
    """
    y_pred = model.predict(X) 
    y_pred = y_pred.tolist()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
        y_prob= y_prob.tolist()

        # Make sure each label probability is assigned properly
        prob = []
        for sample_probs in y_prob:
            sample_dic = {}
            for cls, p in zip(model.classes_, sample_probs):
                sample_dic[str(cls)] = float(p)
            prob.append(sample_dic)
    else:
        y_prob = None
        prob = None
    result = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "prob": prob
    }
    return result

def  validate_input_signal(signal=None, model=None):
    """
    Validate input signal 
    return: normalized input data
    """
    
    # Make sure signal is not empty
    if signal is None:
        raise ValueError("Signal is required")
    
    # expected features
    feature_names = model['feature_names']
    n_features = len(feature_names)
    # check if the signal have enougth columns
    if len(signal.columns) < n_features:
        raise ValueError(f"Signal must have atlest {n_features} values")
    
    # get only featured columns
    df = signal[feature_names] 
    
    # Convert to numeric (safety)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    if df.isnull().any().any():
        raise ValueError("Signal contains invalid (non-numeric) values")
    
    # Normalize (z-transform)
    trained_mean = model['mean']
    trained_std = model['std']
    
    df_scaled = (df - trained_mean) / trained_std    
    return df_scaled