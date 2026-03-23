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
        prob_dict = {str(clsss): float(p) for clsss, p in zip(model.classes_, y_prob[0])}
    else:
        y_prob = None
        prob_dict = None
    result = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "prob_dict": prob_dict
    }
    return result

def  validate_input_signal(signal=None, model=None):
    """
    Validate input signal 
    return: normalized input data
    """
    
    # 1. Basic checks
    if signal is None:
        
        raise ValueError("Signal is required")
    
    if not isinstance(signal, list):
        raise ValueError("Signal must be a list")
    
    # expected features
    feature_names = model['feature_names']
    n_features = len(feature_names)
    
    # 2. Length check 
    if len(signal) != n_features:
        raise ValueError(f"Signal must have {n_features} values")
    
    # 3. Convert to DataFrame (same as training)
    df = pd.DataFrame([signal], columns=feature_names)
    
    # 4. Convert to numeric (safety)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    if df.isnull().any().any():
        raise ValueError("Signal contains invalid (non-numeric) values")
    
    # 5. Normalize (z-score)
    trained_mean = model['mean']
    trained_std = model['std']
    
    df_scaled = (df - trained_mean) / trained_std    
    return df_scaled