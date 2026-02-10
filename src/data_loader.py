import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file {file_path} was not found')
    
    data = pd.read_csv(file_path)
    return data

def pre_processing_split(data):
    data.drop('Time', axis=1, inplace=True)
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    return X_train, X_test, y_train, y_test
