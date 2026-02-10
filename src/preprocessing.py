from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def create_preprocessor(X):
    ct = ColumnTransformer([('scaler', StandardScaler(), X.columns)])
    return ct
    
def features_scaling(X_train, X_test, ct):
    X_train_scaled = ct.fit_transform(X_train)
    X_test_scaled = ct.transform(X_test)
    return X_train_scaled, X_test_scaled

