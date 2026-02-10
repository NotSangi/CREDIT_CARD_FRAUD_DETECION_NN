from keras import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import joblib

def build_model(features):
    num_features = features.shape[1]
    model = Sequential()
    model.add(layers.InputLayer(shape=(num_features,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=12, verbose=1, validation_split=0.2, callbacks=[early_stop])
    
    joblib.dump(history, 'models/classification_model.pkl')
    return history
    
    