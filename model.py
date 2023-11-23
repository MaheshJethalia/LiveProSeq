# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Build LSTM model
def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dense(num_classes, activation='softmax'))  # Adjust num_classes for your task
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Split dataset into train and validation sets
# Assuming you have X_train, X_val, y_train, y_val from your data
X_train, X_val, y_train, y_val = train_test_split(spectra, labels, test_size=0.2, random_state=42)

# Train the LSTM model
lstm_model = build_lstm_model(input_shape, num_classes)  # Assuming you have defined input_shape and num_classes
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
