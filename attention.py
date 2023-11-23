# Implement attention mechanism in LSTM model
from tensorflow.keras import backend as K

def attention_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    lstm_out, state_h, state_c = LSTM(128, return_sequences=True, return_state=True)(inputs)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)
    
    sent_representation = multiply([lstm_out, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    
    output = Dense(num_classes, activation='softmax')(sent_representation)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Train the LSTM model with attention
attention_lstm_model = attention_lstm_model(input_shape, num_classes)
attention_lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
