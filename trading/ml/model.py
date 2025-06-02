import os
import joblib
import numpy as np

class TradingModel:
    """Fallback model when TensorFlow is not available"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.tf_available = False
        self._check_tensorflow()
    
    def _check_tensorflow(self):
        try:
            import tensorflow as tf
            self.tf_available = True
            self._build_model()
        except ImportError:
            print("TensorFlow not available. Using fallback prediction method.")
            self.tf_available = False
    
    def _build_model(self):
        if not self.tf_available:
            return
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

        model = Sequential()
        model.add(Input(shape=(self.sequence_length, 5)))  # Explicit input layer
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def prepare_data(self, df):
        """Prepare data for model training/prediction"""
        features = ['open', 'high', 'low', 'close', 'volume']
        X = df[features].values
        if self.scaler is None:
            self.scaler = joblib.load('models/scaler.save') 
        X_scaled = self.scaler.transform(X)
        
        sequences = []
        for i in range(len(X_scaled) - self.sequence_length):
            sequences.append(X_scaled[i:(i + self.sequence_length)])
            
        return np.array(sequences)
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model on historical data"""
        if self.tf_available and self.model:
            return self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size
            )
        else:
            print("TensorFlow not available. Model training skipped.")
    
    def predict(self, X):
        """Make predictions using available method"""
        if self.tf_available and self.model:
            return self.model.predict(X)
        else:
            # Fallback to simple moving average strategy
            return self._fallback_predict(X)
    
    def _fallback_predict(self, X):
        """Simple moving average crossover strategy"""
        if len(X) < self.sequence_length:
            return 0.5
        
        # Calculate short and long MA
        short_ma = np.mean(X[-20:, 3])  # 20-period MA of close prices
        long_ma = np.mean(X[-50:, 3])   # 50-period MA of close prices
        
        # Generate signal
        if short_ma > long_ma:
            return 0.7  # Bullish
        elif short_ma < long_ma:
            return 0.3  # Bearish
        return 0.5  # Neutral
    
    def save(self, path):
        """Save model and scaler"""
        if self.tf_available and self.model:
            self.model.save(os.path.join(path, 'model.h5'))
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(path, 'scaler.save'))
    
    @classmethod
    def load(cls, path):
        """Load saved model"""
        instance = cls()
        instance.model = load_model(os.path.join(path, 'model.h5'))
        instance.scaler = joblib.load(os.path.join(path, 'scaler.save'))
        return instance
