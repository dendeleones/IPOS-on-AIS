# predictor.py
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from datamodels import Operation, Resource
from datetime import datetime

class DurationPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False

    def featurize(self, operation: Operation, resource: Resource, current_time: datetime):
        dow = current_time.weekday()
        hour = current_time.hour
        return np.array([
            hash(resource.id) % 100,
            hash(operation.item) % 100,
            operation.op_number,
            operation.norm_duration,
            dow,
            hour
        ], dtype=np.float32)

    def train(self, historical_data):
        X = np.array([d['features'] for d in historical_data])
        y = np.array([d['actual_duration'] for d in historical_data])
        self.model.fit(X, y)
        self.trained = True

    def predict(self, operation: Operation, resource: Resource, current_time: datetime) -> float:
        if not self.trained:
            return operation.norm_duration
        f = self.featurize(operation, resource, current_time)
        return float(self.model.predict([f])[0])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'trained': self.trained}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            obj.model = data['model']
            obj.trained = data['trained']
        return obj