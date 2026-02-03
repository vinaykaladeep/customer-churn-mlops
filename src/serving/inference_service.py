# src/serving/inference_service.py

import numpy as np

class InferenceService:
    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def predict(self, features):
        """
        Perform churn prediction.
        """
        X = np.array(features).reshape(1, -1)

        prob = self.model.predict_proba(X)[0][1]

        prediction = "Yes" if prob >= self.threshold else "No"

        return {
            "prediction": prediction,
            "probability": round(float(prob), 4),
            "threshold": self.threshold
        }