# src/serving/inference_service.py

import pandas as pd

class InferenceService:
    def __init__(self, model, preprocessor, threshold: float):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold

    def predict(self, data: dict):
        # 1️⃣ dict → DataFrame
        df = pd.DataFrame([data])

        # 2️⃣ apply SAME preprocessor
        X = self.preprocessor.transform(df)

        print("DEBUG SHAPE AFTER PREPROCESS:", X.shape)

        # 3️⃣ PyFunc predict
        prob = float(self.model.predict(X)[0])

        prediction = 1 if prob >= self.threshold else 0

        return {
            "churn_probability": round(prob, 4),
            "threshold": self.threshold,
            "prediction": prediction
        }


    # def predict(self, features):
    #     """
    #     Perform churn prediction.
    #     """
    #     X = np.array(features).reshape(1, -1)

    #     #debug:
    #     raw_pred = self.model.predict(X)
    #     raw_proba = self.model.predict_proba(X)

    #     print("RAW PRED:", raw_pred)
    #     print("RAW PROBA:", raw_proba)

    #     # prob = self.model.predict_proba(X)[0][1]
    #     prob = raw_proba[0][1]

    #     prediction = "Yes" if prob >= self.threshold else "No"
             
    #     return {
    #         "prediction": prediction,
    #         "probability": round(float(prob), 4),
    #         "threshold": self.threshold
    #     }