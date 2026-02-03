import joblib

PREPROCESSOR_PATH = "artifacts/data_transformation/preprocessor.pkl"


def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)