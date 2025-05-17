import numpy as np
from numpy.ma.core import anomalies
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
import shap


class AnomalyEnsembleMonad:

    def __init__(self, data_frame, features):
        self.data_frame  = data_frame
        self.features = features

    def bind(self, func):

        try:
            func(self)
        except Exception as e:
            print(f"{e}: Exception occurred")
        return self

    def get_data_frame(self):
        return self.data_frame

def get_softmax(x, c):

    x = np.array(x*c, dtype=float)
    max_ = np.max(x)
    diff_ = x - max_
    e_x = np.exp(diff_)
    return e_x / e_x.sum(axis=0)

def apply_softmax_transformation(row, features):
    positive_columns = [f"{col}" for col in features if row[f"{col}_shap"] >= 0]
    negative_columns = [f"{col}" for col in features if row[f"{col}_shap"] < 0]
    res = {}
    # for col in negative_columns:
    #     anomaly_monad.data_frame[f"{col}_softmax_shap"] = get_softmax(row[col], -1)
    negative_softmax_columns = [f"{col}_softmax" for col in negative_columns]
    if negative_columns:
        softmax_values = get_softmax(row[negative_columns], -1)
        for c, n in zip(negative_softmax_columns, softmax_values):
            res[c] = n


    positive_softmax_columns = [f"{col}_softmax" for col in positive_columns]
    for c in positive_softmax_columns:
        res[c] = 0

    return res





class IsolationForestModel:

    def __init__(self, model: IsolationForest):

        self.model = model
        self.column_name = "is_anomaly"
        self.feature_matrix = None
        self.feature_columns = None
        self.shap_columns = None

    def process(self, anomaly_monad: AnomalyEnsembleMonad):

        feature_columns = anomaly_monad.features
        features_matrix = anomaly_monad.data_frame[feature_columns]
        self.feature_matrix = features_matrix
        self.feature_columns = feature_columns

        self.model.fit(features_matrix)
        anomaly_monad.data_frame[self.column_name] = self.model.predict(features_matrix)

    def add_shap_values(self, anomaly_monad: AnomalyEnsembleMonad):
        if self.column_name in anomaly_monad.data_frame:
            # get shap values
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.feature_matrix)
            shap_columns = [f"{col}_shap" for col in self.feature_columns]
            anomaly_monad.data_frame[shap_columns] = shap_values
            self.shap_columns = shap_columns

        else:
            raise Exception("Anomaly Not processed ")

    def softmax_shap(self, anomaly_monad: AnomalyEnsembleMonad):
        if self.shap_columns:
            df = anomaly_monad.data_frame
            anomaly_monad.data_frame = df.join(df.apply(lambda  row : apply_softmax_transformation(row , self.feature_columns), axis=1, result_type="expand"))


class ZScoreModel:
    def __init__(self):
        self.feature_matrix = None
        self.feature_columns = None
        self._key = "z_score"

    @staticmethod
    def check_anomaly_zscore(data):
        mean = np.mean(np.asarray(data, dtype=float))
        std = np.std(np.asarray(data, dtype=float))
        z_score = (np.array(data, dtype=float) - mean) / std
        threshold = 3
        anomalies_ = []
        for i, z_value in enumerate(z_score):
            if abs(z_value) > threshold:
                anomalies_.append(1)
            else:
                anomalies_.append(0)

        return anomalies_

    @staticmethod
    def check_anomaly_robust_zscore(data):
        data = np.asarray(data, dtype=float)
        median = np.median(data)
        mad = median_abs_deviation(data, scale='normal')  # scale='normal' makes it comparable to std dev

        # Avoid division by zero if MAD is 0
        if mad == 0:
            return [0] * len(data)

        robust_z_scores = (data - median) / mad
        threshold = 3
        anomalies_ = [1 if abs(z) > threshold else 0 for z in robust_z_scores]

        return anomalies_

    def get_key(self):
        return self._key

    def calculate_z_values(self, anomaly_monad: AnomalyEnsembleMonad):
        df = anomaly_monad.data_frame
        self.feature_columns = anomaly_monad.features

        for col in self.feature_columns:
            anomalies_ = self.check_anomaly_robust_zscore(df[col])
            df[f"{col}_{self._key}"] = anomalies_

        return df
























