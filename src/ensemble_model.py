from sklearn.ensemble import IsolationForest


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


class IsolationForestModel:

    def __init__(self, model: IsolationForest):

        self.model = model
        self.column_name = "is_anomaly"

    def process(self, anomaly_monad: AnomalyEnsembleMonad):

        feature_columns = anomaly_monad.features
        features_matrix = anomaly_monad.data_frame[feature_columns]

        self.model.fit(features_matrix)
        anomaly_monad.data_frame[self.column_name] = self.model.predict(features_matrix)

    def add_shap_values(self, anomaly_monad: AnomalyEnsembleMonad):
        if self.column_name in anomaly_monad.data_frame:
            # get shap values
            pass

        else:
            raise Exception("Anomaly Not processed ")













