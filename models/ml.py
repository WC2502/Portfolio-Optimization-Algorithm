import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class MLModels:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100)
        self.gbm_model = GradientBoostingRegressor(n_estimators=100)

    def train(self, X, y):
        self.rf_model.fit(X, y)
        self.gbm_model.fit(X, y)

    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        gbm_pred = self.gbm_model.predict(X)
        return 0.5 * (rf_pred + gbm_pred)
