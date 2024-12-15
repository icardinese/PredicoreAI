from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import os
import json
from . import hyperparameter_tuning as ht

class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier()
        self.best_model = None
        self.y_pred = None
        self.label = 'xgb_model'

    def fit(self, x_train, y_train):
        best_params = None
    
        # print(f"Best parameters: {best_params}")
        if os.path.exists('classification_hyperparameters.json'):
            with open('classification_hyperparameters.json', 'r') as f:
                try:
                    data = json.load(f)
                    if self.label in data:
                        best_params = data[self.label]
                        # print(f"Loaded best hyperparameters for {self.label}: {best_params}")
                except json.JSONDecodeError:
                    print("Error loading best hyperparameters. Please run the hyperparameter tuning file!")
        # Train the best model

        if best_params is None:
            ht.hyperparametertuning(self.model, ht.get_xgb_param_grid(), x_train, y_train, self.label)
            if os.path.exists('classification_hyperparameters.json'):
                with open('classification_hyperparameters.json', 'r') as f:
                    try:
                        data = json.load(f)
                        if self.label in data:
                            best_params = data[self.label]
                            print(f"Loaded best hyperparameters for {self.label}: {best_params}")
                    except json.JSONDecodeError:
                        print("Error loading best hyperparameters. Please run the hyperparameter tuning file!")
        self.best_model = XGBClassifier(**best_params)
        self.best_model.fit(x_train, y_train)

    def predict(self, x_test):
        self.y_pred = self.best_model.predict(x_test)
        return self.y_pred
    
    def predict_proba(self, x_test):
        return self.best_model.predict_proba(x_test)

    def get_y_pred(self):
        return self.y_pred
    
    def get_model(self):
        return self.best_model