"""
    Explainable credit classification. Original source authors:
    HERBRETEAU Mathis, LE POTTIER Mathias, ROBERT Paul-Aimé
"""

import shap
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from src.XGBoost_impl.SpecificDataSet import SpecificDataSet
import matplotlib.pyplot as plt


class XGBoostImpl():
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.data_set = SpecificDataSet()
        self.X_train,self.X_test,self.y_train, self.y_test = self.data_set.specific_prep()


    def train(self):
        """
        training
        input:
            test_size (float): size of the testing sample
        output:
            model
        """
        print("Starting training...")


        self.model = xgb.XGBClassifier(enable_categorical = True)

        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")

        self.evaluate()
        return self.model

    def evaluate(self):
        """
        evaluate the model trained
        """
        print("=====Eval=====")

        if self.model is None:
            raise Exception("The model has not been fitted.")
        # y_pred -> numpy array (n_samples, n_targets)
        y_pred = self.model.predict(self.X_test)

        # Collect true labels and predictions
        true_vals = self.y_test.values
        pred_vals = y_pred

        report = classification_report(true_vals, pred_vals, output_dict=True, zero_division=0)

        f1 = report['weighted avg']['f1-score']
        acc = report['accuracy']
        auc = roc_auc_score(true_vals, self.model.predict_proba(self.X_test)[:, 1])

        print(f"* Accuracy : {acc:.2%}")
        print(f"* F1-Score : {f1:.2%}")
        print(f"* AUC : {auc:.2%}")

    def predict(self, input_data):
        """

        """
        if self.model is None:
            raise Exception("The model has not been fitted.")


        # input_data must use the same prepared features as X_train.
        return self.model.predict(input_data)


    def test_random_prediction(self, do_explain = False):
        """
        Select a test row,
        predict its label and compare with the target.
        """
        if self.model is None:
            print("Fit the model before running the example.")
            return

        # 1. Select a sample index
        random_index = 6

        # 2. Extract one row of the test array
        # Keep the (1, n_features) shape
        sample_x = self.X_test[random_index:random_index+1]

        # 3. Extract the true label
        true_y = self.y_test.iloc[random_index]

        # 4. Make a prediction
        pred_y = self.model.predict(sample_x)[0] # Take the only prediction

        print(f"=====Sample prediction (Index: {random_index})=====")

        print("columns :\n",sample_x)
        print("Actual label : ",true_y)
        print("Predicted label : ",pred_y)

        if do_explain:
            self.explain_line(random_index)

    def explain_line(self,line = 3):

        if self.model is None:
            print("Fit the model before running the example.")
            return
        # Create a tree explainer
        xgb_explainer = shap.TreeExplainer(self.model)


        shap_explainer_values = xgb_explainer(self.X_test)
        shap.waterfall_plot(shap_explainer_values[line])

        plt.show()

    def explain_global(self):

        if self.model is None:
            print("Fit the model before running the example.")
            return


        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_test)

        # Global explanation
        shap.summary_plot(shap_values, self.X_test)

        plt.show()
if __name__ == "__main__":
    model = XGBoostImpl()
    model.train()
    model.explain_global()

    model.test_random_prediction(do_explain=True)
