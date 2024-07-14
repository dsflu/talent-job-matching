from abc import abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


class BaseEvaluator:
    """
    Base evaluator.
    """

    @abstractmethod
    def evaluate(self, model, X_test, y_test):
        """
        Evaluate the model on the test data.
        """
        pass


class BinaryClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for binary classification models.
    """

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate the model on given data.
        
        Args:
            model: Trained model to evaluate.
            X_test: Test features.
            y_test: True labels for the test data.

        Returns:
            A dict containing evaluation metrics.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        return metrics