import pandas as pd

class RuleBasedModel:
    """
    A special rule-based model that outputs 1 if all specified interaction features are 1, otherwise 0.
    """

    def __init__(self):
        self.features = [
            'salary_match_binary',
            'degree_match_binary',
            'seniority_match_binary',
            'job_roles_match_binary',
            'language_must_have_match_binary'
        ]

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        No training required for the rule-based model.
        """
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the labels based on the rule.
        """
        return X[self.features].all(axis=1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probabilities. The rule-based model only outputs 0 or 1, so the probabilities are binary.
        """
        probas = self.predict(X)
        return pd.DataFrame({'0': 1 - probas, '1': probas}).to_numpy()