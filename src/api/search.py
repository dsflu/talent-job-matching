import pandas as pd
import joblib
from typing import Any
from src.data_processing.feature_transformer.main_transformer import FeatureTransformer
from src.config.feature_settings import selected_features
from src.utils.config_utils import load_yaml_config

class Search:
    def __init__(self, model=None, model_config_path=None) -> None:
        if model is not None:
            self.model = model
        elif model_config_path is not None:
            self.model = self.load_model(model_config_path)
        else:
            raise ValueError("You need to provide model object or model_config_path.")
        
        self.transformer = FeatureTransformer()

    def load_model(self, model_config_path: str):
        """
        Load the model from model config

        Args:
            model_config_path: Path to the model configuration YAML file.

        Returns:
            Loaded model.
        """
        config = load_yaml_config(model_config_path)
        model_path = config['model_save_path']
        model = joblib.load(model_path)
        return model

    def match(self, talent: dict, job: dict) -> dict:
        """
        This method takes a talent and job as input and uses the machine learning
        model to predict the label. Together with a calculated score, the dictionary
        returned has the following schema:
        {
          "talent": ...,
          "job": ...,
          "label": ...,
          "score": ...
        }

        Args:
            talent: Talent data dict.
            job: Job data dict.

        Returns:
            A dcit with talent, job, predicted label, and score.
        """
        combined_data = [{"talent": talent, "job": job}]
        combined_data_df = pd.json_normalize(combined_data, sep='.')

        transformed_data = self.transformer.transform(combined_data_df)
        X = transformed_data[selected_features]

        label = bool(self.model.predict(X)[0])
        score = float(self.model.predict_proba(X)[0][1])

        return {
            "talent": talent,
            "job": job,
            "label": label,
            "score": score
        }

    def match_bulk(self, talents: list[dict], jobs: list[dict], filter_false_predictions: bool = False) -> list[dict]:
        """
        This method takes multiple talents and jobs as input and uses the machine
        learning model to predict the label for each combination. Together with a
        calculated score, the list returned (sorted descending by score!) has the
        following schema:
        [
          {
            "talent": ...,
            "job": ...,
            "label": ...,
            "score": ...
          },
          ...
        ]

        Args:
            talents: A list of talent data dict.
            jobs: A list of job data dict.
            filter_false_predictions: A flag indicating whether to filter out false predictions, so that we only return matched pairs.

        Returns:
            A list of dicts, each containing talent, job, predicted label, and score, sorted descending by score.
        """
        combined_data = [{"talent": talent, "job": job} for talent in talents for job in jobs]
        combined_data_df = pd.json_normalize(combined_data, sep='.')

        transformed_data = self.transformer.transform(combined_data_df)
        X = transformed_data[selected_features]

        labels = self.model.predict(X)
        scores = self.model.predict_proba(X)[:, 1]

        results = []
        index = 0
        for talent in talents:
            for job in jobs:
                results.append({
                    "talent": talent,
                    "job": job,
                    "label": bool(labels[index]),
                    "score": float(scores[index])
                })
                index += 1

        if filter_false_predictions:
            results = [result for result in results if result["label"]]

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def rank_and_filter(self, talent: dict, jobs: list[dict], criteria: dict[str, Any]) -> list[dict]:
        """
        Rank and filter job opportunities for a given talent based on model predictions and specified filtering criteria.

        Args:
            talent: Talent data dict.
            jobs: A list of job data dicts.
            criteria: A dictionary of criteria to filter jobs, say, {"salary_expectation": 80000, "seniority": "senior"}

        Returns:
            A list of jobs that match the specified criteria for the given talent, sorted by score.
        """
        combined_data = [{"talent": talent, "job": job} for job in jobs]
        combined_data_df = pd.json_normalize(combined_data, sep='.')

        transformed_data = self.transformer.transform(combined_data_df)
        X = transformed_data[selected_features]

        labels = self.model.predict(X)
        scores = self.model.predict_proba(X)[:, 1]

        results = [
            {
                "talent": talent,
                "job": jobs[i],
                "label": bool(labels[i]),
                "score": float(scores[i])
            }
            for i in range(len(labels))
        ]

        # Filter out all non-matched jobs
        results = [result for result in results if result["label"]]

        # Apply additional filtering criteria
        filtered_results = [result for result in results if self.filter_criteria(result["job"], criteria)]

        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return filtered_results
    
    def filter_criteria(self, job: dict, criteria: dict[str, Any]) -> bool:
        """
        Checks if a given job meets the criteria.

        Args:
            job: Job data dict.
            criteria: A dictionary of criteria to check.

        Returns:
            True if the job meets the criteria, False otherwise.
        """
        if "salary_expectation" in criteria:
            if job.get("max_salary", 0) < criteria["salary_expectation"]:
                return False

        if "seniority" in criteria:
            if criteria["seniority"] not in job.get("seniorities", []):
                return False

        return True