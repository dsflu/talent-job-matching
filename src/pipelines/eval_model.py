"""
This pipeline defines how we evaluate the trained models.
"""
import os
import json
import joblib
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.rule_based_model import RuleBasedModel
from src.evaluation.evaluator import BinaryClassificationEvaluator
from src.data_processing.data_loader import CSVDataLoader
from src.utils.config_utils import load_yaml_config
from src.utils.log import log

def load_data(config_path: str, file_type: str = 'csv'):
    """
    Load the processed data for both training and testing.

    Args:
        config_path: Path to the data config YAML file to read the location of the processed data.
        file_type: Type of the data file (default: 'csv').

    Returns:
        X_train, X_test, y_train, y_test: Training and testing data.
    """
    data_config = load_yaml_config(config_path)
    processed_data_save_path = data_config['processed_data_save_path']

    if file_type == 'csv':
        loader = CSVDataLoader
    else:
        raise ValueError("Only csv file is currently supported")

    X_train_loader = loader(os.path.join(processed_data_save_path, 'X_train.' + file_type))
    X_train_loader.load_data()
    X_train = X_train_loader.to_pandas()

    X_test_loader = loader(os.path.join(processed_data_save_path, 'X_test.' + file_type))
    X_test_loader.load_data()
    X_test = X_test_loader.to_pandas()

    y_train_loader = loader(os.path.join(processed_data_save_path, 'Y_train.' + file_type))
    y_train_loader.load_data()
    y_train = y_train_loader.to_pandas()

    y_test_loader = loader(os.path.join(processed_data_save_path, 'Y_test.' + file_type))
    y_test_loader.load_data()
    y_test = y_test_loader.to_pandas()

    return X_train, X_test, y_train, y_test

def evaluate_model(data_config_path: str, model_config_path: str):
    """
    Evaluate a trained model based on the configuration (for both training and testing).

    Args:
        data_config_path: Path to the data configuration YAML file.
        model_config_path: Path to the model configuration YAML file.
    """
    X_train, X_test, y_train, y_test = load_data(data_config_path)
    log.info("Training and testing data loaded")

    model_config = load_yaml_config(model_config_path)
    model_type = model_config['model_type']
    model_save_path = model_config['model_save_path']
    evaluation_save_path = model_config['evaluation_save_path']

    if model_type == 'logistic_regression':
        model = LogisticRegressionModel()
    elif model_type == 'random_forest':
        model = RandomForestModel()
    elif model_type == 'rule_based':
        model = RuleBasedModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type != 'rule_based':
        model = joblib.load(model_save_path)
    log.info(f"{model_type} model loaded")

    log.info("Evaluating the model.")
    evaluator = BinaryClassificationEvaluator()
    train_metrics = evaluator.evaluate(model, X_train, y_train)
    test_metrics = evaluator.evaluate(model, X_test, y_test)

    log.info(f"{model_type} Model Training Metrics: {train_metrics}")
    log.info(f"{model_type} Model Testing Metrics: {test_metrics}")

    os.makedirs(os.path.dirname(evaluation_save_path), exist_ok=True)
    evaluation_metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    with open(evaluation_save_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    log.info(f"{model_type} model evaluation metrics saved to {evaluation_save_path}.")