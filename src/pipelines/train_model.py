import os
import joblib
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.rule_based_model import RuleBasedModel
from src.data_processing.data_loader import CSVDataLoader
from src.utils.config_utils import load_yaml_config
from src.utils.log import log

def load_data(config_path: str, file_type: str = 'csv'):
    """
    Load the processed data for training.

    Args:
        config_path: Path to the data config YAML file to read the location of the processed data.
        file_type: Type of the data file (default: 'csv').

    Returns:
        X_train, y_train: Training data.
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

    y_train_loader = loader(os.path.join(processed_data_save_path, 'Y_train.' + file_type))
    y_train_loader.load_data()
    y_train = y_train_loader.to_pandas()

    return X_train, y_train

def train_model(data_config_path: str, model_config_path: str):
    """
    Train and save model.

    Args:
        data_config_path: Path to the data configuration YAML file.
        model_config_path: Path to the model configuration YAML file.
    """
    log.info("Loading training data.")
    X_train, y_train = load_data(data_config_path)

    model_config = load_yaml_config(model_config_path)
    model_type = model_config['model_type']
    model_params = model_config.get('model_params', {})
    model_save_path = model_config['model_save_path']

    if model_type == 'logistic_regression':
        model = LogisticRegressionModel(**model_params)
    elif model_type == 'random_forest':
        model = RandomForestModel(**model_params)
    elif model_type == 'rule_based':
        model = RuleBasedModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    log.info(f"{model_type} initilized")

    if model_type != 'rule_based':
        log.info("Training the model.")
        model.train(X_train, y_train)
        log.info("Training Done")

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        log.info(f"{model_type} model trained and saved to {model_save_path}.")
    else:
        log.info(f"{model_type} model does not require training and is not saved.")

    