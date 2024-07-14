from src.pipelines.train_model import train_model
from src.pipelines.eval_model import evaluate_model

if __name__ == "__main__":
    data_config_path = 'src/config/data_config.yaml'
    # model_config_path = 'src/config/model_logistic_regression.yaml'
    # model_config_path = 'src/config/model_random_forest.yaml'
    model_config_path = 'src/config/model_rule_based.yaml'
    
    train_model(data_config_path, model_config_path)

    evaluate_model(data_config_path, model_config_path)