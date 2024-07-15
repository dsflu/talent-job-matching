"""
This pipeline defines how we read the raw data and transform features and then split it into training and testing dataset.
"""
import os
from sklearn.model_selection import train_test_split
from src.data_processing.data_loader import JSONDataLoader
from src.data_processing.feature_transformer.main_transformer import FeatureTransformer
from src.config.feature_settings import selected_features, label_column
from src.utils.log import log
from src.utils.config_utils import load_yaml_config


def prepare_data(config_path: str) -> None:
    """Prepare data for training and testing. First load the raw data, then apply feture transformation,
    then split it into training and testing, and save them.

    Args:
        config_path: Path to the configuration YAML file.
    """
    data_config = load_yaml_config(config_path)
    raw_data_path = data_config['raw_data_path']
    processed_data_save_path = data_config['processed_data_save_path']
    
    log.info(f"Loading data from {raw_data_path}")
    json_data_loader = JSONDataLoader(raw_data_path)
    json_data_loader.load_data()
    log.info("Data loaded successfully")
    log.info(f"Converting it to pandas")
    raw_data_df = json_data_loader.to_pandas()
    
    if raw_data_df is not None:
        transformer = FeatureTransformer()
        log.info("Transform features")
        transformed_data = transformer.transform(raw_data_df)
        log.info("Feature transformation done")
        log.info(f"Transformed data shape: {transformed_data.shape}")
        
        X = transformed_data[selected_features]
        Y = raw_data_df[label_column]
        log.info(f"Selected features shape: {X.shape}")
        log.info(f"Label shape: {Y.shape}")
        
        log.info("Splitting the data into training and testing")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        log.info("Saving training and testing data")
        os.makedirs(processed_data_save_path, exist_ok=True)
        X_train.to_csv(os.path.join(processed_data_save_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(processed_data_save_path, 'X_test.csv'), index=False)
        Y_train.to_csv(os.path.join(processed_data_save_path, 'Y_train.csv'), index=False)
        Y_test.to_csv(os.path.join(processed_data_save_path, 'Y_test.csv'), index=False)
        log.info(f"Training and testing datasets saved to {processed_data_save_path}")
    else:
        log.error("Failed to load data")