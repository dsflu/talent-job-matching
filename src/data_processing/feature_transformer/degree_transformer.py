import pandas as pd
from src.config.feature_settings import degree_mapping
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class DegreeTransformer(BaseTransformer):
    """
    Degree-related transformations.
    """

    def map_degree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            talent_degree_label=lambda df: df['talent.degree'].map(degree_mapping),
            job_min_degree_label=lambda df: df['job.min_degree'].map(degree_mapping)
        )
        return df

    def transform_degree_match(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            degree_match_binary=lambda df: (df['talent_degree_label'] >= df['job_min_degree_label']).astype(int)
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_transformation(df, self.map_degree_features)
        df = self.apply_transformation(df, self.transform_degree_match)
        return df