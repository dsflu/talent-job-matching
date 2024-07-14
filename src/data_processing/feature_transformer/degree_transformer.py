import pandas as pd
from src.config.feature_settings import degree_mapping
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class DegreeTransformer(BaseTransformer):
    """
    Degree-related transformations.
    """

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in degree by filling them with "none"
        """
        df['talent.degree'] = df['talent.degree'].fillna("none")
        df['job.min_degree'] = df['job.min_degree'].fillna("none")
        return df

    def map_degree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map degree string to labels to easier processing.

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with transformed degree labels.
        """
        df = df.assign(
            talent_degree_label=lambda df: df['talent.degree'].map(degree_mapping),
            job_min_degree_label=lambda df: df['job.min_degree'].map(degree_mapping)
        )
        return df

    def transform_degree_match(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One degree feature transformation is made here:
        1. degree_match_binary: see if the talent's highest degree is met for the min degree requirements for the job

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with added degree-related features.
        """
        df = df.assign(
            degree_match_binary=lambda df: (df['talent_degree_label'] >= df['job_min_degree_label']).astype(int)
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to the DataFrame.
        """
        df = self.apply_transformation(df, self.handle_missing_values)
        df = self.apply_transformation(df, self.map_degree_features)
        df = self.apply_transformation(df, self.transform_degree_match)
        return df