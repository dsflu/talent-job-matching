import pandas as pd
from src.config.feature_settings import seniority_mapping
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class SeniorityTransformer(BaseTransformer):
    """
    Seniority-related transformations.
    """

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in seniority by filling them with "none" or ["none"]
        """
        df['talent.seniority'] = df['talent.seniority'].fillna("none")
        df['job.seniorities'] = df['job.seniorities'].apply(lambda x: ['none'] if x is None else x)
        return df

    def transform_seniority_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map seniority string to labels to easier processing.

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with transformed seniority labels.
        """
        df = df.assign(
            talent_seniority_label=lambda df: df['talent.seniority'].map(seniority_mapping),
            job_seniority_labels=lambda df: df['job.seniorities'].apply(
                lambda x: [seniority_mapping[seniority] for seniority in x]
            )
        )
        return df

    def transform_seniority_match(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two seniority feature transformations are made here:
        1. seniority_match_binary: see whether the talent's seniority is with the job seniority
        2. seniority_exceed_binary: see whether the talent's seniority is above the job seniority

        Args:
            df: DataFrame with the original features to be transformed

        Returns:
            DataFrame with added seniority-related features.
        """
        df = df.assign(
            seniority_match_binary=lambda df: df.apply(
                lambda row: int(row['talent_seniority_label'] in row['job_seniority_labels']), axis=1
            ),
            seniority_exceed_binary=lambda df: df.apply(
                lambda row: int(all(row['talent_seniority_label'] > label for label in row['job_seniority_labels'])), axis=1
            )
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to the DataFrame.
        """
        df = self.apply_transformation(df, self.handle_missing_values)
        df = self.apply_transformation(df, self.transform_seniority_features)
        df = self.apply_transformation(df, self.transform_seniority_match)
        return df