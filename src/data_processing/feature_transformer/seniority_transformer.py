import pandas as pd
from src.config.feature_settings import seniority_mapping
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class SeniorityTransformer(BaseTransformer):
    """
    Seniority-related transformations.
    """

    def transform_seniority_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            talent_seniority_label=lambda df: df['talent.seniority'].map(seniority_mapping),
            job_seniority_labels=lambda df: df['job.seniorities'].apply(
                lambda x: [seniority_mapping[seniority] for seniority in x]
            )
        )
        return df

    def transform_seniority_match(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df = self.apply_transformation(df, self.transform_seniority_features)
        df = self.apply_transformation(df, self.transform_seniority_match)
        return df