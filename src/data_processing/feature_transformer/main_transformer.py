import pandas as pd
from src.data_processing.feature_transformer.salary_transformer import SalaryTransformer
from src.data_processing.feature_transformer.degree_transformer import DegreeTransformer
from src.data_processing.feature_transformer.seniority_transformer import SeniorityTransformer
from src.data_processing.feature_transformer.job_roles_transformer import JobRolesTransformer
from src.data_processing.feature_transformer.language_transformer import LanguageTransformer

class FeatureTransformer:
    """
    Apply all feature transformations.
    """

    def __init__(self):
        self.salary_transformer = SalaryTransformer()
        self.degree_transformer = DegreeTransformer()
        self.seniority_transformer = SeniorityTransformer()
        self.job_roles_transformer = JobRolesTransformer()
        self.language_transformer = LanguageTransformer()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.salary_transformer.transform(df)
        df = self.degree_transformer.transform(df)
        df = self.seniority_transformer.transform(df)
        df = self.job_roles_transformer.transform(df)
        df = self.language_transformer.transform(df)
        return df