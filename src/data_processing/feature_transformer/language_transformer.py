import pandas as pd
from src.config.feature_settings import rating_mapping
from src.data_processing.feature_transformer.base_transformer import BaseTransformer

class LanguageTransformer(BaseTransformer):
    """
    Language-related transformations.
    """

    @staticmethod
    def must_have_languages_match(talent_languages, job_languages):
        """
        Check if the must-have languages in the job are in the talent's language pool.
        """
        must_have_dict = {lang['title']: rating_mapping[lang['rating']] for lang in job_languages if lang['must_have']}
        talent_language_dict = {lang['title']: rating_mapping[lang['rating']] for lang in talent_languages}

        for title, required_rating in must_have_dict.items():
            if title not in talent_language_dict or talent_language_dict[title] < required_rating:
                return 0
        return 1

    @staticmethod
    def count_good2have_languages(talent_languages, job_languages):
        """
        Count how many non-must-have languages are in the talent's language pool.
        """
        good2have_dict = {lang['title']: rating_mapping[lang['rating']] for lang in job_languages if not lang['must_have']}
        talent_language_dict = {lang['title']: rating_mapping[lang['rating']] for lang in talent_languages}
        
        count = 0
        for title, required_rating in good2have_dict.items():
            if title in talent_language_dict and talent_language_dict[title] >= required_rating:
                count += 1
        
        return count

    def transform_language_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            language_must_have_match_binary=lambda df: df.apply(
                lambda row: self.must_have_languages_match(row['talent.languages'], row['job.languages']), axis=1
            ),
            language_good2have_count=lambda df: df.apply(
                lambda row: self.count_good2have_languages(row['talent.languages'], row['job.languages']), axis=1
            )
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_transformation(df, self.transform_language_features)
        return df