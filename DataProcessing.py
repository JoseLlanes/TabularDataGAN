import pandas as pd

from Processing.CategoricalProcessing import CategoricalToNumericalNorm as c2nn


class DataPreprocessor:
    def __init__(self, path, data_name="", make_preprocess=True, include_categorical=True):
        self.path = path
        self.df = None
        self.cols_to_study = []
        self.categorical_columns = []
        self.nan_columns = []
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical

    def load_data(self):
        """Loads CSV file into a DataFrame."""
        self.df = pd.read_csv(self.path)

    def process_categorical_columns(self):
        """Converts categorical columns into numeric representations."""
        if not hasattr(self, 'categorical_columns'):
            raise AttributeError("categorical_columns must be defined in the subclass")

        for col in self.categorical_columns:
            category_inter = c2nn.create_category_intervals(self.df[col].values)
            self.df[f"{col}_numeric"] = self.df[col].apply(lambda x: c2nn.generate_number(x, category_inter))
            self.cols_to_study.append(f"{col}_numeric")

    def impute_nan_columns(self):
        for col in self.nan_columns:
            if isinstance(self.df[col].dropna().iloc[0], str):
                self.df.loc[self.df[col].isna(), col] = "NotKnown"

    def preprocess(self):
        self.load_data()
        self.impute_nan_columns()
        if self.include_categorical:
            self.process_categorical_columns()

        return self.df[self.cols_to_study].dropna().reset_index(drop=True)
    
    def get_data(self):
        df = self.preprocess() if self.make_preprocess else self.df[self.cols_to_study]

        return {
            "data": df,
            "data_name": self.data_name,
            "cols_to_study": self.cols_to_study
        }


class AdultDataPreprocessor(DataPreprocessor):
    def __init__(self, path="Data/UCI Adult Dataset.csv", data_name="Adult", make_preprocess=True, include_categorical=True):
        super().__init__(path)
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.cols_to_study = [
            'Age', 'Final Weight', 'Education Number', 'Capital Gain', 'Capital Loss', 'Hours Per Week'
        ]
        self.categorical_columns = [
            "Work Class", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Gender", "Salary"
        ]


class MaternalDataPreprocessor(DataPreprocessor):
    def __init__(self, path="Data/Maternal Health Risk Data Set.csv", data_name="Maternal", make_preprocess=True, include_categorical=True):
        super().__init__(path)
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.cols_to_study = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate']
        self.categorical_columns = ["BodyTemp"]


class TitanicDataPreprocessor(DataPreprocessor):
    def __init__(self, path="Data/Titanic.csv", data_name="Titanic", make_preprocess=True, include_categorical=True):
        super().__init__(path)
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.cols_to_study = ["Age", "Fare"]
        self.categorical_columns = ['Survived', "Pclass", "Sex", 'SibSp', 'Parch', 'Cabin', 'Embarked']
        self.nan_columns = ['Cabin', 'Embarked']
