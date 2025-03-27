import pandas as pd

from Processing.CategoricalProcessing import CategoricalToNumericalNorm as c2nn


class DataPreprocessor:
    def __init__(self, path, data_name="", target_column="", make_preprocess=True, include_categorical=True):
        self.path = path
        self.df = None
        self.cols_to_study = []
        self.categorical_columns = []
        self.nan_columns = []
        self.target_column = target_column
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.all_category_inter = {}

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
            self.all_category_inter[col] = category_inter

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

        self.cols_to_study = [col for col in self.cols_to_study if self.target_column != col]
        
        return {
            "data": df,
            "data_name": self.data_name,
            "cols_to_study": self.cols_to_study,
            "target_col": self.target_column,
            "all_category_inter": self.all_category_inter
        }


class AdultDataPreprocessor(DataPreprocessor):
    def __init__(self, path="Data/UCI Adult Dataset.csv", data_name="Adult", make_preprocess=True, include_categorical=True):
        super().__init__(path)
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.target_column = "Salary"
        self.cols_to_study = [
            'Age', 'Final Weight', 'Education Number', 'Capital Gain', 'Capital Loss', 'Hours Per Week'
        ] + [self.target_column]
        self.categorical_columns = [
            "Work Class", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Gender", "Salary"
        ]
        
    def encode_target(self, df, new_target_col="target"):
        target_mapping = {"<=50K": 0, ">50K": 1}
        df[new_target_col] = df[self.target_column].map(target_mapping).fillna(-1)
        
        return df[df[new_target_col] != -1].reset_index(drop=True), new_target_col
        
    
    def custom_preprocess(self):
        data_dict = self.get_data()
        data_with_target, target_col = self.encode_target(data_dict["data"])
        
        data_dict["data"] = data_with_target
        data_dict["target_col"] = target_col
        
        return data_dict


class MaternalDataPreprocessor(DataPreprocessor):
    def __init__(self, path="Data/Maternal Health Risk Data Set.csv", data_name="Maternal", make_preprocess=True, include_categorical=True):
        super().__init__(path)
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.target_column = "RiskLevel"
        self.cols_to_study = [
            'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate'
        ] + [self.target_column]
        self.categorical_columns = ["BodyTemp"]
    
    
    def encode_target(self, df, new_target_col="target"):
        target_mapping = {'high risk': 1, 'low risk': 0, 'mid risk': 0}
        df[new_target_col] = df[self.target_column].map(target_mapping).fillna(-1)
        
        return df[df[new_target_col] != -1].reset_index(drop=True), new_target_col
        
    
    def custom_preprocess(self):
        data_dict = self.get_data()
        data_with_target, target_col = self.encode_target(data_dict["data"])
        
        data_dict["data"] = data_with_target
        data_dict["target_col"] = target_col
        
        return data_dict


class TitanicDataPreprocessor(DataPreprocessor):
    def __init__(self, path="Data/Titanic.csv", data_name="Titanic", make_preprocess=True, include_categorical=True):
        super().__init__(path)
        self.data_name = data_name
        self.make_preprocess = make_preprocess
        self.include_categorical = include_categorical
        self.target_column = "Survived"
        self.cols_to_study = ["Age", "Fare"]
        self.categorical_columns = ['Survived', "Pclass", "Sex", 'SibSp', 'Parch', 'Cabin', 'Embarked']
        self.nan_columns = ['Cabin', 'Embarked']
        
        
    def custom_preprocess(self):
        return self.get_data()
