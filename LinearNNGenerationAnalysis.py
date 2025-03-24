from DataProcessing import AdultDataPreprocessor, MaternalDataPreprocessor
from sklearn.preprocessing import MinMaxScaler

all_data_list = [
    AdultDataPreprocessor().get_data(),
    MaternalDataPreprocessor().get_data()
]


