


#============================================================
class DataSpliter:
#============================================================
    def __init__(self):
        print(f'[DataSpliter][__init__] Get instance !')

    def make_split_data(self, train_ratio: int=8, dev_ratio: int=1, test_ratio: int=1):
        print(f'[DataSpliter][make_split_data] ratio - train: {train_ratio}, dev: {dev_ratio}, test: {test_ratio}')

### MAIN ###
if '__main__' == __name__:
    data_spliter = DataSpliter()

