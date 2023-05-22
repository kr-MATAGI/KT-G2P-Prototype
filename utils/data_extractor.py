import random


#==================================================================
class DataExtractor:
#==================================================================
    def __init__(self, seed: int=42):
        print(f'[DataExtractor][__init__] Get instance !')
        print(f'[DataExtracor]')
        random.seed(42)

    def get_digit_data(
            self,
            digit_data_path: str
    ):
        print(f'[DataExtractor][get_digit_data] digit_data_path: {digit_data_path}')

        print()


### MAIN ###
if '__main__' == __name__:
    print(f'[data_extractor][__main__] MAIN !')

    # init path
    digit_data_path = '../data/digits/num_target_jjk.txt'

    data_extractor = DataExtractor()
    data_extractor.get_digit_data(digit_data_path=digit_data_path)

