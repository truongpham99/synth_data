from .custom_dataset import CustomDataset

class DatasetFactory:

    def __init__(self):

        self.dataset_type_mapping = {"custom": CustomDataset}


    def get_dataset(self, dataset_config):
        
        dataset_name = dataset_config["type"]
        dataset_type = self.dataset_type_mapping[dataset_name]
        del dataset_config["type"]

        created_dataset = dataset_type(**dataset_config)
        return created_dataset