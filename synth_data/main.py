from .dataset import DatasetFactory

import json
import pickle

class DataGeneratorMain:

    def generate_dataset(self, config_location):

        with open(config_location, "r") as config_file:
            config = json.load(config_file)

        output_loc      = config["output_loc"]
        del config["output_loc"]

        dataset_factory = DatasetFactory()
        dataset         = dataset_factory.get_dataset(config)
        dataset.draw()
        numpy_dict = dataset.get_numpy_dict()
        
        with open(output_loc, "wb") as output_location:
            pickle.dump(numpy_dict, output_location)