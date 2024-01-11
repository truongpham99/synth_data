import argparse
import numpy as np 
import pickle
from synth_data.selection.selection import random_selection
from synth_data import DataGeneratorMain


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("dset_loc")
    argument_parser.add_argument("output_file")
    
    args = argument_parser.parse_args()
    return args.dset_loc, args.output_file


if __name__ == "__main__":
    config_loc  = "config\example_config_constrained.json"
    data_gen    = DataGeneratorMain()
    data_gen.generate_dataset(config_loc)

    dset_loc, output_file  = parse_args()
    with open(dset_loc, "rb") as f:
        numpy_dict = pickle.load(f)
    
    clusters = numpy_dict["data"]
    print(clusters)
    selected_point_clusters, _ = random_selection(clusters, 100)

    for i in range(len(selected_point_clusters)):
        print(len(clusters[i]))
        print(selected_point_clusters[i].shape)

    dataset_dictionary          = {}
    dataset_dictionary["data"]  = selected_point_clusters

    with open(output_file, "wb") as output_location:
        pickle.dump(dataset_dictionary, output_location)

    import matplotlib.pyplot as plt
    for cluster in clusters:
        print(cluster.shape)
        plt.scatter(cluster[:,0], cluster[:,1])

    for cluster in selected_point_clusters:
        print(cluster.shape)
        plt.scatter(cluster[:,0], cluster[:,1])

    plt.show()

    
    