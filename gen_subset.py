import argparse
import numpy as np 
import pickle
from synth_data.selection.selection import random_selection


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("dset_loc")
    argument_parser.add_argument("output_file")
    
    args = argument_parser.parse_args()
    return args.dset_loc, args.output_file


if __name__ == "__main__":
    dset_loc, output_file  = parse_args()
    with open(dset_loc, "rb") as f:
        numpy_dict = pickle.load(f)
    
    clusters = numpy_dict["data"]
    selected_point_clusters = random_selection(clusters, 100)

    for i in range(len(selected_point_clusters)):
        print(len(clusters[i]))
        print(selected_point_clusters[i].shape)

    dataset_dictionary          = {}
    dataset_dictionary["data"]  = selected_point_clusters

    with open(output_file, "wb") as output_location:
        pickle.dump(dataset_dictionary, output_location)


    
    