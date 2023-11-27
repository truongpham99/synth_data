import argparse
import matplotlib.pyplot as plt
import numpy as np 
import pickle


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("dset_loc")
    
    args = argument_parser.parse_args()
    return args.dset_loc


if __name__ == "__main__":
    dset_loc  = parse_args()
    with open(dset_loc, "rb") as f:
        numpy_dict = pickle.load(f)
    
    clusters = numpy_dict["data"]
    for cluster in clusters:
        print(cluster.shape)
        plt.scatter(cluster[:,0], cluster[:,1])
    
    plt.savefig("test.png")