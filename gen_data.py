from synth_data import DataGeneratorMain

import argparse

def parse_args():
    argument_parser = argparse.ArgumentParser(prog="Synthetic Data Generator",
                                              description="Generates synthetic data given clustering types and constraints in an extensible manner")
    argument_parser.add_argument("config_loc")
    
    args = argument_parser.parse_args()
    return args.config_loc


if __name__ == "__main__":
    config_loc  = parse_args()    
    data_gen    = DataGeneratorMain()
    data_gen.generate_dataset(config_loc)