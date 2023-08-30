# coding = utf-8

import sys
import argparse

from experiment import Experiment
# from util.config import ConfigFile

def main(argv):
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Command-line parameters for Indexing Embedding experiments')
    
    # Add command-line arguments
    # parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    # parser.add_argument('-E', '--embed', default=False, dest='to_embed', action='store_true', help='whether to embed database/query')
    # Add argument for the model name
    parser.add_argument('-M', '--model', type=str, required=True, dest='model_name', help='name of the model')

    # Parse the command-line arguments
    args = parser.parse_args()
    # args = parser.parse_args(argv[1: ])

    # Load configuration from the provided file path
    # conf = ConfigFile(args.confpath, dump=True)
    model_name = args.model_name

    # # Set 'to_embed' parameter based on the '-E' flag
    # if args.to_embed:
    #     conf.setHP('to_embed', True)

    # Initialize an Experiment instance with the loaded configuration
    experiment = Experiment(model_name)

    # Run the experiment
    experiment.run()

# Execute the script if it's the main module
if __name__ == "__main__":
    main(sys.argv)
    
 