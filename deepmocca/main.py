#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os
import sys
import logging

import torch

@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
@ck.option('--in-file', '-if', help='Input file', required=True)
@ck.option('--model-file', '-mf', default='model.h5', help='Tensorflow model file')
@ck.option('--out-file', '-of', default='results.tsv', help='Output result file')
def main(data_root, in_file, model_file, out_file):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            in_file = os.path.join(data_root, in_file)
            model_file = os.path.join(data_root, model_file)
            if not os.path.exists(in_file):
                raise Exception(f'Input file ({go_file}) is missing!')
            if not os.path.exists(model_file):
                raise Exception(f'Model file ({model_file}) is missing!')
        else:
            raise Exception(f'Data folder {data_root} does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Read input data
    data = load_data(in_file)
    # Load GCN model
    model = load_model(model_file)
    # Run model
    output = model(data)
    # Write the results to a file
    print_results(output, out_file)

    

def load_data(in_file):
    """This function load input data and formats it
    """
    # TODO: Implement
    data = []
    return data

def load_model(model_file):
    """The function for loading a pytorch model
    """
    # TODO: Import the model class
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(model_file))
    model.eval()

def print_results(results, out_file):
    """Write results to a file
    """
    with open(out_file, 'w') as f:
        for item in results:
            f.write(item + '\n')
    

if __name__ == '__main__':
    main()
