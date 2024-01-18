"""
This script is used to visualize the results of the trained model.
It loads the model from the specified path and uses it to predict the output
and visualize it.
"""
import sys

if len(sys.argv) != 3:
    raise ValueError(
        "Provide the curriculum step, seed and the model path as command line arguments."
    )

import torch

import data
import util

import experiment
