from utils import load_json, load_and_save_results
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from plot_figs import plot_curves, plot_images

list_of_params = os.listdir('params/')

for param_file in list_of_params:
    params = load_json('params/' + param_file)
    load_and_save_results(params)

filenames = os.listdir('csv/')

plot_curves(filenames)

for filename in filenames:
    
    plot_images(filename)    
    #print("Error plotting images for ", filename)
