from lib import parser
import pandas as pd
import numpy as np

def parse_to_df(filepath):
    l = []
    with open(filepath, "rb") as f:
        for val in parser.read_vectors(f):
            l.append(val)
    return np.array(l)