from lib import parser
import pandas as pd

def parse_to_df(filepath):
    f = open(filepath, "rb")
    l = []
    for val in parser.read_vectors(f):
        l.append(val)
    return pd.DataFrame(l)