import pandas as pd
from pandas import merge
import py_entitymatching as em

def load_data(left_file_name, right_file_name, label_file_name, blocking_fn, include_self_join=False):
    A = em.read_csv_metadata(left_file_name , key="id", encoding='iso-8859-1')
    B = em.read_csv_metadata(right_file_name , key="id", encoding='iso-8859-1')
    try:
        G = pd.read_csv(label_file_name)
    except:
        G=None
    C = blocking_fn(A, B)
    if include_self_join:
        C_A = blocking_fn(A, A)
        C_B = blocking_fn(B, B)
        return A, B, G, C, C_A,C_B
    else:
        return A, B, G, C
