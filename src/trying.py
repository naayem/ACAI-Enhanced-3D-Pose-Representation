import os
import pickle

dirname=os.path.dirname
ANNOT_PATH = os.path.join(dirname(dirname(__file__)), os.path.join('data', 'h36m/dict_h36m_desc.pkl'))

print(ANNOT_PATH)

with open(ANNOT_PATH, 'rb') as f:
    DICT_H36M_DESC = pickle.load(f)