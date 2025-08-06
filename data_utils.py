import numpy as np
import random
import torch
import torch.nn as nn

import pickle

FType = torch.FloatTensor
LType = torch.LongTensor

class RegionData:

    def __init__(self, config):
        self.config = config
        self.city = config['city']

        a_path = config[config['city']]['poi_f_path']
        s_path = config[config['city']]['source_path']
        s_sum_path = config[config['city']]['source_sum_path']
        d_path = config[config['city']]['destina_path']
        d_sum_path = config[config['city']]['destina_sum_path']

        self.a_m = np.load(a_path)
        self.s_m = np.concatenate([np.load(s_path),np.load(s_sum_path)],axis=1)
        self.d_m = np.concatenate([np.load(d_path),np.load(d_sum_path)],axis=1)
