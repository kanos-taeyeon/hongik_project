"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
import tqdm
import core
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import OrderedDict
from . import utility
import pickle
import logging

logger = logging.getLogger('Numpyout')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-5s %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

# file handler
f_handler = logging.FileHandler('file.log')
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

logger.addHandler(handler)
logger.addHandler(f_handler)

class NumpyOuting:
    def __init__(self,testdir, features, numpyoutput):
        self.testdir = testdir
        self.numpyoutput = numpyoutput
        self.features = features

    def run(self):
        y_feature = []
        name = []
        err = 0
        end = len(next(os.walk(self.testdir))[2])

        for sample in tqdm.tqdm(utility.directory_generator(self.testdir), total=end):
            fullpath = os.path.join(self.testdir, sample)#dir connect

            if os.path.isfile(fullpath):
                binary = open(fullpath, "rb").read()
                name.append(sample)

                try:
                    feat = core.numpy_sample(binary, self.features)
                    y_feature.append(feat)
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    logger.error('{}: {} error is occuered'.format(sample, e))
                    numsize = y_feature[0].shape[-1]
                    y_feature.append(np.zeros((numsize),dtype=np.float64))
                    err += 1

        np.savetxt(self.numpyoutput,y_feature)
        logger.info('{} error is occured'.format(err))
