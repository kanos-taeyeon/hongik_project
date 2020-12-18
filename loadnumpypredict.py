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

logger = logging.getLogger('predicting')
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

class LoadNumpyPredict:
    def __init__(self, modelpath,loadnumpypath, namedir,output):
        # load model with pickle to predict
        #with open(modelpath, 'rb') as f:
        #    #self.model = lgb.Booster(model_file=modelpath)
        #    self.model = pickle.load(open(modelpath, 'rb'))
        with open(modelpath, 'rb') as f:
            self.model = lgb.Booster(model_file=modelpath)
        self.loadnumpypath = loadnumpypath
        self.namedir =namedir
        self.output = output

    def run(self):
        name = []
        err = 0
        end = len(next(os.walk(self.namedir))[2])

        for sample in tqdm.tqdm(utility.directory_generator(self.namedir), total=end):
            fullpath = os.path.join(self.namedir, sample)#dir connect

            if os.path.isfile(fullpath):
                name.append(sample)


        X = np.loadtxt(self.loadnumpypath)
        result_y = self.model.predict(X)
        result_y = np.where(np.array(result_y) > 0.5, 1, 0)
        series = OrderedDict([('hash', name),('result_y', result_y)])
        result = pd.DataFrame.from_dict(series)
        result.to_csv(self.output, index=False, header=None)
        logger.info('{} error is occured'.format(err))
