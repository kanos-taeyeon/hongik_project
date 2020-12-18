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
import multiprocessing

logger = logging.getLogger('Testing')
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
pool = multiprocessing.Pool(48)

class Predictor:
    def __init__(self, main_modelpath,sub_modelpath, testdir, features, sub_feature, resultoutput):
        # load model with pickle to predict
        #with open(main_modelpath, 'rb') as f:
        #    #self.model = lgb.Booster(model_file=modelpath)
        #    self.main_model = pickle.load(open(main_modelpath, 'rb'))
        #with open(sub_modelpath, 'rb') as f:
        #    #self.model = lgb.Booster(model_file=modelpath)
        #    self.sub_model = pickle.load(open(sub_modelpath, 'rb'))
        with open(main_modelpath, 'rb') as f:
            self.main_model = lgb.Booster(model_file=main_modelpath)
        with open(sub_modelpath, 'rb') as f:
            self.sub_model = lgb.Booster(model_file=sub_modelpath)
        self.testdir = testdir
        self.resultoutput = resultoutput
        self.features = features
        self.sub_feature = sub_feature

    def run(self):
        y_pred = []
        name = []
        err = 0
        end = len(next(os.walk(self.testdir))[2])
        for sample in tqdm.tqdm(utility.directory_generator(self.testdir), total=end):
            fullpath = os.path.join(self.testdir, sample)#dir connect
            #print(fullpath)

            if os.path.isfile(fullpath):
                binary = open(fullpath, "rb").read()
                name.append(sample)

                try:
                    y_p= core.predict_sample(self.main_model,self.sub_model, binary, self.features,self.sub_feature)
                    y_pred.append(y_p)
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    logger.error('{}: {} error is occuered'.format(sample, e))
                    y_pred.append(0)
                    err += 1

        y_pred = np.array(y_pred)
        series = OrderedDict([('hash', name),('y_pred', y_pred)])
        result = pd.DataFrame.from_dict(series)
        result.to_csv(self.resultoutput, index=False)
        logger.info('{} error is occured'.format(err))
