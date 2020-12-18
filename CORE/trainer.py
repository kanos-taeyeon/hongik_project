"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import argparse
import os
import core
import sys
import pickle
import jsonlines
import logging
import numpy as np
import pandas as pd
from . import utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import pickle
from sklearn.metrics import accuracy_score

logger = logging.getLogger('Training')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-5s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ModelType(object):
    def train(self):
        raise (NotImplemented)

    def save(self):
        raise (NotImplemented)

class RandomForest(ModelType):
    def __init__(self, output, rows, dim):
        self.datadir = os.path.dirname(output)
        self.output = output
        self.rows = rows
        self.dim = dim
        self.model = None


class Gradientboosted(ModelType):
    """
    Train the LightGBM model from the vectorized features
    """
    def __init__(self, output, rows, dim):
        self.datadir = os.path.dirname(output)
        self.output = output
        self.rows = rows
        self.dim = dim
        self.model = None

    """
    Run Gradientboost algorithm which in lightgbm
    """
    def train(self):
        """
        Train
        """
        X, y = core.read_vectorized_features(self.datadir, self.rows, self.dim)
        lgbm_dataset = lgb.Dataset(X, y)
        self.model = lgb.train({"application": "binary"}, lgbm_dataset)

    def save(self):
        """
        Save a model using a pickle package
        """

        logger.debug('[GradientBoosted] start save')
        logger.debug(self.model)
        if self.model:
            self.model.save_model(self.output)
        logger.debug('[GradientBoosted] finish save')

class Trainer:
    def __init__(self, jsonlpath, output):
        self.jsonlpath = jsonlpath
        self.output = output
        self.outputdir = os.path.dirname(output)
        self.rows = 0
        self.model = None
        featurelist = utility.readonelineFromjson(jsonlpath)
        featuretype = utility.FeatureType()
        self.features = featuretype.parsing(featurelist)
        self.dim = sum([fe.dim for fe in self.features])

    def vectorize(self):
        # To do Error check
        # if file is jsonl file
        if self.rows == 0:
            logger.info('[Error] Please check if jsonl file is empty ...')
            return -1
        #core.create_metadata(self.jsonlpath)
        core.create_vectorized_features(self.jsonlpath, self.outputdir, self.rows, self.features, self.dim)

    def update_rows(self):
        """
        Update a rows variable
        """
        with jsonlines.open(self.jsonlpath) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                self.rows += 1

    def removeExistFile(self):
        """
        Remove Files
        """
        path_X = os.path.join(self.outputdir, "X.dat")
        path_y = os.path.join(self.outputdir, "y.dat")

        if os.path.exists(path_X):
            os.remove(path_X)
        if os.path.exists(path_y):
            os.remove(path_y)

        with open(path_X, 'w') as f:
            pass
        with open(path_y, 'w') as f:
            pass

    def run(self):
        """
        Training
        """
        # self.removeExistFile()
        self.update_rows()
        if self.vectorize() == -1: return


        logger.debug('Start model_train')
        # Training
        gradientboostmodel = Gradientboosted(self.output, self.rows, self.dim)
        gradientboostmodel.train()
        gradientboostmodel.save()

        self.removeExistFile()
