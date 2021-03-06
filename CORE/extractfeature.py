"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
from core import PEFeatureExtractor
import tqdm
import jsonlines
import pandas as pd
import multiprocessing
from . import utility
import logging
import time

logger = logging.getLogger(__name__)

#console handler
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

# add logging handler
logger.addHandler(handler)
logger.addHandler(f_handler)

class Extractor:
    def __init__(self, datadir, label, output, features):
        self.datadir = datadir
        self.output = output
        self.data = pd.read_csv(label, names=['hash', 'value'])
        self.features = features

    def extract_features(self, sample):
        """
        Extract features.
        If error is occured, return None Object
        """
        extractor = PEFeatureExtractor(self.features)
        fullpath = os.path.join(os.path.join(self.datadir, sample))
        try:
            binary = open(fullpath, 'rb').read()
            feature = extractor.raw_features(binary)
            feature.update({"sha256": sample}) # sample name(hash)
            #time.sleep(3)
            feature.update({"label" : self.data[self.data.hash==sample].values[0][1]}) #label
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:  
            logger.error('{}: {} error is occuered'.format(sample, e))            
            return None

        return feature

    def extract_unpack(self, args):
        """
        Pass thorugh function unpacking arguments
        """
        return self.extract_features(args)

    def extractor_multiprocess(self):
        """
        Ready to do multi Process
        Note that total variable in tqdm.tqdm should be revised
        Currently, I think that It is not safely. Because, multiprocess pool try to do FILE I/O.
        """
        pool = multiprocessing.Pool(48)
        queue = multiprocessing.Queue()
        queue.put('safe')
        end = len(next(os.walk(self.datadir))[2])
        error = 0

        extractor_iterator = ((sample) for idx, sample in enumerate(utility.directory_generator(self.datadir)))
        with jsonlines.open(self.output, 'w') as f:
            for x in tqdm.tqdm(pool.imap_unordered(self.extract_unpack, extractor_iterator), total=end):
                if not x:
                    """
                    To input error class or function
                    """
                    error += 1
                    continue
                msg = queue.get()
                if msg == 'safe': 
                    f.write(x)                
                    queue.put('safe')

        pool.close()

    def run(self):
        self.extractor_multiprocess()
