import errno
import logging
import os
from datetime import datetime
from time import time
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

def set_file_handler(path,dataset,seeds,method):
    log_time = datetime.now().strftime("%y-%m-%d_%H:%M")
    log_fpath = path + method + "_" + dataset + "-" + seeds + "_" + log_time + "_result.log"
    file_handler = logging.FileHandler(log_fpath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
