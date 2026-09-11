import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve,auc,average_precision_score,precision_recall_curve
import cv2
import pandas as pd
from label_encoding import encode_test_labels,encode_metadata_labels

from Logger import Logger
import pandas as pd
from dependency import *
import torch
import torch.nn as nn
import sklearn

def predict(net2, net, test_index, df, model_name, out_dir, mode, TTA=4, size=229, img_type='clinic', data_mode='img'):
    os.makedirs(out_dir, exist_ok=True)
    log= Logger()
    log.open(out_dir + 'log.multi_modality_{}_{}_sinlesion.txt'.format(mode, model_name), mode='w')
    log.write('\n-- [START %s] %s\n\n' % ('IDENTIFIER', '-', * 64))
    net.set_mode('valid')
    net2.set_mode('valid')