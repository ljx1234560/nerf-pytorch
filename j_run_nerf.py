#基本库
import os, sys
import numpy as np
import imageio
import json
import random
import time

#jittor
import jittor as jt
import jittor.nn as nn

#功能库
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from run_nerf_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

jt.flags.use_cuda = 0  
np.random.seed(0)
jt.seed(0)  
DEBUG = False
##print("Jittor是否用CUDA：", jt.flags.use_cuda)  # 输出0则为CPU

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return jt.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

