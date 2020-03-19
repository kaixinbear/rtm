from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.getcwd(),"lib/utils"))
import energy
import numpy as np

def optimize(input1,input2,input3,score,hm_score):
    print("input1",input1)
    input2 = np.concatenate([input2[::2],input2[1::2]])
    print("input2",input2)
    print("input3",input3)
    score = np.array([score])
    print("score",score)
    print("hm_score",hm_score)
    optimized_result = energy.optimize(input1,input2,input3,score,hm_score)
    print('optimized_result', optimized_result)
    print('-'*50)
    return optimized_result
