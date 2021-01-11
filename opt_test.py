#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: opt_test.py
@time: 2021/1/11 21:54
"""

import pandas as pd
import numpy as np
import os
import gc
import datetime as dt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)


from gurobipy import *

m = Model()
m.setParam('nonconvex', 2) # for nonconvex problem
# solve the problem :gurobipy.GurobiError: Objective Q not PSD (diagonal adjustment of 1.0e+00     would be required)
#variables
x1 = m.addVar(vtype=GRB.CONTINUOUS, lb=20, ub=30, name="x1")
x2 = m.addVar(vtype=GRB.CONTINUOUS, lb=20, ub=30, name="x2")

m.update()

m.setObjective(x1 * x2, GRB.MAXIMIZE)  # or GRB.MAXIMIZE

m.update()

m.optimize()

m = Model()
m.setParam('nonconvex', 2) # for nonconvex problem
#variables
x1 = m.addVar(vtype=GRB.CONTINUOUS, lb=20, ub=30, name="x1")
x2 = m.addVar(vtype=GRB.CONTINUOUS, lb=20, ub=30, name="x2")

m.update()

m.setObjective(x1 * x2, GRB.MINIMIZE)  # or GRB.MAXIMIZE

m.update()

m.optimize()