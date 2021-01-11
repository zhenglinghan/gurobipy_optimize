#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: main_process.py
@time: 2021/1/4 3:48
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

base_path = './data/'
weight = 1

# log
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == "__main__":
    # 构建模型
    # 读取数据

    for fi in os.listdir(base_path):
        varName = fi.split('.')[0]
        matrix = np.load(base_path + fi)
        print(varName, ':', matrix.shape)
        exec ('{} = matrix'.format(varName))
        exec ('{} = {}.astype(np.int64)'.format(varName, varName))

    # 问题维度
    I = [i for i in range(Y.shape[2])]
    J = [j for j in range(Y.shape[1])]
    P = [p for p in range(Y.shape[0])]
    K = [k for k in range(F.shape[-1])]
    print('problem size :', I, J, P, K)
    print('variable size :Z_ {}*{},Y_ {}'.format(len(J), len(I), len(J)))
    print('*' * 50)

    m = Model('assing_problem')
    m.setParam('MIPGap', 0.0001)
    m.setParam('TimeLimit', 100)

    # 待优化变量
    Z_ = [[0 for _ in I] for _ in J]  # J,I
    Y_ = [0 for _ in J]
    for j in J:
        for i in I:
            Z_[j][i] = m.addVar(lb=0
                                , ub=1
                                , vtype=GRB.BINARY  # 哑变量
                                , name="Z_" + str(j) + str(i)
                                )
    for j in J:
        Y_[j] = m.addVar(lb=0
                         , ub=1
                         , vtype=GRB.BINARY  # 哑变量
                         , name="Y_" + str(j)
                         )

    # 目标
    obj_min = LinExpr()
    # part 1
    sum1 = LinExpr()
    for j in J:
        sum1 += C[j] * Y_[j]
    # part 2
    sum2 = LinExpr()
    for j in J:
        for p in P:
            sum2 += A[p][j] * X[p][j]
    # part 3
    sum3 = LinExpr()
    for i in I:
        for j in J:
            for p in P:
                sum3 += Z_[j][i] * Y[p][j][i] * W[p][j][i]
    # part 4
    sum4 = LinExpr()
    for i in I:
        for j in J:
            for p in P:
                sum4 += H[j] * Y[p][j][i] * Z_[j][i]
    obj_min = sum1 + sum2 + sum3 + sum4

    obj_max = LinExpr()
    # part 1
    sum1_ = LinExpr()
    for p in P:
        for i in I:
            for j in J:
                sum1_ += Y[p][j][i] * Rij[i][j]

    sum2_ = LinExpr()
    for p in P:
        for i in I:
            for k in K:
                sum2_ += F[p][i][k] * Ri[i]

    obj_max = -1 * (sum1_ + sum2_)

    # 优化目标
    obj = weight * obj_min + (1 - weight) * obj_max
    m.setObjective(obj, GRB.MINIMIZE)

    # 约束
    # part1 对所有的p  R0~R6
    for p in P:
        c_1_ = LinExpr()
        for j in J:
            c_1_ += X[p][j]
        for j in J:
            for i in I:
                c_1_ -= Y[p][j][i]  # 常数约束 其实没用
        m.addConstr(c_1_ == 0)

    # part2 对所有的p,j 常数约束 其实没用 R7~R34
    for p in P:
        for j in J:
            c_2_ = LinExpr()
            c_2_ -= E[p][j]
            for i in I:
                c_2_ += Y[p][j][i]
            m.addConstr(c_2_ <= 0)

    # part3 对所有的p R35~R41 常数约束 没有用
    for p in P:
        c_3_ = LinExpr()
        c_3_ -= G[p]
        for j in J:
            c_3_ += X[p][j]
        m.addConstr(c_3_ <= 0)

    # part4 U~V R42~R45
    c_4_min = LinExpr()
    for j in J:
        c_4_min += Y_[j]
    m.addConstr(c_4_min >= U)
    # R46~R49
    c_4_max = LinExpr()
    for j in J:
        c_4_max += Y_[j]
    m.addConstr(c_4_max <= V)

    # part5 R50~%53
    for i in I:
        c_5_ = LinExpr()
        for j in J:
            c_5_ += Z_[j][i]
        m.addConstr(c_5_ == 1)

    # part6 qly~quY R54~R57
    for j in J:
        c_6_min_ = LinExpr()
        c_6_min_ += Ql * Y_[j]
        for i in I:
            c_6_min_ -= Z_[j][i]
        m.addConstr(c_6_min_ <= 0)
    # R58~R61
    for j in J:
        c_6_max_ = LinExpr()
        c_6_max_ -= Qu * Y_[j]
        for i in I:
            c_6_max_ += Z_[j][i]
        m.addConstr(c_6_max_ <= 0)
    #
    # part7 Ypij*z<=b 修改 # R62~R65
    for j in J:
        c_7_ = LinExpr()
        c_7_ -= B[j]
        for p in P:
            for i in I:
                c_7_ += Y[p][j][i] * Z_[j][i]  # y为给定常数
        m.addConstr(c_7_ <= 0)
    #
    # part8 sy+dz<=fi i个约束 # R66~R69
    for i in I:
        c_8_ = LinExpr()
        for j in J:
            c_8_ += S[j] * Y_[j]
            c_8_ -= D[j][i] * Z_[j][i]
        c_8_ -= F_[i]
        m.addConstr(c_8_ <= 0)
    #
    # part9 Y>fpik # R70~R73
    for i in I:
        c_9_ = LinExpr()
        for p in P:
            for k in K:
                c_9_ += F[p][i][k]
            for j in J:
                c_9_ -= Y[p][j][i]
        m.addConstr(c_9_ <= 0)

    # 求解
    m.write('assign.lp')
    m.optimize()
    if m.solCount == 0:
        print("Model is infeasible")
        m.computeIIS()
        m.write("model_iis.ilp")
    print('optimal value: %d' % obj.getValue())
    print('optimal solution:{} {}'.format(Z_, Y_))
    Z_solution = np.array([[0 for _ in I] for _ in J])
    Y_solution = np.array([0 for _ in J])
    for j in J:
        for i in I:
            Z_solution[j][i] = Z_[j][i].x
    for j in J:
        Y_solution[j] = Y_[j].x
    print('Z:')
    print(Z_solution)
    print('Y:')
    print(Y_solution)
