import numpy as np
import pandas as od
import tensorflow as tf
import random
import math
import csv
import os
import sklearn
from autograd import grad
import eytan #as WhyIsThisGoddamnLuaPackageNotWorking 
#Eytan will write some code and I will use the stuff he writes under the eytan package
class nickNet():
    def __init__(self, X, Y, layers, hiddenSize):
        self.X = X
        self.Y = Y
        self.input = X[0].size()
        self.output = 1
        self.layers = layers
        self.hidden = hiddenSize
        self.W1 = np.random.randn(self.input, self.hidden)
        self.Wfinal = np.random.randn(self.hidden,self.output)
        self.Wmid = np.array((), dtpye = float)
        for k in range(layers - 1):
            if (k%2 != 0):
                np.append(Wmid,wfinal.T)
            else:
                np.append(Wmid,wfinal)
       
    def activate(self, num):
        return 1/(1+np.exp(-num))
    def netRun(self):
        final = []
        for k in range(X.size):
            result = np.dot(X[k],W1)
            result = self.activate(result)
            for k in rnage(Wmid.size()):
                result = np.dot(result, wMid(k))
                result = self.activate(result)
            result = np.dot(result, self.wfinal)
            result = activate(result)
            final.append(result)
        return final
    def net(self, var):
        return 7
    def cost(self):
        out = netRun(X)#Make a method that runs every input
        for k in range(len(out)):
            tot+= (out[k]-Y[k])**2
        tot*=.5
        return tot
    def costPrime(self):
        return grad(self.cost())
    def mutate():
        self.W1 += self.W1*.01*random.choice((-1,1))
        for k in range(Wmid.size()):
            Wmid[k] += self.Wmid*.01*random.choice((-1,1))
        self.Wfinal += self.Wfinal*.01*random.choice((-1,1))
        
class Trainer():
    def __init__(self, NN, tol):
        self.NN = NN
        self.tol = tol
    def optimize(self):
        while (sum(abs(self.NN.costPrime())) > tol):
            return 7
            
        
class VarEliminator():
    def __init__(self, X, Y, NN, Train):
        self.X = X
        self.Y = Y
        self.NN = NN
        self.Train = Train
        self.Result = NN.netRun(X)
    def extranious(self):
        k = 0
        while(k < X.size()):
            tempX = self.X.delete(k)
            newNet = nickNet(tempX, self.Y, self.NN.layers, self.NN.hidden)
            Train = Trainer(newNet, 1)
            Train.train()
            test = Train.NN.netRun()
            if(sklearn.metrics.accuracy_score(self.Y, test) >= sklearn.metrics.accuracy_score(self.Y, self.Result) -.05):
                self.X = self.X.delete(k)
                k-=1
            k+=1
        return Train.NN
#inputs = [u, d, l, r, a, b, x, y, rb, lb, upLeft, upright, upa, upb, upx, upy, uprb, uplb, downleft, downright, downa, downb, downx, downy, downrb, downlb. lefta, leftb, leftx, ly, leftrb, leftlb, righta, rightb, rightx,righty,rightlb,rightrb] 
#nums  = [0,1,2,3,4,5,||||6,7,8,9,10,11,12,13,14,15,16,17]

def playGame(NN):
    inputs =['u', 'd', 'l', 'r', 'a', 'b', 'x', 'y', 'q', 'p', 'ud', 'ul', 'ur', 'ua', 'ub', 'ux', 'uy', 'uq', 'up', 'dl', 'dr', 'da', 'db', 'dx', 'dy', 'dq', 'dp', 'lr', 'la', 'lb', 'lx', 'ly', 'lq', 'lp', 'ra', 'rb', 'rx', 'ry', 'rq', 'rp', 'ab', 'ax', 'ay', 'aq', 'ap', 'bx', 'by', 'bq', 'bp', 'xy', 'xq', 'xp', 'yq', 'yp', 'qp', 'udl', 'udr', 'uda', 'udb', 'udx', 'udy', 'udq', 'udp', 'ulr', 'ula', 'ulb', 'ulx', 'uly', 'ulq', 'ulp', 'ura', 'urb', 'urx', 'ury', 'urq', 'urp', 'uab', 'uax', 'uay', 'uaq', 'uap', 'ubx', 'uby', 'ubq', 'ubp', 'uxy', 'uxq', 'uxp', 'uyq', 'uyp', 'uqp', 'dlr', 'dla', 'dlb', 'dlx', 'dly', 'dlq', 'dlp', 'dra', 'drb', 'drx', 'dry', 'drq', 'drp', 'dab', 'dax', 'day', 'daq', 'dap', 'dbx', 'dby', 'dbq', 'dbp', 'dxy', 'dxq', 'dxp', 'dyq', 'dyp', 'dqp', 'lra', 'lrb', 'lrx', 'lry', 'lrq', 'lrp', 'lab', 'lax', 'lay', 'laq', 'lap', 'lbx', 'lby', 'lbq', 'lbp', 'lxy', 'lxq', 'lxp', 'lyq', 'lyp', 'lqp', 'rab', 'rax', 'ray', 'raq', 'rap', 'rbx', 'rby', 'rbq', 'rbp', 'rxy', 'rxq', 'rxp', 'ryq', 'ryp', 'rqp', 'abx', 'aby', 'abq', 'abp', 'axy', 'axq', 'axp', 'ayq', 'ayp', 'aqp', 'bxy', 'bxq', 'bxp', 'byq', 'byp', 'bqp', 'xyq', 'xyp', 'xqp', 'yqp']
    ramCommand = pd.readcsv('isThereANewFrame.csv')
    command = ramCommand.values
    while(True):
        if(not np.array_equal(pd.readscv('isThereANewFrame.csv').value, command):
           command = pd.readscv('isThereANewFrame.csv').vsalue
           Mario = NN.net(command[command.size-1])
           


    
    # [up, down, left, right, a, b, x, y, rb, lb] [0 = off, 1 = on]
         
def main():  
    data_df = pd.readcsv('ramValuesAndInputs.csv')
    data = data_df.values
    #Eliminate all values which don't change much
    for k in range(data.size()):
        if(data[k].var() < 5):
            data.delete(k)
            k-=1
    NN = nickNet(data, data[5], 2, 4)
    Train = Trainer(NN, 1)
    NN = Train.optimize()
    howFar = eytan.playGame(NN)
    marios = []
    marios.append(NN)
    indexBest = 0
    for k in range(9):
        marios.append(clone.deepclone(NN.mutate()))
    for k in range(1000):
        for i in range(10):
            run = eytan.playGame(marios[k])
            if(run > howFar):
                howFar = run
                indexBest = i
        Best = clone.deepclone(marios[indexBest])
        marios = []
        marios.append(Best)
        for i in range():
            marios.append(clone.deepclone(Best.mutate()))
main()