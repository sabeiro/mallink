"""
rein_learn:
reinforcement learning agent based for optimization problem
"""

import random
import numpy as np
import pandas as pd
import matplotlib as plt

try:
    from keras.callbacks import Callback
    from keras.layers.core import Dense, Activation, Dropout
    from keras.models import Sequential
    from keras.optimizers import RMSprop
except ImportError:
    print('Keras not imported, ML functionality may be impaired')

from mallink.monte_markov import MonteMarkov


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class reinLearn(MonteMarkov):
    """reinforcement move in Monte Carlo"""

    def __init__(self, spotL, pathL=None, pairL=None, conf=None):
        """initialize the quantities for reinforcement learning"""
        MonteMarkov.__init__(self, spotL, pathL, pairL, conf)
        self.epsilon = 1.
        self.alpha = 0.1
        self.gamma = 0.6
        self.state = 0
        self.train_frame = 1000
        self.buffer_size = 500
        self.observe = 10
        self.batchSize = 6
        self.q_table = []
        self.senseL = []
        self.rewardL = []
        self.actionL = []
        self.loss_log = []
        self.nSense = len(self.sense())
        self.model = self.neural_net()

    def neural_net(self):
        """initiate the neural network"""
        net_layer = self.conf['learn']['net_layer']
        link = self.conf['learn']['link']
        nSense = self.nSense
        nAction = len(self.moveS)
        model = Sequential()
        model.add(Dense(net_layer[0], kernel_initializer='lecun_uniform', input_shape=(nSense,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(net_layer[1], kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(nAction, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)
        if self.conf['learn']['load_model'] != '':
            model.load_weights(self.conf['learn']['load_model'])
        return model

    def sense(self, move=None):
        """fill the sensor array"""
        mc_conf = self.conf['monte_carlo']
        pathv = self.pathL[self.pathL['agent']>0]
        pathv = pathv[~pathv['phantom']]
        length = np.mean(pathv['distance'])
        area = np.mean(pathv['area'])
        nrg = np.mean(pathv['energy'])
        senseD = {"energy":nrg,"completion":self.completion,"acc_rate":self.acc_rate,"lenght":length,"area":area,"step":self.step}
        senseD1 = {"cost_route":mc_conf['cost_route'],"cost_stop":mc_conf['cost_stop'],"phantom":self.phantom,"temperature":mc_conf['temperature']}
        senseL = [senseD[x] for x in senseD]
        return senseL

    def tryChange(self,m=None):
        """ reinforcement learning status change """
        mode = "reinforce"
        if np.random.uniform(0, 1) < self.epsilon:
            m = np.random.choice(self.moveS, p=self.moveP)
            mode = "explore"
        else:
            sense = np.array([self.sense()])
            qval = self.model.predict(sense, batch_size=1)
            action = qval.argmax()
            m = self.moveS[action]
            mode = "exploit"
            self.logL.append({"action":m})
            print("action",m)
        if m == "single":
            move = self.tryMove()
        elif m == "distance":
            move = self.tryDistance()
        elif m == "markov":
            move = self.tryMarkov()
        elif m == "extrude":
            move = self.tryExtrude()
        elif m == "flat":
            move = self.tryMarkovFlat()
        elif m == "outset":
            move = self.tryOutset()
        else:
            move = self.tryMove()
        status, move = self.checkAllowed(move)
        return status, move

    def updateHistory(self, move, dEn):
        """update sense history"""
        sense = self.sense()
        moveL = list(np.zeros(len(self.moveS)))
        for i, m in enumerate(self.moveS):
            if move['move'] == m:
                self.actionL.append(i)
        self.rewardL.append(dEn)
        self.senseL.append(sense)
        if len(self.rewardL) > self.buffer_size:
            self.rewardL.pop(0)
            self.senseL.pop(0)
            self.actionL.pop(0)
        self.fitState()
        if self.epsilon > 0.1:
            self.epsilon -= 1.0 / self.train_frame

    def prepareBatch(self):
        """prepare a training batch from history"""
        stateN = np.array(self.senseL[1: ])
        stateO = np.array(self.senseL[:-1])
        actionL = np.array(self.actionL[1:])
        rewardL = np.array(self.rewardL[1:])
        old_qval = self.model.predict(stateO, batch_size=self.batchSize)
        new_qval = self.model.predict(stateN, batch_size=self.batchSize)
        maxQs = np.max(new_qval, axis=1)
        X_train = stateO
        y_train = old_qval
        for i, j in enumerate(actionL):
            y_train[i, j] = rewardL[i] + self.gamma * maxQs[i]
        return X_train, y_train

    def fitState(self):
        """predict next state"""
        if self.step < self.observe:
            return False
        if (self.step % 5) != 0:
            return False
        X_train, y_train = self.prepareBatch()
        history = LossHistory()
        self.model.fit(
            X_train, y_train, batch_size=self.batchSize,
            epochs=1, verbose=0, callbacks=[history]
        )
        self.loss_log.append(history.losses)
        return True

    # def isMetropolis(self, dEn, weight=1.):
    #     """turn off metropolis acceptance"""
    #     return True

    def plotLoss(self):
        """ """
        loss = self.loss_log
        y0, y1 = [], []
        for row in loss:
            y0.append(float(row[0]))
            y1.append(float(row[1]))
        if len(y0) == 0:
            return
        window_size = 100
        window = np.ones(int(window_size))/float(window_size)
        y_av0 = np.convolve(y0, window, 'same')
        y_av1 = np.convolve(y1, window, 'same')
        arr = np.array(y_av0)
        plt.clf()  # Clear.
        plt.title("loss")
        plt.plot(y_av0[:-50])
        #plt.plot(y_av1[:-50])
        plt.ylabel('Smoothed Loss')
        plt.show()
    
    def discReward(self):
        """"""
        timeL = pd.DataFrame(self.timeL)
        def discount(k):
            return 1 / max(1, k)

        G = sum([discount(i) * x['current'] for i, x in timeL.iterrows()])

    def stateValue(self):
        """ """
        self.calcEnergy()

    def actionValue(self):
        """ """

    def saveModel(self):
        """save model to disk"""
        if self.step % 25000 == 0:
            fName = self.conf['learn']['save_model'] +  '-' + str(self.step) + '.h5'
            model.save_weights(fName,overwrite=True)
            print("Saving model %s - %d" % (fName, self.step))
