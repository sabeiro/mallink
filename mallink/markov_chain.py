"""
markov_chain:
creation of Markov chains and Bayesian inference
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import spatial
import matplotlib.pyplot as plt
import geomadi.graph_viz as g_v
import geomadi.geo_octree as g_o
import networkx as nx

gO = g_o.h3tree()

class MarkovChain():
    """Monte Carlo for path optimization"""
    def __init__(self,spotL,pairL=None,confM=None):
        """initialize quantities and Markov chains"""
        self.spotL = spotL
        if confM == None:
            self.confM = {"exp":1,"threshold":"row","leg":7,"doc":"threshold:[percentile,mean,len,row]"}
        else:
            self.confM = confM
        self.loadRouted(pairL)
        self.markovC = self.defineMarkov(spotL)
        self.sampDis = np.zeros([spotL.shape[0],spotL.shape[0]])
        self.stateL = spotL.index

    def defineMarkov(self,spotL,isPlot=False):
        """define Markov chains from"""
        distM = self.distM.copy()
        distM[distM > 0.06] = 0.
        markovC = 1./self.distM
        expo = int(self.confM['exp'])
        threshold = self.confM['threshold']
        leg = self.confM['leg']
        markovC.replace(float('inf'),0,inplace=True)
        markovC.replace(float('nan'),0,inplace=True)
        markovC = markovC/markovC.sum(axis=0)
        markovC = markovC**expo
        markovC = markovC/markovC.sum(axis=0)
        mL = markovC.values.ravel()
        mL = mL[mL>0]
        m = 1./(len(markovC))
        if threshold == "percentile":
            m = np.percentile(mL,99)
        if threshold == "mean":
            m = np.mean(mL)
        if threshold == "row":
            m = np.array([np.percentile(x,(1-leg/len(markovC))*100) for i,x in markovC.iteritems()])
            m[np.isnan(m)] = 2.
            
        markovC[markovC<m] = 0.
        markovC = markovC/markovC.sum(axis=0)
        print("mean links per node %.1f threshold %.1e" % ((markovC>0).sum().mean(),np.mean(m)))
        return markovC

    def updateSampling(self,s1,s2):
        """update frequency table"""
        self.sampDis[s1,s2] = self.sampDis[s1,s2] + 1.

    def samplingDistribution(self):
        """return the sampling distribution"""
        return self.sampDis/self.sampleDis.sum()

    def norm(self,p):
        """return a correct normalized probability"""
        p[p!=p] = 0.
        p = p.abs()
        p = p.replace(float('inf'),0.)
        p = p.replace(float('nan'),0.)
        if p.sum() > 0.: p = p/p.sum()
        else: p.loc[:] = 1./len(p)
        return p
    
    def moveProb(self,s):
        """move probability from state s"""
        return self.markovC.loc[:,s]

    def moveSample(self,s):
        """find a move from the sampling probability"""
        sp = self.moveProb(s)
        s1 = np.random.choice(self.stateL,p=sp)
        return s1
        
    def loadRouted(self,pairL):
        """load a routed pair relationship between spots"""
        if not isinstance(pairL,pd.DataFrame):
            pos = self.spotL[['x','y']].sort_index()
            self.distM = pd.DataFrame(sp.spatial.distance_matrix(pos.values,pos.values),index=pos.index,columns=pos.index)
        else: 
            odm = pairL.pivot_table(index="geohash_o",columns="geohash_d",values="length",aggfunc=np.sum)
            odw = pairL.pivot_table(index="geohash_o",columns="geohash_d",values="weight",aggfunc=np.sum)
            odm.replace(float('nan'),10000.,inplace=True)
            odw.replace(float('nan'),0.,inplace=True)
            self.distM = odm
            self.markovC = odw
        
    def plotMarkov(self):
        """Display Markov chain graph"""
        pos = self.spotL[['x','y']].sort_index()
        g_v.graphAdjacency(self.markovC,pos)
        cumP = markovC.cumsum(axis=0)
        cumP = pd.concat([cumP.head(1),cumP],axis=0)
        cumP.iloc[0] = 0
        P = cumP[cumP<0.99999999]
        m = cumP.idxmax().sort_values()
        idx = pd.DataFrame({"first":P.isna().sum(axis=0)})
        idx.sort_values("first",inplace=True)
        P = cumP.loc[cumP.index,cumP.index]
        plt.imshow(P)
        plt.show()

    def getDict(self):
        """export markov chain as a graph"""
        edgeL = {}; nodeL = {}
        for i,g in self.markovC.iterrows():
            m = g[g>0]
            x, y = gO.decode(i)
            nodeL[i] = {"x":x,"y":y,"name":i}
            if m.shape[0] == 0: continue
            for j, k in m.iteritems():
                if not i in edgeL: edgeL[i] = {}
                edgeL[i][j] = k

        net = {"nodes":nodeL,"edges":edgeL}
        return net

    def getGraph(self):
        """export markov chain as a graph"""
        G = nx.DiGraph()
        for i,g in self.markovC.iterrows():
            m = g[g>0]
            x, y = gO.decode(i)
            G.add_node(i,x=x,y=y,name=i)
            if m.shape[0] == 0: continue
            for j, k in m.iteritems():
                G.add_edge(i,j,weight=k)
        return G
