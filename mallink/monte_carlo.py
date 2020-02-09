"""
monte_carlo:
Monte Carlo library for paths optimization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geomadi.graph_viz as g_v
import geomadi.lib_graph as l_g
import scipy as sp
from mallink.path_opt import pathOpt
import time

class MonteCarlo(pathOpt):
    """Monte Carlo for path optimization"""
    def __init__(self,spotL,pathL=None,pairL=None,conf=None):
        """initialize quantities"""
        self.step = 1
        self.moveN = 0
        self.timeL = []
        self.moveL = []
        self.perfL = []
        self.acc_move = 0.
        self.acc_rate = 0.
        self.chemPot = 0.
        self.En, self.cost, self.revenue = 0, 0, 0
        self.start_time = time.time()
        pathOpt.__init__(self,spotL,pathL=pathL,pairL=pairL,conf=conf)
        self.initProb()
        self.En = self.calcEnergy(self.spotL)
        print('initial energy %.0f' % (self.En))
        self.start = time.time()
        

    def updateConf(self):
        """update configuration"""
        super().updateConf()
        self.conf['simulation']['step'] = self.step
        self.conf['simulation']['energy'] = self.En
        self.conf['simulation']['acceptance_move'] = self.acc_move
        self.conf['simulation']['acceptance_rate'] = self.acc_rate
        self.conf['simulation']['chem_pot'] = self.chemPot
        self.conf['simulation']['cost'] = self.cost
        self.conf['simulation']['revenue'] = self.revenue
        
        
    def initProb(self):
        """init probabilities"""
        l = self.conf['moveProb']
        l = [l[k] for k in l.keys()]
        self.moveP = [x/sum(l) for x in l]
        self.moveS = list(self.conf['moveProb'].keys())
        l = self.conf['chainProb']
        l = [l[k] for k in l.keys()]
        self.chainP = [x/sum(l) for x in l]
        self.chainS = list(self.conf['chainProb'].keys())

        
    def calcEnergy(self,route):
        """calculate the route energy"""
        route1 = route[route['agent'] > 0]
        if route1.shape[0] == 0: return 0.0
        mc_conf = self.conf['monte_carlo']
        length = sum(route1['distance'])
        stop = route1[route1['agent']>0].shape[0]
        cost = length*mc_conf['cost_route'] + stop*mc_conf['cost_stop']
        chem, area, sepa = self.calcEnergyCh()
        c_area = mc_conf['cost_area']*area/self.len_mean
        c_sepa = mc_conf['cost_separation']*sepa/self.len_mean
        revenue = route1['potential'].sum()
        energy = revenue - cost
        energy = energy + c_sepa - c_area
        self.cost, self.revenue = cost, revenue
        self.timeL.append({"step":self.step,"length":length,"stop":stop,"revenue":revenue,"energy":energy,"cost":cost,"current":self.En})
        return energy
    

    def calcEnergyCh(self):
        """calculate the route energy"""
        mc_conf = self.conf['monte_carlo']
        costP = self.pathL['distance']*mc_conf['cost_route'] + self.pathL['load']*mc_conf['cost_stop']
        self.pathL['energy'] = self.pathL['potential'] - costP
        pathv = self.pathL[self.pathL['agent'] > 0]
        pathv = pathv.loc[~pathv['phantom']]
        area = np.sqrt(np.sum(pathv['area']))
        pos = pathv[['x_c','y_c']]
        distA = sp.spatial.distance_matrix(pos.values, pos.values)
        sepa = distA.sum().sum()
        self.chemPot = np.mean(pathv['energy'])
        return self.chemPot, area, sepa

    def isMetropolis(self,dEn,weight=1.):
        """metropolis criterium"""
        temp = self.conf['monte_carlo']['temperature']
        if dEn/temp > 3.: return True
        if weight*np.exp(-dEn/temp) < np.random.uniform(0,1): return True
        return False

    def move(self,move):
        """actuate a move"""
        route1 = self.spotL.copy()
        p, v, s = move['state'], move['agent'], move['action']
        for i,j in enumerate(p):
            route1.loc[j,'agent'] = v[i]
            route1.loc[j,'ops'] = s[i]
        route1 = self.simplifyRoute(route1)
        return route1

    def checkEnergy(self,route,move):
        """check energy"""
        En1 = self.calcEnergy(route)
        ts = time.time() - self.start_time
        dEn = (En1 - self.En)
        string = "step: %d, time: %d, En: %.2f, dEn: %.2f, completion: %.2f, phantom: %d" % (self.step,ts,self.En,dEn,self.completion,self.phantom)
        print(string,end="\r",flush=True)
        self.logL.append({"energy":string})
        return En1

    def loop(self,m=None):
        """a single iteration for loop"""
        self.grandCanonical()
        start = time.time()
        self.step += 1
        status, move = self.tryChange(m)
        move_time = time.time() - start
        self.logL.append({"move":move['move']})
        if not status:
            self.updateHistory(move,-10)
            return False
        route1 = self.move(move)
        En1 = self.checkEnergy(route1,move)
        nrg_time = time.time() - start
        dEn = (En1 - self.En)
        self.updateHistory(move,dEn)
        status = self.isMetropolis(dEn,weight=move['weight'])
        self.logL.append({"metropolis":"%.2f %d" % (dEn,1*status)})
        if not status: return False
        self.En = En1
        self.updateSys(route1)
        self.moveN += 1
        end_time = time.time() - start
        self.moveL.append([self.step,end_time,move['move'],self.En,self.cost,self.revenue,self.completion])
        self.perfL.append([move_time,nrg_time,end_time])
        self.acc_move += 1.
        self.acc_rate = self.acc_move/self.step
        return True

    def updateHistory(self,move,dEn):
        """update learning history"""
        return False

    def probNorm(self, p):
        """return a correct normalized probability"""
        # p[p!=p] = 0.
        p = p.abs()
        p = p.replace(float('inf'), 0.)
        p = p.replace(float('nan'), 0.)
        if p.sum() > 0.:
            p = p / p.sum()
        else:
            p.loc[:] = 1. / len(p)
        return p

    def probPath(self):
        """weight for favoring incomplete paths"""
        agentL = self.pathL[self.pathL['agent'] > 0]
        pv = agentL['completion']
        pv = 1.5 - pv
        return self.probNorm(pv)
    
    def tryChange(self,m=None):
        """perform a move and try to accept it"""
        if m == None:
            m = np.random.choice(self.moveS,p=self.moveP)
        if   m == "single": move = self.tryMove()
        elif m == "distance": move = self.tryDistance()
        elif m == "outset": move = self.tryOutset()
        else: move = self.tryMove()
        status, move = self.checkAllowed(move)
        return status, move

    def tryMove(self):
        """try a move favoring uncomplete"""
        pv = self.probPath()
        ps = self.probNorm(self.spotL['occupancy'])
        v1 = np.random.choice(pv.index,p=pv)
        p1 = np.random.choice(ps.index,p=ps)
        s1 = np.random.choice(self.opsL)
        weight = 1. # - ps[s1] - pv[v2]
        return {"weight":weight,"agent":[v1],"state":[p1],"action":[s1],"move":"uniform"}

    def tryRemove(self):
        """try a move favoring uncomplete"""
        route1 = self.spotL[spotL['agent']>0]
        pv = self.probPath()
        ps = self.probNorm(route1['occupancy'])
        v1 = 0
        p1 = np.random.choice(ps.index,p=ps)
        s1 = np.random.choice(self.opsL)
        weight = 1. # - ps[s1] - pv[v2]
        return {"weight":weight,"agent":[v1],"state":[p1],"action":[s1],"move":"uniform"}

    def tryDistance(self):
        """try a move favouring reducing distances"""
        pv = self.probPath()
        v1 = np.random.choice(pv.index,p=pv)
        routev = self.spotL[self.spotL['agent'] == v1]
        if routev.shape[0] == 0:
            ps = self.probNorm(self.spotL['occupancy'])
            p1 = np.random.choice(ps.index,p=ps)
            routev = self.spotL.loc[p1:p1]
        ps = self.probNorm(routev['distance'])
        p1 = np.random.choice(ps.index,p=ps)
        s1 = np.random.choice(self.opsL)
        weight = 1. # - pv[v1] - ps[s1]
        return {"weight":weight,"agent":[v1],"state":[p1],"action":[s1],"move":"dist"}

    def tryOutset(self,radius=0.01):
        """try outsetting the path"""
        agentL = self.pathL[self.pathL['agent'] > 0]
        agentL = agentL[agentL['completion'] > 0]
        if len(agentL) == 0:
            return {"weight":1.,"agent":[],"state":[],"action":[],"move":"outset"}
        pv = agentL['completion']
        pv = 1. - pv
        pv = self.probPath()
        pv = pv[pv.index > 0]
        v1 = np.random.choice(pv.index,p=pv)
        neiL = self.outsetPath(v1,radius=radius)
        s1 = np.random.choice(self.opsL)
        vL = [v1 for x in neiL]
        sL = [s1 for x in neiL]
        weight = 1.
        return {"weight":weight,"agent":vL,"state":neiL,"action":sL,"move":"outset"}

    
    def grandCanonical(self,m=None):
        """insert or remove paths in case of local minima"""
        #if (self.step % 10) != 0: return False
        if m == None:
            m = np.random.choice(self.chainS,p=self.chainP)
        if   m == "swap": status = self.trySwapCh()
        elif m == "insert": status = self.tryInsertCh()
        elif m == "remove": status = self.tryRemoveCh()
        return status

    
    def tryInsertCh(self):
        """try inserting a phantom chain"""
        v = max(self.agentL) + 1
        route = self.startPos(complete="all",agentL=[v]) ## TODO: slow
        En1 = self.calcEnergy(route[route['agent'] == v])
        dEn = En1 - self.chemPot
        print({"chain":"insertion %.2f %.2f %.2f" % (En1,self.chemPot,dEn)})
        self.logL.append({"chain":"insertion %.2f %.2f %.2f" % (dEn,En1,self.chemPot)})
        if not self.isMetropolis(dEn,weight=.3): return False
        print("+++ inserted phantom %d" % (v))
        self.spotL = route
        self.insertPhantom(n=1)
        return True
        
        
    def tryRemoveCh(self):
        """try inserting a phantom chain"""
        pathP = self.pathL[self.pathL['phantom']]
        if len(pathP) == 0: return False
        p = np.random.choice(pathP.index)
        dEn = pathP.loc[p,'energy'] - self.chemPot
        print({"chain":"deletion %.2f %.2f %.2f" % (pathP.loc[p,'energy'],self.chemPot,dEn)})
        if not self.isMetropolis(dEn,weight=.3): return False
        self.removePhantom(n=1,v=p)
        print("--- removed phantom %d" % (p))
        return True

    
    def trySwapCh(self):
        """look whether is convenient to swap a path"""
        pathv = self.pathL[self.pathL['agent'] > 0]
        idx = pathv['phantom']
        pathP = pathv.loc[idx].sort_values('energy')
        pathN = pathv.loc[~idx].sort_values('energy')
        if pathN.shape[0]*pathP.shape[0] == 0:
            return False
        vp = np.random.choice(pathP.index)
        vn = np.random.choice(pathN.index)
        vp = pathP.iloc[0]['agent']
        vn = pathN.iloc[-1]['agent']
        pathP = self.pathL.loc[vp]
        pathN = self.pathL.loc[vn]
        if pathP['energy']*pathP['load'] > pathN['energy']*pathN['load']: return False
        print("~~~ phantom swap (%d:%d/%.0f)->(%d:%d/%.0f)" % (vp,pathP['energy'],pathP['load'],vn,pathN['energy'],pathN['load']))
        status = self.swapPath(vp,vn)
        #self.restartPath(vp)
        return status

    def plotHistory(self,ax=None):
        """plot energy distribution over time"""
        timeL = pd.DataFrame(self.timeL)
        if len(timeL) == 0: return ax
        if ax == None: fig, ax = plt.subplots(1,1)
        ax.set_title("energy distribution")
        ax.plot(timeL['revenue'],label="revenue")
        ax.plot(timeL['energy'],label="energy")
        ax.plot(timeL['cost'],label="cost")
        ax.plot(timeL['current'],label="chosen")
        #ax.legend()
        return ax

    def printSummary(self):
        """print summary of the simulation"""
        summ = {"step":self.step}
        summ['total_time'] = time.time() - self.start
        summ['acceptance'] = len(self.timeL)/self.step
        summ['time_move']  = summ['total_time']/self.step
        summ['completion'] = self.completion
        summ['move_col'] = ['step','time','move','energy','cost','revenue','completion']
        summ['move'] = self.moveL
        summ['performance'] = self.perfL
        return summ

    def plotStrategy(self,ax=None):
        """plot strategy distribution"""
        moveL = self.moveL
        if len(moveL) == 0: return ax
        stepL = [x[2] for x in moveL]
        if ax == None: fig, ax = plt.subplots(1,1)
        l, x = np.unique(stepL,return_counts=True)
        ax.pie(x,labels=l)
        return ax

    def plotAcceptance(self,ax=None):
        """plot acceptance graph"""
        moveL = [x[0] for x in self.moveL]
        if len(moveL) == 0: return ax
        l = 100.*np.arange(0,len(moveL),1)/moveL
        l = l[l == l]
        if ax == None: fig, ax = plt.subplots(1,1)
        ax.plot(l)
        return ax

    def plotState(self,ax=None):
        """plot graph and kpi plots"""
        route = self.spotL
        n = int(route.loc[route['agent']>0,'occupancy'].sum())
        if ax == None: fig, ax = plt.subplots(1,1)
        g_v.graphRoute(route,ax=ax)
        axins1 = ax.inset_axes([0.07, 0.75, 0.1, 0.2])
        self.plotHistory(axins1)
        axins2 = ax.inset_axes([0.07, 0.05, 0.1, 0.1])
        self.plotAcceptance(axins2)
        axins3 = ax.inset_axes([0.10, 0.15, 0.1, 0.1])
        self.plotStrategy(axins3)
        axins4 = ax.inset_axes([0.85, 0.05, 0.1, 0.1])
        self.plotOccupancy(axins4)
        l_g.insetStyle(axins1)
        l_g.insetStyle(axins2)
        l_g.insetStyle(axins3)
        l_g.insetStyle(axins4)
        xe, ye = ax.get_xlim(), ax.get_ylim()
        xd, yd = xe[1]-xe[0], ye[1]-ye[0]
        dt2 = (time.time() - self.start)/self.step
        nv = len(self.pathL) - 1
        string = "%.2f %% %.0f en %d agent %.2f s" % (self.completion,-self.En,nv,dt2)
        ax.text(xe[0]+.32*xd,ye[0]+.0*yd,string)
        #axins2.set_axis_off()
        # plt.show()
        return ax
    
