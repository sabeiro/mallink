"""
monte_markov:
Markov chain Monte Carlo library for paths optimization
"""
import numpy as np

from mallink.markov_chain import MarkovChain
from mallink.monte_carlo import MonteCarlo


class MonteMarkov(MonteCarlo, MarkovChain):
    """Monte Carlo for path optimization"""

    def __init__(self, spotL, pathL=None, pairL=None, conf=None, confM=None):
        """initialize quantities and Markov chains"""
        MonteCarlo.__init__(self, spotL, pathL, pairL, conf)
        MarkovChain.__init__(self, self.spotL, pairL, confM)
        if conf['init']['extrude']: self.initChains()

    def initChains(self):
        """stretch chains"""
        agentv = self.pathL[self.pathL['agent'] > 0]
        route1 = self.spotL
        for v, g in agentv.iterrows():
            move = self.tryExtrude(v1=g['agent'], fill="full")
            route1 = self.move(move)
            En1 = self.checkEnergy(route1, move)
            self.updateSys(route1)
            self.En = En1
            self.moveN += 1
            self.moveL.append([self.step, self.start, move['move'], self.En, self.cost, self.revenue, self.completion])
        return route1, move

    def tryInsertChain(self):
        """insert new chains"""
        pathv = self.spotL[self.spotL['agent'] > 0]
        agentv = self.pathL[self.pathL['agent'] > 0]
        v = max(pathv.index) + 1
        move = self.tryExtrude(v1=v, fill="full")
        route1 = self.move(move)
        En1 = self.checkEnergy(route1, move)
        dEn = (self.En - En1) - self.chemPot
        if not self.isMetropolis(dEn, weight=move['weight']): return False
        self.updateSys(route1)
        self.En = En1
        self.moveN += 1
        self.moveL.append([self.step, self.start, move['move'], self.En, self.cost, self.revenue, self.completion])
        self.insertPhantom(n=1)
        return route1, move

    def tryChange(self, m=None):
        """perform a move and try to accept it"""
        if m == None:
            m = np.random.choice(self.moveS, p=self.moveP)
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
        return True, move

    def tryMarkov(self):
        """continue the path"""
        pv = self.probPath()
        v1 = np.random.choice(pv.index, p=pv)
        routev = self.spotL[self.spotL['agent'] == v1]
        if routev.shape[0] == 0:
            p1 = np.random.choice(self.spotL.index)
        elif routev.shape[0] == 1:
            p1 = routev.index[0]
        else:
            pp = self.probNorm(routev['distance'] / routev['potential'])
            p1 = np.random.choice(routev.index, p=pp)
        st = self.probNorm(self.markovC.loc[:, p1])
        p2 = np.random.choice(st.index, p=st)
        v2 = self.spotL.loc[p1, 'agent']
        s1 = np.random.choice(self.opsL)
        w = 1  # - st[p2]
        return {"weight": w, "agent": [v1], "state": [p2], "action": [s1], "move": "markov"}

    def tryMarkovFlat(self):
        """continue the path"""
        v1 = np.random.choice(self.pathL.index)
        routev = self.spotL[self.spotL['agent'] == v1]
        if routev.shape[0] == 0:
            p1 = np.random.choice(self.spotL.index)
        else:
            p1 = np.random.choice(routev.index)
        st = self.probNorm(self.markovC.loc[:, p1])
        for i in range(100):
            p2 = np.random.choice(st.index, p=st)
            if not p2 in routev.index: break
        v2 = self.spotL.loc[p1, 'agent']
        s1 = np.random.choice(self.opsL)
        w = 1  # - st[p2]
        return {"weight": w, "agent": [v1], "state": [p2], "action": [s1], "move": "flat"}

    def tryExtrude(self, v1=None, fill="random"):
        """continue the path"""
        pv = self.probPath()
        if v1 == None:
            v1 = np.random.choice(pv.index, p=pv)
        if v1 not in self.pathL.index:
            return {"weight": 1., "agent": [], "state": [], "action": [], "move": "error: path not in list"}
        routev = self.spotL[self.spotL['agent'] == v1]
        if routev.shape[0] == 0:
            ps = self.probNorm(self.spotL['potential'])
            p1 = np.random.choice(ps.index, p=ps)
            routev = self.spotL.loc[p1:p1]
        agent = self.pathL[self.pathL['agent'] == v1]
        n = int((agent['capacity'] - agent['load']).values[0])
        if fill == "random":
            n = np.random.randint(0, max(n, 1), size=None)
        if n <= 0:
            return {"weight": 1., "agent": [], "state": [], "action": [], "move": "chain full"}
        p1 = p2 = routev.index[0]
        s1 = np.random.choice(self.opsL)
        pL, vL, sL = [p1], [v1], [s1]
        weight = 0.
        for i in range(n):
            st = self.probNorm(self.markovC.loc[:, p1])
            for j in range(100):
                p2 = np.random.choice(st.index, p=st)
                if not p2 in pL: break
            if p2 in pL: continue
            pL.append(p2)
            vL.append(v1)
            sL.append(s1)
            weight += 1. - st[p2]
            p1 = p2
        self.logL.append({"extrusion":"extrusion completion %.2f/%d" % (len(pL) / n, v1)})
        weight = 1.
        return {"weight": weight, "agent": vL, "state": pL, "action": sL, "move": "extrude"}


if False:
    def uniform_prior_distribution(p, size, **args):
        prior = 1.0 / size
        return prior


    def mcmc(prior_dist, size=100000, burn=1000, thin=10, Z=3, N=10):
        import random
        from scipy.stats import binom
        mc = [0]  # Initialize markov chain
        while len(mc) < thin * size + burn:
            cand = random.gauss(mc[-1], 1)  # Propose candidate
            args1 = {"p": cand, "q": 0.2, "size": size}
            args2 = {"p": mc[-1], "q": 0.2, "size": size}
            ratio = (binom.pmf(Z, N, cand) * prior_dist(**args1)) / (binom.pmf(Z, N, mc[-1]) * prior_dist(**args2))
        if ratio > random.random():  # Acceptence criteria
            mc.append(cand)
        else:
            mc.append(mc[-1])
        # Take sample
        sample = []
        for i in range(len(mc)):
            if i >= burn and (i - burn) % thin == 0:
                sample.append(mc[i])
            sample = sorted(sample)
            # Estimate posterior probability
        post = []
        for p in sample:
            post.append(binom.pmf(Z, N, p) * prior_dist(p, size))
            return sample, post, mc
