"""
path_opt:
Path optimization library
"""
import geomadi as gem
import geomadi.geo_ops as gos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import shapely as sh
from sklearn.cluster import KMeans

def standardConf():
    conf = {"n_iter": 101
            ,"markov_chain":{"exp":1,"threshold":"row","leg":7,"doc":"threshold:[percentile,mean,len,row]"}
            ,"init":{"phantom": 4, "cluster": True, "extrude": False, "reset": True}
            ,"monte_carlo": {"cost_route": 70.35, "cost_stop": .1, "max_n": 50, "temperature": .5,"cost_area":1.,"cost_separation":10.}
            ,"moveProb": {"single": 1, "distance": 1, "markov": 4, "extrude": 3, "flat": 1, "outset": 3}
            ,"chainProb" : {"swap":1,"insert":0,"remove":0}
            ,"learn":{"net_layer": [8, 8], "load_model": "", "save_model": "q_table", "link": 5}
            ,"action": ['collect', 'potential']
            ,"kpi":{"completion":0., "density":0., "duration":0.}
            ,"simulation":{"step":0, "energy":0., "acceptance_move":0., "acceptance_rate":0., "cost":0., "revenue":0., "chemical_potential":0.}
            ,"version":"0.3.2 - grand canonical chains"
    }
    return conf

def updatePath(spotL, pathL):
    """update derived quantities """
    agentG = spotL.groupby("agent").sum()
    agentA = spotL.groupby("agent").agg(np.mean)
    agentL = spotL.groupby("agent").agg(len)
    idx = agentG.index
    idxn = [x for x in idx if not x in pathL.index]
    pathL1 = pathL.tail(len(idxn))
    pathL1.index = idxn
    pathL1['agent'] = idxn
    pathL = pd.concat([pathL, pathL1])
    pathL['load'] = 0
    pathL.loc[idx, 'distance'] = agentG['distance']
    pathL.loc[idx, 'load'] = agentL['occupancy']
    pathL.loc[idx, 'potential'] = agentG['potential']
    pathL.loc[idx, 'duration'] = agentG['duration']
    pathL.loc[idx, "x"] = agentA['x']
    pathL.loc[idx, "y"] = agentA['y']
    pathL['completion'] = pathL['load'] / pathL['capacity']
    pathL['completion'] = pathL['completion'].apply(lambda x: min(1, x))
    pathL.replace(float('inf'), 0., inplace=True)
    pathL.replace(float('nan'), 0., inplace=True)
    spotL['agent'] = spotL['agent'].astype(int)
    pathL['agent'] = pathL['agent'].astype(int)
    return pathL


class pathOpt:
    """optimization of paths"""

    def __init__(self, spotL, pathL=None, pairL=None, conf=None):
        """initialize quantities"""
        if spotL.shape[0] == 0:
            raise Exception("---------------no-spot------------------")
            return False
        self.logL = []
        self.colorL = ['black', 'blue', 'red', 'green', 'brown', 'orange', 'purple', 'magenta', 'olive', 'maroon',
                       'steelblue', 'midnightblue', 'darkslategrey', 'crimson', 'teal', 'darkolivegreen']
        self.colorL = ["#B4aaaaf0", "#8b122870", "#6CAF3070", "#F8B19570", "#F6728070", "#C06C8470", "#6C5B7B70",
                       "#355C7D70", "#99B89870", "#2A363B70", "#67E68E70", "#9F53B570", "#3E671470", "#7FA8A370",
                       "#6F849470", "#38577770", "#5C527A70", "#E8175D30", "#47474730", "#36363630", "#A7226E30",
                       "#EC204930", "#F26B3830", "#F7DB4F30", "#2F959930", "#E1F5C430", "#EDE57430", "#F9D42330",
                       "#FC913A30", "#FF4E5030", "#E5FCC230", "#9DE0AD30", "#45ADA830", "#54798030", "#594F4F30",
                       "#FE436530", "#FC9D9A30", "#F9CDAD30", "#C8C8A930", "#83AF9B30"]
        self.dens = 0.
        self.duration = 0.
        self.completion = 0.
        self.len_mean = 0.
        self.checkInit(spotL.copy(), pathL.copy())
        self.checkConf(conf)
        self.loadRouted(pairL)
        self.calcDistance()
        self.updatePath()
        self.insertPhantom(n=conf['init']['phantom'])
        self.cap = 5
        if pathL.shape[0] == 0:
            print("---------------no-path------------------")
            return False
        if conf['init']['reset']:
            self.spotL['agent'] = 0
        if conf['init']['cluster']:
            self.spotL = self.startPos(complete="all")
            self.updatePath()

    def checkConf(self, conf):
        """check configuration file"""
        conf1 = standardConf()
        if conf == None:
            conf = conf1
        for i in list(conf1):
            if not i in conf:
                conf[i] = conf1[i]
        self.phantom = conf['init']['phantom']
        self.opsL = conf['action']
        self.conf = conf

    def checkInit(self, spotL, pathL):
        """check columns in initial data frame"""
        print('-------------check-init---------------')
        if not isinstance(pathL, pd.DataFrame):
            pathL = spotL.groupby("agent").sum().reset_index()
            pathL.index = pathL['agent']
        setL = spotL['x'] == spotL['x']
        if sum(~setL) > 0:
            print("removing %d invalid coordinates" % (sum(~setL)) )
        spotL = spotL.loc[setL]
        setL = spotL['y'] == spotL['y']
        spotL = spotL.loc[setL]
        l1 = [x for x in
              ['duration', 'load', 'distance', 'agent', 'x', 'y', 'active', 'potential', 'occupancy', 'geohash'] if
              x not in spotL.columns]
        l2 = [x for x in
              ['agent', 'id', 'capacity', 'load', 'warehouse', 'x', 'y', 'x_c', 'y_c', 'x_start', 'y_start', 'x_end',
               'y_end', 'lenght', 'distance', 'score', 'phantom', 'energy', 't_start', 't_end', 'area'] if
              not x in pathL.columns]
        if len(l1) != 0: print("replacing columns", l1)
        if len(l2) != 0: print("replacing columns", l2)
        for l in l1: spotL[l] = 0
        for l in l2: pathL[l] = 0
        pathL.loc[:, "color"] = self.colorL[:pathL.shape[0]]
        pathL.loc[:, "phantom"] = False
        spotL.index = spotL['geohash']
        spotL.index.names = ['index']
        spotL.replace(float('nan'), 0., inplace=True)
        if len(set(spotL.index)) != spotL.shape[0]:
            print("duplicated geohash entries")
            tL = ['occupancy', 'priority', 'distance', 'potential', 'load']
            occL = spotL[['geohash'] + tL].groupby('geohash').agg(sum)
            spotL = spotL.groupby('geohash').first().reset_index()
            spotL.index = spotL['geohash']
            # spotL = spotL.loc[~spotL.index.duplicated(keep='first')]
            for t in tL:
                spotL[t] = occL[t]
        spotL.loc[:, 'agent'] = spotL['agent'].astype(int)
        pathL.loc[:, 'agent'] = pathL['agent'].astype(int)
        self.dens = gos.densArea(spotL['x'], spotL['y'])
        self.duration = np.mean(spotL['duration'])
        densL = gos.densPoint(spotL[['x', 'y']], radius=0.005)
        spotL.loc[:, 'density'] = densL
        self.spotL = spotL
        self.pathL = pathL

    def updateSys(self, route):
        """update system from accepted solution"""
        self.spotL = route
        self.updatePath()

    def updateConf(self):
        """update configuration"""
        self.conf['kpi']['completion'] = self.completion
        self.conf['kpi']['density'] = self.dens
        self.conf['kpi']['duration'] = self.duration
        self.conf['kpi']['len_mean'] = self.len_mean


    def updatePath(self):
        """update path frame from spot frame"""
        self.pathL = updatePath(self.spotL, self.pathL)
        pathv = self.pathL.loc[self.pathL['agent'] > 0]
        pathv = pathv.loc[~pathv['phantom']]
        self.completion = np.mean(pathv['completion'])
        self.updateConf()

    def addScore(self):
        """add score value"""
        self.pathL['score'] = 0.
        pathL = self.pathL[self.pathL['agent'] > 0]
        wp = pathL['potential'] / pathL['potential'].max()
        wl = 1.5 - pathL['distance'] / pathL['distance'].max()
        wc = pathL['load'] / pathL['capacity']
        wc = wc / wc.max()
        pathL.loc[:, 'score'] = wp * wl * wc
        self.pathL.loc[pathL.index, "score"] = pathL['score']

    def insertPhantom(self, n=0):
        """insert phantom agents"""
        if n == 0: return False
        path = self.pathL.tail(1).copy()
        pathS = None
        for i in range(n):
            pathS = pd.concat([pathS, path])
        nv = max(self.pathL['agent'])
        l = list(range(nv + 1, nv + n + 1))
        pathS.index = l
        pathS.loc[:, 'agent'] = l
        pathS['phantom'] = True
        self.pathL = pd.concat([self.pathL, pathS])
        self.updatePath()
        self.pathL.loc[:, "color"] = self.colorL[:self.pathL.shape[0]]
        self.phantom = self.phantom + n
        self.agentL = list(self.pathL[self.pathL['agent']>0].index)
        self.logL.append({"phantom":"inserting %d paths" % (n)} )
        return True

    def removePhantom(self, n=1, v=None):
        """remove phantom agent"""
        if self.phantom <= 0: return False
        pathv = self.pathL[self.pathL['agent'] > 0]
        pathv = pathv[pathv['phantom']]
        enL = pathv['energy']
        enL = enL.sort_values()
        if v: idx = [v]
        else: idx = enL[:n]
        for v in idx:
            self.pathL.drop(v, inplace=True)
            self.spotL.loc[self.spotL['agent'] == v, 'agent'] = 0
        self.phantom = len(pathv)
        self.agentL = list(self.pathL[self.pathL['agent']>0].index)
        return True

    def swapPath(self, vp, vn):
        """remove phantom agent"""
        if not vp in self.pathL.index:
            print("%d not in index" % (vp))
            return False
        if not vn in self.pathL.index:
            print("%d not in index" % (vn))
            return False
        nrgP = self.pathL.loc[vp, 'energy']
        nrgN = self.pathL.loc[vn, 'energy']
        setLP = self.spotL['agent'] == vp
        setLN = self.spotL['agent'] == vn
        self.spotL.loc[setLP, 'agent'] = vn
        self.spotL.loc[setLN, 'agent'] = vp
        self.updatePath()
        self.pathL.loc[vp, 'energy'] = nrgN
        self.pathL.loc[vn, 'energy'] = nrgP
        return True

    def erasePath(self, vp):
        """erase path"""
        for i in ['load', 'distance']:
            self.pathL.loc[vp, i] = 0
        self.spotL.loc[self.spotL['agent'] == vp, "agent"] = 0

    def restartPath(self, vp):
        """erase and restart a path"""
        if not hasattr(self, 'centroid'):
            self.spotL = self.startPos(complete="all")
            self.updatePath()
        self.erasePath()
        s = np.random.choice(self.centroid)
        self.spotL[s, "agent"] = vp

    def getPath(self, isPhantom=False):
        """return spot and path list without phantoms"""
        pathL = self.pathL.copy()
        spotL = self.spotL.copy()
        if not isPhantom:
            pathL = pathL.loc[~pathL['phantom']]
            agentL = list(pathL.index)
            setL = [not x in agentL for x in spotL['agent']]
            spotL.loc[setL, "agent"] = 0
        # self.addScore()
        # self.pathL = self.pathL.sort_values('score')
        return spotL, pathL, self.conf

    def getLog(self, key = None):
        """return the logs"""
        logL = self.logL
        if key != None:
            logL = [x[key] for x in logL if next(iter(x)) == key]
        return logL

    def checkAllowed(self, move):
        """check if the move is allowed"""
        vL, pL, sL = move['agent'], move['state'], move['action']
        if len(vL) == 0:
            self.logL.append({"move":"move [%s] empty" % (move['move']) })
            return False, move
        vL1, pL1, sL1 = [], [], []
        o1, o2, o3 = 0, 0, 0
        for v in vL:
            o1 += self.pathL.loc[v, 'load']
            o2 += self.pathL.loc[v, 'capacity']
        o1 = o1//len(vL)
        o2 = o2//len(vL)
        o3 = 0
        for i,p in enumerate(pL):
            o3 += self.spotL.loc[p, 'occupancy']
            if o1 + o3 > o2 + self.cap: break
            vL1.append(vL[i])
            pL1.append(pL[i])
            sL1.append(sL[i])
        move['agent'], move['state'], move['action'] = vL1, pL1, sL1
        if not vL1:
            self.logL.append({"move":'move %s not allowed %d tasks deleted' % (move['move'],len(vL))})
            return False, move
        return True, move
    

    def calcDistance(self):
        """lookup distances and update data frames"""
        for v, g in self.spotL.groupby("agent"):
            if v == 0: continue
            idx = list(g.index)
            for j1, j2 in zip(idx[:-1], idx[1:]):
                self.spotL.loc[j1, 'distance'] = self.distM.loc[j1, j2]
            self.spotL.loc[idx[-1], 'distance'] = self.distM.loc[idx[0], idx[-1]]
        agentG = self.spotL.groupby("agent").agg(sum).reset_index()
        agentG.index = agentG['agent']
        return agentG

    def simplifyRoute(self, route1, isPlot=False, isSingle=True):
        """sort route considering the shortest distance"""
        if sum(route1['agent'] > 0) == 0: route1
        distM = self.distM
        route = route1.copy()
        route.sort_values("agent", inplace=True)
        routeL = []
        route.loc[:, "distance"] = 0.
        for v, g in route.groupby("agent"):
            if v == 0:
                routeL.append(g)
                continue
            if not v in self.pathL['agent']:
                g1 = g.copy()
                g1['agent'] = 0
                routeL.append(g1)
                continue
            agent = self.pathL.loc[v]
            spot = g.copy()
            pos = spot[['x', 'y']]
            distM1 = distM.loc[spot.index, spot.index]
            tree = sp.spatial.KDTree(pos.values)
            if len(pos) > 1:
                line = sh.geometry.LineString(pos.values)
                area = line.convex_hull.area
                x_c, y_c = line.centroid.xy
                self.pathL.loc[v,'area'] = area
                self.pathL.loc[v,'x_c'] = x_c[0]
                self.pathL.loc[v,'y_c'] = y_c[0]
            nearest_dist, nearest_ind = tree.query(pos.values, k=2)
            ls, js = tree.query(agent[['x_start', 'y_start']])
            le, je = tree.query(agent[['x_end', 'y_end']])
            js = pos.iloc[js].name
            idx = sorted(list(g.index))
            distL = [0]
            id_sort = [js]
            for i in range(pos.shape[0] - 1):
                idx = [x for x in idx if x != js]
                j1 = distM1.loc[js, idx].idxmin()
                distL.append(distM1.loc[js, j1])
                js = j1
                id_sort.append(js)
            l2 = distM1.loc[id_sort[0], id_sort[-1]]
            # distL[0] = l2 # closed path
            # distL[0] +=  ls*.1 # distance to start
            # distL[-1] += le*.1 # distance to end
            spot.loc[id_sort, "distance"] = distL
            n_cap = g.shape[0] - self.pathL.loc[v, 'capacity']
            if n_cap > self.cap:  # remove exceeding
                n_cap = n_cap - self.cap
                self.logL.append({"simplify":'removing %d exceeding tasks from path %d' % (n_cap, v)})
                spot.iloc[-n_cap:, g.columns.get_loc('agent')] = 0
                # print(g.iloc[-n_cap:,g.columns.get_loc('agent')].tail(1))
            routeL.append(spot.loc[id_sort, :])
            route = pd.concat(routeL)
        if isPlot:
            gem.graph_viz.graphRoute(route)
            # plt.imshow(distM)
            plt.show()
        return route

    def startPos(self, spotL=None, isPlot=False, complete="none", agentL=None):
        """
        propose a start pos for each path and allocate the centroids
        complete = ["none","all","first"]
        """
        if spotL == None:
            spotL = self.spotL.copy()
        if agentL == None:
            agentL = self.agentL
        min_clust = max(3., np.mean(self.pathL['capacity']))
        n_clust = len(agentL)
        route = spotL[spotL['agent'] == 0].copy()
        colL = ['x','y'] # ,'potential']
        pos = route[colL]
        n_clust = int(route.shape[0] / min_clust)
        n_clust = int(route.shape[0] / min_clust) + np.random.randint(4) + 2
        kmeans = KMeans(n_clusters=n_clust).fit(pos)
        clusters = kmeans.predict(pos)
        clusters = [x + 1 for x in clusters]
        add = np.random.randint(n_clust)
        clusters = [(x + add) % n_clust for x in clusters]
        c1, c2 = np.unique(clusters, return_counts=True)
        cD = pd.DataFrame({"cluster": c1, "count": c2})
        cD['cluster2'] = cD['cluster']
        cD.loc[cD['count'] < 10, 'cluster2'] = 0
        cD1 = cD.groupby('cluster2').agg(sum).reset_index()
        m = len(cD1)
        n = max(0,m-len(agentL))
        agentI = agentL[:m] + [int(x) for x in np.zeros(n)]
        np.random.shuffle(agentI)
        cD1['cluster3'] = cD1.index
        cD1.loc[:,'cluster3'] = agentI
        cD = cD.merge(cD1, on="cluster2", how="left", suffixes=["", "_y"])
        cD.loc[~cD['cluster3'].isin(agentL), 'cluster3'] = 0
        route['cluster'] = clusters
        route = route.merge(cD[['cluster', 'cluster3']], on="cluster", how="left")
        route['agent'] = route['cluster3']
        route = route.drop(columns={"cluster","cluster3"})
        if complete == "all":
            for i,g in route.iterrows():
                spotL.loc[g['geohash'],'agent'] = g['agent']
        elif complete == "first":
            agentG = route.groupby('agent').agg(np.mean).reset_index()
            tree = sp.spatial.KDTree(pos.values)
            nearest_dist, nearest_ind = tree.query(pos.values, k=2)
            l1, cL = tree.query(agentG[colL])
            cL = route.index[cL]
            agentG['spot'] = cL
            self.centroid = cL
            for i, g in agentG.iterrows():
                spotL.loc[g['spot'], 'agent'] = i

        route2 = self.simplifyRoute(spotL) #TODO removes insert phantoms
        self.logL.append({"cluster":"clustering %d paths into %d areas, completion %.2f" % (self.pathL.shape[0], n_clust,self.completion)})
        return route2
        if isPlot:
            route['color'] = [self.colorL[x] for x in route['cluster3']]
            route['color'] = [colorL[x] for x in route['cluster3']]
            cent = kmeans.cluster_centers_
            plt.scatter(route['x'], route['y'], color=route['color'])
            plt.scatter(route.loc[cL, 'x'], route.loc[cL, 'y'], color='red')
            # plt.scatter(route.loc[cL1,'x'],route.loc[cL1,'y'],color='purple')
            plt.scatter(cent[:, 0], cent[:, 1], color='purple')
            plt.show()

    def outsetPath(self,v1,radius=0.01):
        """
        outset a path taking the closest non assigned spots
        """
        pos0 = self.spotL.loc[self.spotL['agent'] == 0 ,['x','y']].values
        idx = self.spotL.loc[self.spotL['agent'] == 0 ,['x','y']].index
        pos1 = self.spotL.loc[self.spotL['agent'] == v1,['x','y']].values
        if len(pos0)*len(pos1) == 0: return []
        tree0 = sp.spatial.KDTree(np.array(pos0))
        tree1 = sp.spatial.KDTree(np.array(pos1))
        r = radius*(1.+np.random.uniform())
        neighbors = tree1.query_ball_tree(tree0, r, p=2.0)
        neiL = list(set(sum(neighbors,[])))
        neiL = [idx[x] for x in neiL]
        return neiL

    def outsetAll(self,radius=0.01):
        """outset all paths"""
        pathv = self.pathL[self.pathL['agent'] > 0]
        for i,g in pathv.iterrows():
            neiL = self.outsetPath(i,radius=radius)
            self.spotL.loc[neiL,'agent'] = i
        self.spotL = self.simplifyRoute(self.spotL)
        self.updatePath()
    
    def loadRouted(self, pairL):
        """load a routed pair relationship between spots"""
        if not isinstance(pairL, pd.DataFrame):
            pos = self.spotL[['x', 'y']].sort_index()
            self.distM = pd.DataFrame(sp.spatial.distance_matrix(pos.values, pos.values), index=pos.index,
                                      columns=pos.index)
        else:
            odm = pairL.pivot_table(index="geohash_o", columns="geohash_d", values="length", aggfunc=np.sum)
            odw = pairL.pivot_table(index="geohash_o", columns="geohash_d", values="weight", aggfunc=np.sum)
            odm.replace(float('nan'), 10000., inplace=True)
            odw.replace(float('nan'), 0., inplace=True)
            self.distM = odm
            self.markovC = odw
        self.len_mean = np.median(self.distM)

    def plotOccupancy(self, ax=None):
        """plot occupanvy on agent level"""
        agentv = self.pathL[self.pathL['agent'] > 0]
        agentv.loc[:, 'width'] = .5
        if ax == None: fig, ax = plt.subplots(1, 1)
        colors = agentv['color']
        x = range(agentv.shape[0])
        y = agentv['load'] / agentv['capacity']
        ax.bar(x, y, width=.5, color=colors)
        return ax
