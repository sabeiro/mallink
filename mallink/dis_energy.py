"""
dis_energy:
utils for calculating distribution and energies
"""
import numpy as np
try:
    import albio.series_stat as s_s
    import albio.series_interp as s_i
except ImportError:
    print('albio not installed, some time series functionality may be impaired')


def sumProb(used,left,max_occ=20):
    """ordinal probability from a row of positive events: [4,3,5,6,...]"""
    prob = np.zeros(max_occ)
    for i,j in zip(used,left):
        ## summing used, left over, unknown
        p = [1. for x in range(i)] + [0. for x in range(j)] + [0 for x in range(max_occ-j-i)]
        #p = s_s.interpMissing(p)
        prob = prob + np.array(p[:max_occ])
    prob = prob/len(used)
    return prob

def chemPot(used,left,max_occ=20,thermal_noise=[0.75]):
    """compute the chemical potential from a Fermi-Dirac distribution"""
    if len(used) < 3: return float('nan')
    prob = sumProb(used,left)
    if sum(prob) >= max_occ - 0.01: return max_occ
    noise = list(thermal_noise)[0]
    t, y1, x0, err = s_i.fitFermi(prob,thermal_noise=noise)
    return x0[0], noise, err

def numericProb(chem_pot,thermal_noise=0.75,max_occ=20):
    """compute the probability numerically"""
    t = np.array(range(max_occ))
    return sum(s_i.dis_Fermi([chem_pot,thermal_noise],t,[0.75,thermal_noise]))
