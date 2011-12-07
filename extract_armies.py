import sys, os, pickle, copy, itertools
try:
    import numpy as np
except:
    print "you need numpy"
    sys.exit(-1)

ut = {'T' : set(), 'P' : set(), 'Z' : set()}
armyut = {'T' : [], 'P' : ['Protoss Observer', 'Protoss Dragoon', 'Protoss Zealot', 'Protoss Archon', 'Protoss Reaver', 'Protoss High Templar', 'Protoss Arbiter', 'Protoss Carrier', 'Protoss Shuttle', 'Protoss Scout', 'Protoss Dark Archon', 'Protoss Corsair', 'Protoss Dark Templar'], 'Z' : ['Zerg Zergling', 'Zerg Devourer', 'Zerg Guardian', 'Zerg Ultralisk', 'Zerg Queen', 'Zerg Hydralisk', 'Zerg Mutalisk', 'Zerg Scourge', 'Zerg Lurker', 'Zerg Defiler']}
armies = {'5' : {'T' : [], 'P' : [], 'Z' : []}, '10' : {'T' : [], 'P' : [], 'Z' : []}, '15' : {'T' : [], 'P' : [], 'Z' : []}}

def features_scaling(tt):
    # find max
    mline = np.array(tt[0])
    for line in tt:
        for (i, e) in enumerate(line):
            if e > mline[i]:
                mline[i] = e
    # as we use discrete time steps, we can have null unit numbers!
    for (i, e) in enumerate(mline):
        if e <= 0.0:
            mline[i] = 1.0
    # divide each feature by its max
    return tt / mline

def extract_from(f):
    """ Will extract all the unit numbers from all players in replay 'f'
    and dump the unit numbers for each types at 5, 10 and 15 minutes in
    the armies datastruct (through the dump function)"""
    armies_players = [] # list of tuples ('player', {'unit type' : number})
    preproc = True # true while preprocessing the header of a rep
    finished = False # true when finished processing a rep
    players_list = False # true when listing the players in the rep's header
    min5 = False # true when in game time > 5 min
    min10 = False # true when in game time > 10 min
    min15 = False # true when in game time > 15 min

    def dump(l, t):
        #print '**** we are at the min:',
        #print t
        for ap in l:
            #wf = open(ap[0] + t + '.dat', 'a')
            tmplist = []
            for elem in armyut[ap[0][0]]:
                if ap[1].has_key(elem):
                    tmplist.append(float(ap[1][elem]))
                else:
                    tmplist.append(0.0)
            armies[t][ap[0][0]].append(tmplist)
            #print ap[1]
            #for (ut, nb) in ap[1].iteritems():
            #    print ut,
            #    print nb

    for line in f:
        if preproc:
            if players_list:
                l = line.split(',')
                if len(l) > 1:
                    armies_players.append((l[2].strip(' '), {}))
                elif 'Begin replay data' in line:
                    preproc = False
            if 'The following players are in this replay' in line:
                players_list = True
        elif not finished:
            if 'EndGame' in line:
                finished = True
                break
            l = line.split(',')
            p = int(l[1])
            if p < 0: # removes neutrals
                continue
            if 'Created' in l[2] or 'Morph' in l[2]:
                uname = l[4].strip(' ')
                ut[uname[0]].add(uname)
                armies_players[p][1][uname] = armies_players[p][1].get(uname, 0) + 1
            elif 'Destroyed' in l[2]:
                uname = l[4].strip(' ')
                armies_players[p][1][uname] = armies_players[p][1].get(uname, 0) - 1
            elif 'PlayerLeftGame' in l[2]:
                finished = True
            if not min5 and (7200 - int(l[0])) < 0:
                min5 = True
                dump(armies_players, '5')
            elif not min10 and (14400 - int(l[0])) < 0:
                min10 = True
                dump(armies_players, '10')
            elif not min15 and (21600 - int(l[0])) < 0:
                min15 = True
                dump(armies_players, '15')

def clusterize_dirichlet(*args):
    """ Clustering and plotting with Dirichlet process GMM """
    ### Clustering
    try:
        from sklearn import mixture
        from scipy import linalg
        import pylab as pl
        import matplotlib as mpl
        from sklearn.decomposition import PCA
    except:
        print "You need SciPy and scikit-learn"
        sys.exit(-1)

    models = []
    for arg in args:
        dpgmm = mixture.DPGMM(n_components = 15, cvtype='full')
        dpgmm.fit(arg)
        print dpgmm
        models.append(copy.deepcopy(dpgmm))
        print raw_input("any key to pass")

    ### Plotting
    color_iter = itertools.cycle (['r', 'g', 'b', 'c', 'm'])
    for i, (clf, data) in enumerate(zip(models, args)):
        pca = PCA(n_components=2)
        X_r = pca.fit(data).transform(data)
        splot = pl.subplot(len(args), 1, 1+i)
        pl.scatter(X_r[:,0], X_r[:,1])
        #pl.title('PCA of unit types / numbers')
        Y_ = clf.predict(data)
        for i, (mean, covar, color) in enumerate(zip(clf.means, clf.covars,
                                                     color_iter)):
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            pl.scatter(data[Y_== i, 0], data[Y_== i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1]/u[0])
            angle = 180 * angle / np.pi # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        pl.xlim(0.0, 1.0)
        pl.ylim(0.0, 1.0)
        pl.xticks(())
        pl.yticks(())
        pl.title("Dirichlet process GMM")
    pl.show()

def clusterize_r_em(*args):
    """ Clustering and plotting with EM GMM"""
    try:
        from rpy2.robjects import r
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
    except:
        print "You need rpy2"
        sys.exit(-1)

    r.library("mclust")
    for arg in args:
        model = r.Mclust(arg)
        print model
        print r.summary(model)
        r.quartz("plot")
        r.plot(model, arg)
        print raw_input("any key to pass")

#TEST print features_scaling(np.array([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]]))

f = sys.stdin
if len(sys.argv) > 1:
    if os.path.exists('fscaled1.blob') and os.path.exists('fscaled2.blob') and os.path.exists('fscaled3.blob'):
        fscaled1 = pickle.load(open('fscaled1.blob', 'r'))
        fscaled2 = pickle.load(open('fscaled2.blob', 'r'))
        fscaled3 = pickle.load(open('fscaled3.blob', 'r'))
        fscaled1.tofile('fscaled1.csv', ',')
        fscaled2.tofile('fscaled2.csv', ',')
        fscaled3.tofile('fscaled3.csv', ',')
        clusterize_dirichlet(fscaled1, fscaled2, fscaled3)
        clusterize_r_em(fscaled1, fscaled2, fscaled3)
    else:
        if sys.argv[1] == '-d':
            import glob
            for fname in glob.iglob(sys.argv[2] + '/*.rgd'):
                f = open(fname)
                extract_from(f)
        else:
            for arg in sys.argv[1:]:
                f = open(arg)
                extract_from(f)
        print ut
        armies_np5 = {'T' : np.array(armies['5']['T']), 'P' : np.array(armies['5']['P']), 'Z' : np.array(armies['5']['Z'])}
        armies_np10 = {'T' : np.array(armies['10']['T']), 'P' : np.array(armies['10']['P']), 'Z' : np.array(armies['10']['Z'])}
        armies_np15 = {'T' : np.array(armies['15']['T']), 'P' : np.array(armies['15']['P']), 'Z' : np.array(armies['15']['Z'])}
        fscaled1 = features_scaling(armies_np5['P'])
        pickle.dump(fscaled1, open('fscaled1.blob', 'w'))
        fscaled2 = features_scaling(armies_np10['P'])
        pickle.dump(fscaled2, open('fscaled2.blob', 'w'))
        fscaled3 = features_scaling(armies_np15['P'])
        pickle.dump(fscaled3, open('fscaled3.blob', 'w'))
        clusterize_dirichlet(fscaled1, fscaled2, fscaled3)
        clusterize_r_em(fscaled1, fscaled2, fscaled3)

