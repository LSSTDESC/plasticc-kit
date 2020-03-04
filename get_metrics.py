import matplotlib
#gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']

#for gui in gui_env:
#    try:
 #       print("testing", gui)
matplotlib.use('TKAgg',warn=False, force=True)
from matplotlib import pyplot as plt
#break
#except:
#    continue
#print("Using:",matplotlib.get_backend())

import sys
import os
from collections import Counter, OrderedDict
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from astropy.table import Table
import schwimmbad
from cesium.time_series import TimeSeries
import cesium.featurize as featurize
from tqdm import tnrange, tqdm_notebook
import sklearn 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


def plasticc_log_loss(y_true, y_pred, relative_class_weights=None):
        """
        Implementation of weighted log loss used for the Kaggle challenge
        """
        predictions = y_pred.copy()

        # sanitize predictions
        epsilon = sys.float_info.epsilon # this is machine dependent but essentially prevents log(0)
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]

        predictions = np.log(predictions)
        # multiplying the arrays is equivalent to a truth mask as y_true only contains zeros and ones
        class_logloss = []
        for i in range(predictions.shape[1]):
            # average column wise log loss with truth mask applied
            result = np.average(predictions[:, i][y_true[:, i] == 1])
            class_logloss.append(result)
        return -1 * np.average(class_logloss, weights=relative_class_weights)
    


fig, ax = plt.subplots()
pbmap = OrderedDict([(0,'u'), (1,'g'), (2,'r'), (3,'i'), (4, 'z'), (5, 'y')])

# it also helps to have passbands associated with a color
pbcols = OrderedDict([(0,'blueviolet'), (1,'green'), (2,'red'),\
                      (3,'orange'), (4, 'black'), (5, 'brown')])

pbnames = list(pbmap.values())
datadir = '/Users/reneehlozek/Data/plasticc/'
metafilename = datadir+'training_set_metadata.csv'
metadata = Table.read(metafilename, format='csv')
nobjects = len(metadata)
counts = Counter(metadata['target'])
labels, values = zip(*sorted(counts.items(), key=itemgetter(1)))
nlines = len(labels)


featurefile = datadir+'plasticc_featuretable.npz'
if os.path.exists(featurefile):
    featuretable, _ = featurize.load_featureset(featurefile)
else:
    features_list = []
    with tqdm_notebook(total=nobjects, desc="Computing Features") as pbar:
        with schwimmbad.MultiPool() as pool:  
            results = pool.imap(worker, list(tsdict.values()))
            for res in results:
                features_list.append(res)
                pbar.update()
            
    featuretable = featurize.assemble_featureset(features_list=features_list,\
                              time_series=tsdict.values())
    featurize.save_featureset(fset=featuretable, path=featurefile)



old_names = featuretable.columns.values
new_names = ['{}_{}'.format(x, pbmap.get(y,'meta')) for x,y in old_names]
cols = [featuretable[col] for col in old_names]
allfeats = Table(cols, names=new_names)
del featuretable


splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
splits = list(splitter.split(allfeats, metadata['target']))[0]
train_ind, test_ind = splits


corr = allfeats.to_pandas().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                square=True, linewidths=.2, cbar_kws={"shrink": .5})

Xtrain = np.array(allfeats[train_ind].as_array().tolist())
Ytrain = np.array(metadata['target'][train_ind].tolist())

Xtest  = np.array(allfeats[test_ind].as_array().tolist())
Ytest  = np.array(metadata['target'][test_ind].tolist())


ncols = len(new_names)
npca  = (ncols  - 3)//len(pbnames)  + 3

pca = PCA(n_components=npca, whiten=True, svd_solver="full", random_state=42)
Xtrain_pca = pca.fit_transform(Xtrain)
Xtest_pca = pca.transform(Xtest)

fig, ax = plt.subplots()
ax.plot(np.arange(npca), pca.explained_variance_ratio_, color='C0')
ax2 = ax.twinx()
ax2.plot(np.arange(npca), np.cumsum(pca.explained_variance_ratio_), color='C1')
ax.set_yscale('log')
ax.set_xlabel('PCA Component')
ax.set_ylabel('Explained Variance Ratio')
ax2.set_ylabel('Cumulative Explained Ratio')
fig.tight_layout()

plt.savefig('corr.png')
plt.clf()

original=False
# Original notebook from Gautham
if original:
    clf = RandomForestClassifier(n_estimators=200, criterion='gini',\
                                     oob_score=True, n_jobs=-1, random_state=42,\
                                     verbose=1, class_weight='balanced', max_features='sqrt')
    
    clf.fit(Xtrain_pca, Ytrain)
    Ypred = clf.predict(Xtest_pca)

    print(Ypred[0:5], 'yoyo')
    print(type(Ypred), type(Ytest), 'types')
    print(np.shape(Ypred), np.shape(Ytest), 'shapes')
    cm = confusion_matrix(Ytest, Ypred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.around(cm, 2)

# we didn't release the weights with the contest at Kaggle's request
# they have been probed through the leaderboard though
# we leave the reader to adjust this as they see fit
# the official metric is always what is on Kaggle's leaderboard. 
# This notebook is solely for demonstration.

    weights = np.ones(nlines)
# we want the actual prediction probabilities
    Ypredprob = clf.predict_proba(Xtest_pca)
    print(Ypred[0:5])
    print(Ytest, 'truth labels', np.shape(Ytest))
    print(np.shape(Ypred), np.shape(Ytest), 'shapes')
# we also need to express the truth table as a matrix
    sklearn_truth = np.zeros((len(Ytest), nlines))
    label_index_map = dict(zip(clf.classes_, np.arange(nlines)))

    print(labels, 'labels')
    print(label_index_map, 'mapping')

    for i, x in enumerate(Ytest):
        sklearn_truth[i][label_index_map[Ytest[i]]] = 1

    print(sklearn_truth, 'truth sklearn')

    logloss = plasticc_log_loss(sklearn_truth, Ypredprob, relative_class_weights=weights)
    print(logloss, 'logloss')
    fig, ax = plt.subplots(figsize=(9,7))

    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap='Blues', annot=annot, lw=0.5)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    plt.savefig('testcm.png')

      
# Now we are working with challenge entries
else:

	name = '1_Kyle' #'3_MajorTom' #2_MikeSilogram' #'1_Kyle'
	print(name)
	Rprob = pd.read_csv('/Users/reneehlozek/Dropbox/First_10_submissions/'+name+'.csv') #, nrows=20)
	Rtruth = pd.read_csv('/Users/reneehlozek/Dropbox/First_10_submissions/'+name+'_truth.csv') #, nrows=20)
	weightinds=[51,60,4,91,99]
	weightvals = 2
	obj_truth = np.shape(Rtruth)[0]
	nlines=obj_truth
	nclass_truth = np.shape(Rtruth)[1]-1
	Rcols=Rprob.columns.tolist()
	print('hey renee')
	print(Rcols)

	# Removing the object ID from the list
	if name == '3_MajorTom':
		Rcols.pop(-1)
	if name == '1_Kyle':
		Rcols.pop(0)
	if name == '2_MikeSilogram':
		Rcols.pop(-2)

	labels = [int(j.split('_')[1]) for j in Rcols[:]]
	truvalmat = np.array(Rtruth[Rcols[:]])
	Rprobb = np.array(Rprob[Rcols[:]])
		
	print(labels, 'labels')
    # Pull off the truth matrix where each column is either a one or zero

# Making a vector of labels of the true values

	indtru = [None] * np.shape(truvalmat)[0]
	truvals = [None] * np.shape(truvalmat)[0]
	
	for j in range(np.shape(truvalmat)[0]):
#		print(np.where(truvalmat[j]), 'where')
		indtru[j] = np.where(truvalmat[j])[0][0]
		try:
			truvals[j] = labels[indtru[j]]
		except:
			print('eep')
			print(indtru[j],  truvalmat[j], len(labels))

    # Making a vector of predicted labels from the probabilities
	predvals = [labels[np.argmax(Rprobb[j,:])] for j in range(np.shape(truvalmat)[0])]
	indpred = [np.argmax(Rprobb[j,:]) for j in range(np.shape(truvalmat)[0])]

	rcm = confusion_matrix(truvals, predvals, labels=labels)
	rcm = rcm.astype('float') / rcm.sum(axis=1)[:, np.newaxis]
	rannot = np.around(rcm, 2)
	
	fig, ax = plt.subplots(figsize=(9,7))
	sns.heatmap(rcm, xticklabels=labels, yticklabels=labels, cmap='Blues', annot=rannot, lw=0.5)
	ax.set_xlabel('Predicted Label')
	ax.set_ylabel('True Label')
	ax.set_aspect('equal')
	plt.savefig('testrcm_'+name+'.png')
    
	nclass = np.shape(truvalmat[:,:])
	weights = np.ones(np.shape(truvalmat[:,:])[1])
	inds=[0,4,6,13,14]
	weights[inds]=2


	print(nclass, 'nclass')
	print(weights, 'weights')

    # The log loss function takes in matrices of truth and probability
	logloss = plasticc_log_loss(truvalmat[:,:], Rprobb[:,:], relative_class_weights=weights)
	print(logloss, name)




