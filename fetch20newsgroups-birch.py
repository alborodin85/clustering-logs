import pandas
from sklearn import datasets
from Profiler import Profiler
from TextPrepareOptions import TextPrepareOptions
from TextPreparer import TextPreparer
from ClusterVisualisator import ClusterVisualisator
from ResultHandler import ResultHandler
from Estimator import Estimator
import json

profiler = Profiler()
profiler.start()
textPrepareOptions = TextPrepareOptions()

dataHome = r'fetch_20newsgroups'
subset = 'all'
randomState = 1
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

fetch20Newsgroups = datasets.fetch_20newsgroups(
    data_home=dataHome,
    subset=subset,
    random_state=randomState,
    categories=categories,
    return_X_y=True,
)

texts = pandas.Series(fetch20Newsgroups[0])
profiler.addPoint('text retrieving')

textPreparer = TextPreparer()
texts = textPreparer.sliceMessages(texts, 0)
train = textPreparer.prepare(texts, textPrepareOptions)
profiler.addPoint('text prepare', True)

train = TextPreparer.clearRearWords(
    train=train,
    minCountRepeat=textPrepareOptions.minCountRepeat,
    inAllDocument=textPrepareOptions.inAllDocument
)
print('train.shape', train.shape)
profiler.addPoint('clearRearWords', True)

features = TextPreparer.tfIdf(train)
tfIdf = features[0]
train = tfIdf
profiler.addPoint('tfidf-2', True)
print('Вектор признаков: ', features[1].shape)

[reducedTsne, reducedPca] = ClusterVisualisator.reduceDimensionality(train, isTsne=False, isPca=True)
ClusterVisualisator.visualizeMonocrome(reducedTsne, reducedPca)
profiler.addPoint('dimensionality reduction')

from sklearn.cluster import Birch
brc = Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True)
batches = textPreparer.splitTrainData(textPrepareOptions.birchItemsInButch, train)
for i in range(len(batches)):
    brc.partial_fit(batches[i])

birchPredictions = brc.predict(train)


targetsUniq = list(set(birchPredictions))
n_clusters = len(targetsUniq)

birchPredictions = TextPreparer.vectorizeLabels(birchPredictions)
ClusterVisualisator.visualizeColor(n_clusters, reducedPca, birchPredictions)

profiler.addPoint('clustering BIRCH', True)

[clustersItems, clustersItemsCount, clustersWords] = ResultHandler.parsePridictions(features[1], birchPredictions, train)

print('n_clusters', n_clusters)

testCluster = clustersItems[0]
f = open('samples.txt', 'w')
for sampleId in testCluster:
    f.write(texts[sampleId])
f.close()

profiler.addPoint('build words BIRCH', True)

estimationByFeatures = Estimator.estimateByFeatures(birchPredictions, train)
print('estimationByFeaturesBirch')
print(estimationByFeatures)
profiler.addPoint('estimationByFeaturesBirch duration')

predictionsFile = open('results/dbscanPredictions.json', 'w')
json.dump(birchPredictions.tolist(), predictionsFile, indent=4)
predictionsFile.close()

profiler.addPoint('serialize result')

profiler.print()
