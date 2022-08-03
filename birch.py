import pandas_options
from Profiler import Profiler
from DataRetriever import DataRetriever
from TextPreparer import TextPreparer
from ClusterVisualisator import ClusterVisualisator
from ResultHandler import ResultHandler
from Estimator import Estimator
from TextPrepareOptions import TextPrepareOptions

profiler = Profiler()
profiler.start()
textPrepareOptions = TextPrepareOptions()

textFile = DataRetriever.readFile(textPrepareOptions.logPath)
texts = DataRetriever.splitText(textPrepareOptions.startRowRegExp, textFile, textPrepareOptions.countRows)
profiler.addPoint('text retrieving')

textPreparer = TextPreparer()
texts = textPreparer.sliceMessages(texts, 0)
train = textPreparer.prepare(texts, textPrepareOptions)
profiler.addPoint('text prepare', True)

train = TextPreparer.clearRearWords(train=train, minCountRepeat=textPrepareOptions.minCountRepeat, inAllDocument=textPrepareOptions.inAllDocument)
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
print(birchPredictions)

targetsUniq = list(set(birchPredictions))
n_clusters = len(targetsUniq)

birchPredictions = TextPreparer.vectorizeLabels(birchPredictions)
ClusterVisualisator.visualizeColor(n_clusters, reducedPca, birchPredictions)

profiler.addPoint('clustering Birch', True)

[clustersItems, clustersItemsCount, clustersWords] = ResultHandler.parsePridictions(features[1], birchPredictions, train)

print('n_clusters', n_clusters)

testCluster = clustersItems[0]
f = open('samples.txt', 'w')
for sampleId in testCluster:
    f.write(texts[sampleId])
f.close()

profiler.addPoint('build words Birch', True)

estimationByFeatures = Estimator.estimateByFeatures(birchPredictions, train)
print('estimationByFeaturesBirch')
print(estimationByFeatures)
profiler.addPoint('estimationByFeaturesBirch duration')
profiler.print()

from sklearn.cluster import KMeans
clusteringModel = KMeans(n_clusters=n_clusters, max_iter=300, random_state=21)
clusteringModel.fit(train)
kmeansPredictions = clusteringModel.predict(train)

ClusterVisualisator.visualizeColor(n_clusters, reducedPca, kmeansPredictions)

profiler.addPoint('clustering KMeans')

estimationByFeatures = Estimator.estimateByFeatures(kmeansPredictions, train)
print('estimationByFeaturesKmeans')
print(estimationByFeatures)
profiler.addPoint('estimationByFeaturesKmeans duration')

estimationByTargets = Estimator.estimateByTargets(birchPredictions, kmeansPredictions)
print('estimationByTargets')
print(estimationByTargets[0])
profiler.addPoint('F1-estimation duration')

profiler.print()
