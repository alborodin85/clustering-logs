import importer
import pandas_options
import numpy
from Profiler import Profiler
from DataRetriever import DataRetriever
from TextPreparer import TextPreparer
from ClusterVisualisator import ClusterVisualisator
from ResultHandler import ResultHandler
from Estimator import Estimator

profiler = Profiler()
profiler.start()

logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\application-2022-07-08.log'
startRowRegExp = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z)'

# logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\error.log'
# startRowRegExp = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'

textFile = DataRetriever.readFile(logPath)
countRows = 100_000
texts = DataRetriever.splitText(startRowRegExp, textFile, countRows)

profiler.addPoint('text retrieving')

textPreparer = TextPreparer()
train = textPreparer.prepare(
    texts,
    strip=True,
    lower=True,
    clearPunctuation=True,
    clearDigits=True,
    stopWordsEnglish=True,
    stopWordsRussian=False,
    lemmatizationEnglish=True,
    stemmingEnglish=False,
    stemmingRussian=False,
    sinonymizeEnglish=False,
)
profiler.addPoint('text prepare')

features = TextPreparer.tfIdf(train)
print('до очистки редких слов: ', features[1].shape)
profiler.addPoint('tfidf-1')

train = TextPreparer.clearRearWords(train=train, minCountRepeat=3, inAllDocument=True)
print('train.shape', train.shape)
profiler.addPoint('clearRearWords')

features = TextPreparer.tfIdf(train)
tfIdf = features[0]
train = tfIdf
print('после очистки редких слов: ', features[1].shape)
profiler.addPoint('tfidf-2')

[reducedTsne, reducedPca] = ClusterVisualisator.reduceDimensionality(train, isTsne=False, isPca=True)
ClusterVisualisator.visualizeMonocrome(reducedTsne, reducedPca)
profiler.addPoint('dimensionality reduction')

from sklearn.cluster import Birch
brc = Birch(threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=True)

itemsInButch = 1000
# countSamples = train.shape[0]
# countButches = countSamples // itemsInButch
# itemsInLastButch = countSamples % itemsInButch
# if itemsInLastButch:
#     countButches += 1
#
# batches = []
# sampleNum = 0
# for i in range(countButches):
#     batches.append([])
#     for j in range(itemsInButch):
#         if sampleNum < train.shape[0]:
#             batches[i].append(train[sampleNum])

# print(batches)

# brc.fit(train)

batches = textPreparer.splitTrainData(itemsInButch, train)

for i in range(len(batches)):
    brc.partial_fit(batches[i])

birchPredictions = brc.predict(train)
print(birchPredictions)

targetsUniq = list(set(birchPredictions))
n_clusters = len(targetsUniq)

birchPredictions = TextPreparer.vectorizeLabels(birchPredictions)
ClusterVisualisator.visualizeColor(n_clusters, reducedPca, birchPredictions)

profiler.addPoint('clustering Birch')

[clustersItems, clustersItemsCount, clustersWords] = ResultHandler.parsePridictions(features[1], birchPredictions, train)

print('clustersItemsCount')
print(clustersItemsCount)
print('clustersWords')
print(clustersWords)

# testCluster = clustersItems[2]
# for sampleId in testCluster:
#     print(texts[sampleId])

profiler.addPoint('build words Birch')

from sklearn.cluster import KMeans
clusteringModel = KMeans(n_clusters=n_clusters, max_iter=300, random_state=21)
clusteringModel.fit(train)
kmeansPredictions = clusteringModel.predict(train)

ClusterVisualisator.visualizeColor(n_clusters, reducedPca, kmeansPredictions)

profiler.addPoint('clustering KMeans')

# estimationByFeatures = Estimator.estimateByFeatures(birchPredictions, train)
# print('estimationByFeaturesBirch')
# print(estimationByFeatures)
#
# estimationByFeatures = Estimator.estimateByFeatures(kmeansPredictions, train)
# print('estimationByFeaturesKmeans')
# print(estimationByFeatures)

estimationByTargets = Estimator.estimateByTargets(birchPredictions, kmeansPredictions)
print('estimationByTargets')
print(estimationByTargets[0])

profiler.addPoint('estimation')

print('timeProfiler')
print(profiler.points)
