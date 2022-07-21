import numpy
import pandas

import importer
import pandas_options
import re
import pandas as pd
import os
from TextPreparer import TextPreparer
from Profiler import Profiler
from Estimator import Estimator
from DataRetriever import DataRetriever
from ClusterVisualisator import ClusterVisualisator
import math

profiler = Profiler()
profiler.start()

# logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\application-2022-07-10.log'
# startRowRegExp = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z'

logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\error.log'
startRowRegExp = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'

textFile = DataRetriever.readFile(logPath)
logRecordsList = re.split(startRowRegExp, textFile, flags=re.MULTILINE)
logRecordsSeries = pd.Series(logRecordsList)
print(logRecordsSeries)
countSamples = 10_000
texts = logRecordsSeries[:countSamples]
profiler.addPoint('text retrieving')

textPreparer = TextPreparer()
train = textPreparer.prepare(
    texts,
    strip=True,
    lower=True,
    clearPunctuation=True,
    clearDigits=False,
    stopWordsEnglish=False,
    stopWordsRussian=False,
    lemmatizationEnglish=True,
    stemmingEnglish=False,
    stemmingRussian=False,
    sinonymizeEnglish=False,
)

profiler.addPoint('text prepare')

features = TextPreparer.tfIdf(train)
print(features[1].shape)
profiler.addPoint('tfidf-1')

train = TextPreparer.clearRearWords(train=train, minCountRepeat=1, inAllDocument=True)
profiler.addPoint('clearRearWords')

features = TextPreparer.tfIdf(train)
tfIdf = features[0]
train = tfIdf
print(features[1].shape)
profiler.addPoint('tfidf-2')

[reducedTsne, reducedPca] = ClusterVisualisator.reduceDimensionality(train, isTsne=False, isPca=True)
ClusterVisualisator.visualizeMonocrome(reducedTsne, reducedPca)
profiler.addPoint('dimensionality reduction')

directDictionary = features[1]
# inverseDictionary = pandas.Series(features[1].index.values, index=features[1].values)
countSamples = train.shape[0]
distance = numpy.zeros((countSamples, countSamples))
maxDistance = 0
for i in range(countSamples):
    for j in range(countSamples):
        measureOfSimilarity = train[i] * train[j]
        measureOfSimilarity = measureOfSimilarity.sum()
        distance[i, j] = measureOfSimilarity
        if measureOfSimilarity > maxDistance:
            maxDistance = measureOfSimilarity

scaleFactor = math.pi / 2 / maxDistance

scaledDistance = numpy.zeros((countSamples, countSamples))
for i in range(countSamples):
    for j in range(countSamples):
        distanceValue = distance[i, j] * scaleFactor
        scaledDistance[i, j] = math.cos(distanceValue)

print(scaledDistance)

profiler.addPoint('calculate distance')

from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.1, min_samples=5, n_jobs=-1, metric='precomputed')
predictions = clustering.fit_predict(scaledDistance)
targetsUniq = list(set(predictions))
n_clusters = len(targetsUniq)
ClusterVisualisator.visualizeColor(n_clusters, reducedPca, predictions)

estimationByFeatures = Estimator.estimateByFeatures(predictions, train)
print('estimationByFeaturesKmeans')
print(estimationByFeatures)

# from sklearn.cluster import DBSCAN
# eps = 0.2
# min_samples = 5
# clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
# dbscanPredictions = clustering.fit_predict(train)
# targetsUniq = list(set(dbscanPredictions))
# n_clusters = len(targetsUniq)

from sklearn.cluster import KMeans
clusteringModel = KMeans(n_clusters=n_clusters, max_iter=500, random_state=21)
clusteringModel.fit(train)
kmeansPredictions = clusteringModel.predict(train)
ClusterVisualisator.visualizeColor(n_clusters, reducedPca, kmeansPredictions)

estimationByFeatures = Estimator.estimateByFeatures(kmeansPredictions, train)
print('estimationByFeaturesKmeans')
print(estimationByFeatures)

estimationByTargets = Estimator.estimateByTargets(predictions, kmeansPredictions)
print('estimationByTargets')
print(estimationByTargets[0])

profiler.addPoint('clustering')

profiler.print()
