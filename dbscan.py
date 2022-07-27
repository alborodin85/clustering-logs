import pandas_options
from TextPreparer import TextPreparer
from Profiler import Profiler
from Estimator import Estimator
from DataRetriever import DataRetriever
from ClusterVisualisator import ClusterVisualisator
from ResultHandler import ResultHandler
from TextPrepareOptions import TextPrepareOptions

profiler = Profiler()
profiler.start()

# logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\application-2022-07-08.log'
# startRowRegExp = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z)'

logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\error.log'
startRowRegExp = r'(\d{4}-\d{2}-\d{2} {1,2}\d{1,2}:\d{2}:\d{2})'

# logPath = r'C:\borodin_admin\Институт\_ВКР\2022-06-14 Приложение\clustering-logs\syslog'
# startRowRegExp = r'\w{3} \d{2} \d{2}:\d{2}:\d{2}'

textFile = DataRetriever.readFile(logPath)
countSamples = 10_000
texts = DataRetriever.splitText(startRowRegExp, textFile, countSamples)
profiler.addPoint('text retrieving')
textPrepareOptions = TextPrepareOptions()
textPreparer = TextPreparer()

train = textPreparer.prepare(texts, textPrepareOptions)
profiler.addPoint('text prepare', True)

features = TextPreparer.tfIdf(train)
print('до очистки редких слов: ', features[1].shape)
profiler.addPoint('tfidf-1')

train = TextPreparer.clearRearWords(train=train, minCountRepeat=0, inAllDocument=False)
print('train.shape', train.shape)
profiler.addPoint('clearRearWords', True)

features = TextPreparer.tfIdf(train)
tfIdf = features[0]
train = tfIdf
print('после очистки редких слов: ', features[1].shape)
profiler.addPoint('tfidf-2', True)

[reducedTsne, reducedPca] = ClusterVisualisator.reduceDimensionality(train, isTsne=False, isPca=True)
ClusterVisualisator.visualizeMonocrome(reducedTsne, reducedPca)
profiler.addPoint('dimensionality reduction')

from sklearn.cluster import DBSCAN
eps = 0.2
min_samples = 5
clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
dbscanPredictions = clustering.fit_predict(train)
targetsUniq = list(set(dbscanPredictions))
n_clusters = len(targetsUniq)

dbscanPredictions = TextPreparer.vectorizeLabels(dbscanPredictions)
ClusterVisualisator.visualizeColor(n_clusters, reducedPca, dbscanPredictions)

profiler.addPoint('clustering DBSCAN', True)

[clustersItems, clustersItemsCount, clustersWords] = ResultHandler.parsePridictions(features[1], dbscanPredictions, train)

print('clustersItemsCount')
print(clustersItemsCount)
print('clustersWords')
print(clustersWords)

testCluster = clustersItems[0]
f = open('samples.txt', 'w')
for sampleId in testCluster:
    f.write(sampleId)
f.close()

profiler.addPoint('build words DBSCAN')
# n_clusters = 6

from sklearn.cluster import KMeans
clusteringModel = KMeans(n_clusters=n_clusters, max_iter=300, random_state=21)
clusteringModel.fit(train)
kmeansPredictions = clusteringModel.predict(train)

ClusterVisualisator.visualizeColor(n_clusters, reducedPca, kmeansPredictions)

profiler.addPoint('clustering KMeans')

estimationByFeatures = Estimator.estimateByFeatures(dbscanPredictions, train)
print('estimationByFeaturesDbscan')
print(estimationByFeatures)

estimationByFeatures = Estimator.estimateByFeatures(kmeansPredictions, train)
print('estimationByFeaturesKmeans')
print(estimationByFeatures)

estimationByTargets = Estimator.estimateByTargets(dbscanPredictions, kmeansPredictions)
print('estimationByTargets')
print(estimationByTargets[0])

profiler.addPoint('estimation')

print('timeProfiler')
print(profiler.points)
