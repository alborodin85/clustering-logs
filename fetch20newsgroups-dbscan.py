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

# dataFrame = pandas.read_csv('hierarchical_text_classification/train_40k.csv')
# texts = dataFrame['Title'] + ' ' + dataFrame['Text']
# texts = texts[:10000]

countRows = 0
for text in texts:
    subArr = str(text).split('\n')
    countRows += len(subArr)

print('countRows', countRows)
profiler.addPoint('text retrieving')

exit()

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

print('n_clusters', n_clusters)

testCluster = clustersItems[0]
f = open('samples.txt', 'w')
for sampleId in testCluster:
    f.write(texts[sampleId])
f.close()

profiler.addPoint('build words DBSCAN', True)

estimationByFeatures = Estimator.estimateByFeatures(dbscanPredictions, train)
print('estimationByFeaturesDbscan')
print(estimationByFeatures)
profiler.addPoint('estimationByFeaturesDbscan duration')

predictionsFile = open('results/dbscanPredictions.json', 'w')
json.dump(dbscanPredictions.tolist(), predictionsFile, indent=4)
predictionsFile.close()

profiler.addPoint('serialize result')

profiler.print()
