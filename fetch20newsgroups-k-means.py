import pandas
import numpy
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
# profiler.addPoint('text retrieving')

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

from sklearn.cluster import KMeans
clusteringModel = KMeans(n_clusters=textPrepareOptions.nClusters, max_iter=300, random_state=21)
clusteringModel.fit(train)
kmeansPredictions = clusteringModel.predict(train)
ClusterVisualisator.visualizeColor(textPrepareOptions.nClusters, reducedPca, kmeansPredictions)
profiler.addPoint('clustering KMeans')

estimationByFeatures = Estimator.estimateByFeatures(kmeansPredictions, train)
print('estimationByFeaturesKmeans')
print(estimationByFeatures)
profiler.addPoint('estimationByFeaturesKmeans duration')

predictionsFile = open('results/dbscanPredictions.json', 'r')
dbscanPredictions = json.load(predictionsFile)
predictionsFile.close()
dbscanPredictions = numpy.array(dbscanPredictions)
profiler.addPoint('reading dbscanPredictions')

estimationByTargets = Estimator.estimateByTargets(dbscanPredictions, kmeansPredictions)
print('estimationByTargets')
print(estimationByTargets[0])
profiler.addPoint('F1-estimation duration')

profiler.print()
