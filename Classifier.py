### 4 types of Classifier chain used for topic classfication and calculate Jaccard score and ensemble CC

#LinearSVC and ClassifierChain
from sklearn.svm import LinearSVC
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
# Fit an ensemble of Linear SVM classifier chains and take the average prediction of all the chains
chains = [ClassifierChain(LinearSVC(), order='random', random_state=i)
          for i in range(20)]
for chain in chains:
  chain.fit(vectorizedTrainData,trainBinaryLabel)

# Make predictions on train data
pred_train_chains = np.array([chain.predict(vectorizedTrainData)
                             for chain in chains])

# Find the optimal threshold value which maximizes the train precision score using the average of the binary predictions of the chain
pred_train_ensemble = pred_train_chains.mean(axis=0)
threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
optimal_threshold = 0.0
max_precision = 0.0

for thres in threshold:
  curPredictionsTrain = (pred_train_ensemble >= thres)
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'micro')
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'macro')
  if max_precision < precisionTrain:
    max_precision = precisionTrain
    optimal_threshold = thres

# Make predictions on test data using the optimal threshold value found above using thresold
pred_test_chains = np.array([chain.predict(vectorizedTestData)
                             for chain in chains])
pred_test_ensemble = pred_test_chains.mean(axis=0)
pred_test_ensemble = (pred_test_ensemble >= optimal_threshold)

# Derive Jaccard scores
chain_jaccard_scores = [jaccard_score(testBinaryLabel, pred_test_chain, average='macro')
                        for pred_test_chain in pred_test_chains]
ensemble_jaccard_score = jaccard_score(testBinaryLabel, pred_test_ensemble, average='macro')
model_scores = chain_jaccard_scores + [ensemble_jaccard_score]
print(model_scores)

# Testing results using ensmeble CC
precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'micro')
print ("ClassifierChain multi-label Test micro,")
print (precisionTest)

precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'macro')
print ("ClassifierChain multi-label Test macro,")
print (precisionTest)

#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(testBinaryLabel.argmax(axis=1), pred_test_ensemble.argmax(axis=1))
print('Confusion Matrix\n')
print(confusion)

#GaussianNB and ClassifierChain and deriving precision_score and jaccard_score
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
vectorizedTrainData=vectorizedTrainData.todense()
vectorizedTestData=vectorizedTestData.todense()
# Fit an ensemble of GaussianNB classifier chains and take the average prediction of all the chains
chains = [ClassifierChain(GaussianNB(), order='random', random_state=i)
          for i in range(20)]
for chain in chains:
  chain.fit(vectorizedTrainData,trainBinaryLabel)

# Make predictions on train data
pred_train_chains = np.array([chain.predict(vectorizedTrainData)
                             for chain in chains])

# Find the optimal threshold value which maximizes the train precision score
pred_train_ensemble = pred_train_chains.mean(axis=0)
threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
optimal_threshold = 0.0
max_precision = 0.0

for thres in threshold:
  curPredictionsTrain = (pred_train_ensemble >= thres)
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'micro')
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'macro')
  if max_precision < precisionTrain:
    max_precision = precisionTrain
    optimal_threshold = thres

# Make predictions on test data using the optimal threshold value found above using thresold
pred_test_chains = np.array([chain.predict(vectorizedTestData)
                             for chain in chains])
pred_test_ensemble = pred_test_chains.mean(axis=0)
pred_test_ensemble = (pred_test_ensemble >= optimal_threshold)

# Derive Jaccard scores
chain_jaccard_scores = [jaccard_score(testBinaryLabel, pred_test_chain, average='macro')
                        for pred_test_chain in pred_test_chains]
ensemble_jaccard_score = jaccard_score(testBinaryLabel, pred_test_ensemble, average='macro')
model_scores = chain_jaccard_scores + [ensemble_jaccard_score]
print(model_scores)

# Testing results using ensmeble CC
precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'micro')
print ("ClassifierChain multi-label Test micro,")
print (precisionTest)

precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'macro')
print ("ClassifierChain multi-label Test macro,")
print (precisionTest)

####GradientBoostingClassifier and ClassifierChain
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
# Fit an ensemble of Gradient Boosting classifier chains and take the average prediction of all the chains
chains = [ClassifierChain(GradientBoostingClassifier(), order='random', random_state=i)
          for i in range(20)]
for chain in chains:
  chain.fit(vectorizedTrainData,trainBinaryLabel)

# Make predictions on train data
pred_train_chains = np.array([chain.predict(vectorizedTrainData)
                             for chain in chains])

# Find the optimal threshold value which maximizes the train precision score
pred_train_ensemble = pred_train_chains.mean(axis=0)
threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
optimal_threshold = 0.0
max_precision = 0.0

for thres in threshold:
  curPredictionsTrain = (pred_train_ensemble >= thres)
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'micro')
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'macro')
  if max_precision < precisionTrain:
    max_precision = precisionTrain
    optimal_threshold = thres

# Make predictions on test data using the optimal threshold value found above using thresold
pred_test_chains = np.array([chain.predict(vectorizedTestData)
                             for chain in chains])
pred_test_ensemble = pred_test_chains.mean(axis=0)
pred_test_ensemble = (pred_test_ensemble >= optimal_threshold)

# Derive Jaccard scores
chain_jaccard_scores = [jaccard_score(testBinaryLabel, pred_test_chain, average='macro')
                        for pred_test_chain in pred_test_chains]
ensemble_jaccard_score = jaccard_score(testBinaryLabel, pred_test_ensemble, average='macro')
model_scores = chain_jaccard_scores + [ensemble_jaccard_score]
print(model_scores)

# Test results using ensmeble CC
precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'micro')
print ("ClassifierChain multi-label Test micro,")
print (precisionTest)

precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'macro')
print ("ClassifierChain multi-label Test macro,")
print (precisionTest)

#DecisionTreeClassifier and ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
# Fit an ensemble of Decision Tree classifier chains and take the average prediction of all the chains
chains = [ClassifierChain(DecisionTreeClassifier(), order='random', random_state=i)
          for i in range(20)]
for chain in chains:
  chain.fit(vectorizedTrainData,trainBinaryLabel)

# Make predictions on train data
pred_train_chains = np.array([chain.predict(vectorizedTrainData)
                             for chain in chains])

# Find the optimal threshold value which maximizes the train precision score
pred_train_ensemble = pred_train_chains.mean(axis=0)
threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
optimal_threshold = 0.0
max_precision = 0.0

for thres in threshold:
  curPredictionsTrain = (pred_train_ensemble >= thres)
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'micro')
  precisionTrain = precision_score(trainBinaryLabel, curPredictionsTrain, average = 'macro')
  if max_precision < precisionTrain:
    max_precision = precisionTrain
    optimal_threshold = thres

# Make predictions on test data using the optimal threshold value found above
pred_test_chains = np.array([chain.predict(vectorizedTestData)
                             for chain in chains])
pred_test_ensemble = pred_test_chains.mean(axis=0)
pred_test_ensemble = (pred_test_ensemble >= optimal_threshold)

# Derive Jaccard scores
chain_jaccard_scores = [jaccard_score(testBinaryLabel, pred_test_chain, average='macro')
                        for pred_test_chain in pred_test_chains]
ensemble_jaccard_score = jaccard_score(testBinaryLabel, pred_test_ensemble, average='macro')
model_scores = chain_jaccard_scores + [ensemble_jaccard_score]
print(model_scores)

# Test results using ensmeble CC
precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'micro')
print ("ClassifierChain multi-label Test micro,")
print (precisionTest)

precisionTest = precision_score(testBinaryLabel, pred_test_ensemble, average = 'macro')
print ("ClassifierChain multi-label Test macro,")
print (precisionTest)