# Naive Bayes Model Parameters
import numpy as np

classes_ = np.array([0, 1])
class_prior_ = np.array([0.4652071931196247, 0.5347928068803753])
theta_ = np.array([[-0.09883094480251715, 0.3432554197357634, -0.16350262487014397, -0.015700730632089488, 0.11572901951934496, 0.057387917847202166, 0.24058590862361776, 0.16721745001564767, 0.008400472960074276, -0.2422360629452795, -0.45417844212875275], [0.08597136280335946, -0.29859206833739765, 0.14222816052300485, 0.013657799307154531, -0.10067071142399137, -0.04992077648989109, -0.2092816017997851, -0.14545962391709, -0.007307428963812081, 0.21071704305912614, 0.395082124366387]])
var_ = np.array([[0.8039085472254792, 1.0045319663633627, 0.8702100177936551, 0.8533293055214369, 1.3966680631066084, 1.0225049749009447, 1.2226908007767265, 0.6933572189458784, 0.9644355806940319, 1.1005277856091729, 0.5185486440044063], [1.154688924876792, 0.8044071521673745, 1.069418579052802, 1.1271853786891837, 0.6331600660266159, 0.9750663725214453, 0.7121360577227326, 1.2212614671524562, 1.0308221027073914, 0.8171076468839537, 1.0832787027344322]])

def predict_proba(X):
    """Predict class probabilities"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    log_proba = []
    for i in range(len(classes_)):
        log_prior = np.log(class_prior_[i])
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var_[i]))
        log_likelihood -= 0.5 * np.sum(((X - theta_[i]) ** 2) / var_[i], axis=1)
        log_proba.append(log_prior + log_likelihood)
    
    log_proba = np.array(log_proba).T
    log_proba -= np.max(log_proba, axis=1, keepdims=True)
    proba = np.exp(log_proba)
    return proba / proba.sum(axis=1, keepdims=True)

def predict(X):
    """Predict class labels"""
    proba = predict_proba(X)
    return classes_[np.argmax(proba, axis=1)]
