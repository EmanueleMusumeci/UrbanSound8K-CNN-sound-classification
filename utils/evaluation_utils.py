import numpy as np

import sklearn

def evaluate_class_prediction(labels, predictions, compute_confusion_matrix = True):
    '''Returns a dictionary containing performance metrics for the provided predictions on the Argument Classification task

    Evaluation function copied from the docker framework and adapted to also return a
    confusion matrix and a dictionary containing the distribution of predicted labels
    '''

    gold_distribution = {}
    pred_distribution = {}

    for prediction, label in zip(predictions, labels):
        gold_distribution[prediction] = 1 + gold_distribution.get(prediction, 0)
        pred_distribution[label] = 1 + pred_distribution.get(label, 0)

    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    f1 = sklearn.metrics.f1_score(labels, predictions, average="macro")
    precision = sklearn.metrics.precision_score(labels, predictions, average="macro")
    recall = sklearn.metrics.recall_score(labels, predictions, average="macro")  

    if compute_confusion_matrix: 
        cm = sklearn.metrics.confusion_matrix(np.array(labels), np.array(predictions))
    else: 
        cm = None

    keys = set(gold_distribution.keys()) | set(pred_distribution.keys())
    distribution={}
    for k in keys:
        distribution[k] = (gold_distribution.get(k, 0),pred_distribution.get(k, 0))

    return {
        'accuracy' : accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion matrix': cm,
        'distribution': distribution,
    }

def print_table(title, results, fields):
    output = title+"\n"
    output += "=" * len(title) + "\n"
    for field in fields:
        if field=="confusion matrix" or field=="distribution" or field=="gradient_stats": continue
        output+=str(field)+" = {:0.4f}\n".format(results[field])

    return output
  
def print_distribution(results):
    '''Prints the prediction/gold distribution in results'''
    distribution = results["distribution"]
    
    output = ""
    for k in distribution.keys():
        output+="\t# "+str(k)+": ("+str(distribution.get(k, 0)[0])+", "+str(distribution.get(k, 0)[1])+")\n"
    
    return output

def print_confusion_matrix(results):
    '''Prints the confusion matrix in results'''
    cm = results["confusion matrix"].tolist()
    
    output = ""
    for row in cm:
        output+=str(row)+"\n"
    
    return output
