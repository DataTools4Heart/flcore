from sklearn.metrics import confusion_matrix

def measurements_metrics(model,X_test, y_test):
    accuracy = model.score( X_test, y_test )
    print('accuracy client')
    print(accuracy)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    balanced_accuracy = (sensitivity+specificity)/2
    precision = tp/(tp+fp)
    F1_score = (2*precision*sensitivity)/(precision+sensitivity)
    #return {'accuracy':accuracy,'specificity':specificity,'sensitivity':sensitivity,\
    #        'balanced_accuracy':balanced_accuracy, 'precision':precision,'F1_score': F1_score}
    return accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score


def get_metrics(y_pred, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    balanced_accuracy = (sensitivity+specificity)/2
    precision = tp/(tp+fp)
    F1_score = (2*precision*sensitivity)/(precision+sensitivity)

    # Create a dict of metrics
    metrics = {'accuracy': accuracy,'specificity': specificity,'sensitivity': sensitivity,\
                 'balanced_accuracy': balanced_accuracy, 'precision': precision,'F1_score': F1_score}
    
    # Convert each metric value to float
    for key in metrics.keys():
        metrics[key] = float(metrics[key])
    
    return metrics
