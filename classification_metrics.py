import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import metriculous

def make_confusion_matrix(cf: np.array,
                          group_names: list = None,
                          categories: str = 'auto',
                          count: bool = True,
                          percent: bool = True,
                          cbar: bool = True,
                          xyticks: bool = True,
                          xyplotlabels: bool = True,
                          sum_stats: bool = True,
                          figsize: tuple = None,
                          cmap: str = 'Blues',
                          title: str = ""):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

#     rc('text', usetex=True)
#     rc('font', family='serif')
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        # figsize = plt.rcParams.get('figure.figsize')
        figsize = (cf.shape[0]+2, cf.shape[0]+2) # increase plot size if there are many classes

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel(r'True label', fontsize=10+cf.shape[0])
        plt.xlabel('Predicted label' + stats_text,fontsize=10+cf.shape[0])
    else:
        plt.xlabel(stats_text,fontsize=10+cf.shape[0])
    
    if title:
        plt.title(f"{title} Confusion Matrix", fontsize=10+cf.shape[0])
    plt.show()
    

def make_pred_histogram(probs: list[list], title: str="", fig_size: tuple[int] = (10,10)) -> None:
    """
    This function plots a histogram of the prediction probabilities of a classifier.
    probs (list[list]):     a 2D array/list of the predictions made by the model
    title (str):            the plot title
    fig_size (tuple[int]):  a tuple describing the plot size of the histogram
    
    Plots the histogram and does not return anything.
    """
#     rc('text', usetex=True)
#     rc('font', family='serif')
    _ = plt.figure(figsize=fig_size)
    flattened = [p for ps in probs for p in ps]
    _ = plt.hist(flattened, bins=max(10,int(len(flattened)/100)))
    plt.title(f"{title} Histogram", fontsize=10+fig_size[0])
    plt.xlabel("Prediction Probability", fontsize=10+fig_size[0])
    plt.ylabel("Count", fontsize=10+fig_size[0])
    plt.show()
    
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from matplotlib import rc
def get_metrics(y_test: list[int], predictions: list[list], model_names: list[str] = None, class_names: list[str] = None) -> None:
    if model_names is not None:
        assert len(model_names) == len(predictions) # make sure there's a name given for each model
    prediction_probabilities = []
    for idx,preds in enumerate(predictions):
        if np.max(preds) > 1:
            pred_probs = None
            disc_preds = preds
        else:
            pred_probs = preds
            disc_preds = np.argmax(pred_probs,axis=1)
            prediction_probabilities.append(pred_probs)
            if model_names is not None:
                make_pred_histogram(pred_probs, model_names[idx])    
            else: 
                make_pred_histogram(pred_probs)    
            
        conf_mat = confusion_matrix(disc_preds,y_test)

        if model_names is not None:
            make_confusion_matrix(conf_mat, title=model_names[idx])
        else:
            make_confusion_matrix(conf_mat)
        
    if len(prediction_probabilities) == len(predictions):
        import metriculous
        metriculous.compare_classifiers(
            ground_truth=y_test,
            model_predictions=prediction_probabilities,
            model_names=model_names,
            class_names=class_names,
            one_vs_all_figures=True
        ).display()
