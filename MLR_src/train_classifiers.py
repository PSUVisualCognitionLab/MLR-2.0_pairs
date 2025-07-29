# prerequisites

import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import utils
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt
from joblib import dump

def train_classifiers(dataloaders, vae, checkpoint_folder):
    print('training object on object map')
    clf_oo = classifier_train(vae, 'object', dataloaders['quickdraw'])
    dump(clf_oo, f'checkpoints/{checkpoint_folder}/oo.joblib')
    pred_oo,  OOreport = classifier_test(vae, 'object', clf_oo, dataloaders['quickdraw'],'quickdraw', 1)

    print('training object on color map')
    clf_co = classifier_train(vae, 'color', dataloaders['quickdraw'])
    dump(clf_co, f'checkpoints/{checkpoint_folder}/co.joblib')
    pred_co,  coreport = classifier_test(vae, 'color', clf_co, dataloaders['quickdraw'],'quickdraw', 1)

    print('training object on color map')
    clf_co = classifier_train(vae, 'color', dataloaders['quickdraw'])
    dump(clf_co, f'checkpoints/{checkpoint_folder}/co.joblib')
    pred_co,  coreport = classifier_test(vae, 'color', clf_co, dataloaders['quickdraw'],'quickdraw', 1)




def classifier_train(vae, whichcomponent, train_dataset):
    vae.eval()
    clf = svm.SVC(C=10, gamma='scale', kernel='rbf', probability= True)  # define the classifier for shape
    passin = 'digit'   #temporary until we fix this in activations
    labelindex = 0
 
    if(whichcomponent== 'object'):
        passin = 'object'
        whichcomponent = 'shape' #used to fix a hack in activations where the same label is used for objects and shapes
        
    if(whichcomponent== 'color'):
        labelindex = 1   
    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        data = data[1]
        train_labels=labels[labelindex].clone()   #0 for shape/object, 1 for color
        utils.save_image(data[0:10],'train_sample.png')
        data = data.to(device)        
        activations = vae.activations(data, False, None, passin)
        z = activations[whichcomponent].to(device)
        clf.fit(z.cpu().numpy(), train_labels.cpu())
    return clf

def classifier_test(vae, whichcomponent, clf, test_dataset, dataname,confusion_mat=0):
    vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    vae.eval()
    device = next(vae.parameters()).device
    passin = 'digit'   #temporary until we fix this in activations
    labelindex = 0
 
    if(whichcomponent== 'object'):
        passin = 'object'
        whichcomponent = 'shape' #used to fix a hack in activations where the same label is used for objects and shapes
        
    if(whichcomponent== 'color'):
        labelindex = 1   
    with torch.no_grad():
        data, labels = next(iter(test_dataset))
        data=data[1]
        test_labels=labels[labelindex].clone() #0 for shape/object, 1 for color
        data = data.cuda()
        activations = vae.activations(data, False, None, passin)
        z = activations[whichcomponent].to(device)
        pred = clf.predict(z.cpu())
        test_labels = test_labels.cpu().numpy()

        report = accuracy_score(pred,test_labels)#torch.eq(test_shapelabels.cpu(), pred_ss).sum().float() / len(pred_ss)
     
        if confusion_mat == 1:
            cm = confusion_matrix(test_labels, pred)
            # Plot the confusion matrix
            '''
            plt.imshow(cm, cmap="Greys")
            plt.title("Confusion Matrix for " + dataname+ ' classification from ' + whichcomponent)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            for i in range(0,36):
                plt.annotate(f'{vals[i]}', (i,i), fontsize=10)
            plt.draw()
            ....
            plt.show()
            '''

            print('----**********------- '+ dataname+ ' classification from ' + whichcomponent)
            print(confusion_matrix(test_labels, pred))
            print(classification_report(test_labels, pred))
            print('accuracy:')
            print(report)
            

    return pred, report
