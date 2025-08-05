# prerequisites

import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import utils
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import matplotlib.pyplot as plt
from joblib import dump
import joblib
'''
dataset options: 
emnist-map
mnist-map
emnist-skip
quickdraw
mnist-skip
'''

def train_classifiers(dataloaders, vae, checkpoint_folder):


#this takes days to run 
#    print('training object data on skip using object labels')   #this should be low but above chance
#    clf_oko = classifier_train(vae, 'skip', dataloaders['quickdraw'],'object')
#    dump(clf_oko, f'checkpoints/{checkpoint_folder}/oco.joblib')
#    pred_oko,  coreport = classifier_test(vae, 'skip', clf_oco, dataloaders['quickdraw'],'quickdraw','object' ,1)


    print('training object data on color map using color labels')   #this should be high
    clf_occ = classifier_train(vae, 'color', dataloaders['quickdraw'], 'color')
    dump(clf_occ, f'checkpoints/{checkpoint_folder}/occ.joblib')
    pred_occ,  coreport = classifier_test(vae, 'color', clf_occ, dataloaders['quickdraw'],'quickdraw','color', 1)

    print('training object data on object map using object labels')  #this should be high
    clf_ooo = classifier_train(vae, 'object', dataloaders['quickdraw'],'object')
    dump(clf_ooo, f'checkpoints/{checkpoint_folder}/ooo.joblib')
    pred_ooo,  oooreport = classifier_test(vae, 'object', clf_ooo, dataloaders['quickdraw'],'quickdraw','object', 1)

    print('training object data on color map using object labels')   #this should be low but above chance
    clf_oco = classifier_train(vae, 'color', dataloaders['quickdraw'],'object')
    dump(clf_oco, f'checkpoints/{checkpoint_folder}/oco.joblib')
    pred_oco,  coreport = classifier_test(vae, 'color', clf_oco, dataloaders['quickdraw'],'quickdraw','object' ,1)

#how to load a classifier
    print('test object data on color map using color labels')   #this should be high
    clf_occ = joblib.load(f'checkpoints/{checkpoint_folder}/occ.joblib')
    pred_occ,  coreport = classifier_test(vae, 'color', clf_occ, dataloaders['quickdraw'],'quickdraw','color', 1)



def classifier_train(vae, whichcomponent, train_dataset,whichlabel):
    #trains an svm using a given dataset, from a given latent space (whichcomponent) and datalabel
    #print(' Training'+ whichlabel + ' classification from the map ' + whichcomponent + ' using dataset' + dataname)

    vae.eval()
    clf = svm.SVC(C=10, gamma='scale', kernel='rbf', probability= True)  # define the classifier for shape
    passin = 'digit'   #temporary until we fix this in activations
   
    labelindex = 0  #shape by default
    if(whichlabel== 'color'):  
        labelindex = 1   
 
    if(whichcomponent== 'object'):
        passin = 'object'
        whichcomponent = 'shape' #used to fix a hack in activations where the same label is used for objects and shapes
        
    device = next(vae.parameters()).device
    with torch.no_grad():
        data, labels = next(iter(train_dataset))
        data = data[1]
        train_labels=labels[labelindex].clone()   #0 for shape/object, 1 for color
        utils.save_image(data[0:10],'train_sample.png')
        data = data.to(device)        
        activations = vae.activations(data, False, None, passin)  #pass in determines whether object or digit map is used for activations
        z = activations[whichcomponent].to(device)
        clf.fit(z.cpu().numpy(), train_labels.cpu())
    return clf

def classifier_test(vae, whichcomponent, clf, test_dataset, dataname,whichlabel,verbose=0):
    vals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    vae.eval()
    device = next(vae.parameters()).device
    passin = 'digit'   #temporary until we fix this in activations
    labelindex = 0  #shape by default
    if(whichlabel== 'color'):
        labelindex = 1  
 
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
     
        if verbose == 1:
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

            print('test accuracy '+ whichlabel + ' classification from the map ' + whichcomponent + ' using dataset' + dataname)
            print(confusion_matrix(test_labels, pred))
            print(classification_report(test_labels, pred))
            print('accuracy:')
            print(report)
            

    return pred, report
