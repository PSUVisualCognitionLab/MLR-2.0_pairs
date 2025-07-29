# prerequisites
from MLR_src.classifiers import classifier_shape_train, classifier_color_train, clf_sc, clf_ss, clf_cc, clf_cs, classifier_shape_test, classifier_color_test
from joblib import dump

def train_classifiers(dataloaders, vae, checkpoint_folder):
    print('training shape classifiers')
    #print(dataloaders)
    classifier_shape_train(vae, 'object', dataloaders['quickdraw'])
    
    #dump(clf_sc, f'checkpoints/{checkpoint_folder}/sc.joblib')
    dump(clf_ss, f'checkpoints/{checkpoint_folder}/ss.joblib')

    pred_ss, pred_sc, SSreport, SCreport = classifier_shape_test(vae, 'cropped', clf_ss, clf_sc, dataloaders['quickdraw'],1)
    print('accuracy:')
    print('SS:',SSreport)
    print('SC:',SCreport)

    print('training color classifiers')
    classifier_color_train(vae, 'cropped', dataloaders['quickdraw'])
    dump(clf_cc, f'checkpoints/{checkpoint_folder}/cc.joblib')
    

    pred_cc, pred_cs, CCreport, CSreport = classifier_color_test(vae, 'cropped', clf_cc, clf_cs, dataloaders['quickdraw'],1)
    print('accuracy:')
    print('CC:',CCreport)
    print('CS:',CSreport)