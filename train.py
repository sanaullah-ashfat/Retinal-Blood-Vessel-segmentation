import numpy as np
from model import  Unet_model  #,dice_coef,dice_coef_loss
import getdata
import random
from skimage.io import imread, imshow
import os
from PIL import Image
import argparse
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import Model
from keras import activations
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import precision_recall_curve



n_rounds=1
n_round=1
batch_size=100
filter_size=32
val_ratio=0.05
init_lr=2e-4
N_epochs=2
alpha_recip=.001
schedules={'lr_decay':{},'step_decay':{}}
rounds_for_evaluation=range(n_rounds)


dataset='DRIVE'
img_size= (32,32) 
img_out_dir="segmentation_results".format(dataset)
model_out_dir="{}/model_output_result".format(dataset)
auc_out_dir="auc".format(dataset)
train_dir="F:/my model/data/DRIVE/training/".format(dataset)
test_dir="F:/my model/data/DRIVE/test/".format(dataset)
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(auc_out_dir):
    os.makedirs(auc_out_dir)

model = Unet_model(img_size, filter_size,init_lr)
model.summary()
plot(model,model_out_dir+'model.png')

# set training and validation dataset
train_imgs, train_vessels =getdata.ready_image(train_dir, img_size=img_size, dataset=dataset)
train_vessels=np.expand_dims(train_vessels, axis=3)


# set test dataset
test_imgs, test_vessels, test_masks=getdata.ready_image(test_dir,  img_size=img_size, dataset=dataset, mask=True)

checkpointer = ModelCheckpoint(filepath= model_out_dir+'_best_weights.h5', verbose=1, monitor='val_loss',
                               mode='auto', save_best_only=True)

model.fit(train_imgs, train_vessels, epochs=N_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[checkpointer])

##for n_round in range(n_rounds):
##
##    if n_round in rounds_for_evaluation:    
##        generated=model.predict(test_imgs,batch_size=batch_size)
##        generated=np.squeeze(generated, axis=3)
##        vessels_in_mask, generated_in_mask = getdata.pixel_values_in_mask(test_vessels, generated , test_masks)
##        auc_roc=getdata.AUC_ROC(vessels_in_mask,generated_in_mask)
##        auc_pr=getdata.AUC_PR(vessels_in_mask, generated_in_mask)
##
##       
##        segmented_vessel=getdata.remain_in_mask(generated, test_vessels)
##
##        
##        for index in range(segmented_vessel.shape[0]):
##            Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_out_dir,
##                                                                            str(n_round)+"{:02}_test.png".format(index+1)))
##

generated=model.predict(test_imgs,batch_size=batch_size)
generated=np.squeeze(generated, axis=3)


#vessels_in_mask, generated_in_mask = getdata.pixel_values_in_mask(test_vessels, generated , test_masks)
#auc_roc=getdata.AUC_ROC(vessels_in_mask,generated_in_mask)
#auc_pr=getdata.AUC_PR(vessels_in_mask, generated_in_mask)

       
segmented_vessel=getdata.remain_in_mask(generated, test_vessels)


for index in range(segmented_vessel.shape[0]):
    Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_out_dir,
                                                                            str(n_round)+"{:02}_test.png".format(index+1)))















































































