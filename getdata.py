import os
import sys
from PIL import Image, ImageEnhance
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_image_file(path,file_title=None, append_path=True,sort =True):
    if append_path:
        if file_title is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(file_title)]
    else:
        if file_title is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(file_title)]
    
    if sort:
        filenames = sorted(filenames)
    
    return filenames



def image_shape(filenames):
    img = Image.open(filenames)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    print(len(img_shape))
    return img_shape


def image_2_array(filenames):
    img_shape = image_shape(filenames[0])
    if len(img_shape) == 3:
        image_array = np.empty((len(filenames),img_shape[0], img_shape[1], img_shape[2]),dtype = np.float32)

    if len(img_shape) == 2:
        image_array = np.empty((len(filenames),img_shape[0], img_shape[1]),dtype = np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        image_array[file_index] = np.asarray(img).astype(np.float32)
        

    return image_array




def datapath(data_path):
    image_dir = os.path.join(data_path,'images')
    vessel_dir = os.path.join(data_path,'1st_manual')
    mask_dir =os.path.join(data_path,'mask')
    image_files = get_image_file(image_dir,file_title='.tif')
    vessel_files = get_image_file(vessel_dir,file_title='.gif')
    mask_files = get_image_file(mask_dir,file_title='gif')

    return image_files,vessel_files,mask_files



        
def print_metrics(itr, **kargs):
    print ("*** Round {}  ====> ".format(itr)),
    for name, value in kargs.items():
        print (( "{} : {}, ".format(name, value))),
    print ("")
    sys.stdout.flush()
    
def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    
    AUC_ROC=roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    print("AUC_ROC:",AUC_ROC)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' )
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    
    AUC_prec_rec = auc(recall, precision)
    
    print('AUC_prec_rec:',AUC_prec_rec)

    plt.title('precision_recall_curve')

    plt.plot(precision, recall ,'g', label = 'AUC = %0.2f' )
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'b --')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    


class FetchOriginalImage(Iterator):
    def __init__(self,train_images,train_vessels_image,batch_size):
        self.tarin_images = train_images
        self.train_vessel_images = train_vessels_image
        self.n_train_images = train_images.shape[0]
        self.batch_size = batch_size

    def next(self):
        indices = list(np.random.choice(self.n_train_images,self.batch_size))

        return self.tarin_image[indices,:,:,:],self.train_vessel_images[indices,:,:,:]

def pad_imgs(imgs, img_size):
    img_h,img_w=imgs.shape[1], imgs.shape[2]
    target_h,target_w=img_size[0],img_size[1]
    if len(imgs.shape)==4:
        d=imgs.shape[3]
        padded=np.zeros((imgs.shape[0],target_h, target_w,d))
    elif len(imgs.shape)==3:
        padded=np.zeros((imgs.shape[0],img_size[0],img_size[1]))
    padded[:,(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
    
    return padded



def ready_image(target_dir,img_size,dataset,mask= False):
    if dataset == "DRIVE":
        image_files,vessel_files,mask_files = datapath(target_dir)
    else:
        print("You Have Choose Wrong DATA_SET")

    main_image = image_2_array(image_files)
    vessel_image = image_2_array(vessel_files)/255.
    main_image=pad_imgs(main_image, img_size)
    vessel_image=pad_imgs(vessel_image, img_size)
    assert(np.min(vessel_image)>=0 and np.max(vessel_image)<=1)
   
    mask_image = image_2_array(mask_files)/255.
    mask_image=pad_imgs(mask_image, img_size)
    
   
    
    # z score with mean, std of each image
    n_all_imgs=main_image.shape[0]
    for index in range(n_all_imgs):
        mean=np.mean(main_image[index,...][main_image[index,...,0] > 40.0],axis=0)
        print(len(mean))
        std=np.std(main_image[index,...][main_image[index,...,0] > 40.0],axis=0)
        print(len(std))
        assert len(mean)==3 and len(std)==3
        main_image[index,...]=(main_image[index,...]-mean)/std
    
    if mask:
        return main_image, vessel_image, mask_image
    else:
        return main_image, vessel_image
   

def pixel_values_in_mask(true_vessels, pred_vessels,masks):
    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)<=1.0 and np.min(true_vessels)>=0.0
    assert np.max(masks)<=1.0 and np.min(masks)>=0.0
    assert pred_vessels.shape[0]==true_vessels.shape[0] and masks.shape[0]==true_vessels.shape[0]
    assert pred_vessels.shape[1]==true_vessels.shape[1] and masks.shape[1]==true_vessels.shape[1]
    assert pred_vessels.shape[2]==true_vessels.shape[2] and masks.shape[2]==true_vessels.shape[2]
     
    return true_vessels[masks==1].flatten(), pred_vessels[masks==1].flatten() 


def remain_in_mask(imgs,masks):
    imgs[masks==0]=0
    return imgs

def load_images_under_dir(path_dir):
    files=all_files_under(path_dir)
    return imagefiles2arrs(files)

def crop_to_original(imgs, ori_shape):
    pred_shape=imgs.shape
    assert len(pred_shape)<4

    if ori_shape == pred_shape:
        return imgs
    else: 
        if len(imgs.shape)>2:
            ori_h,ori_w =ori_shape[1],ori_shape[2]
            pred_h,pred_w=pred_shape[1],pred_shape[2]
            return imgs[:,(pred_h-ori_h)//2:(pred_h-ori_h)//2+ori_h,(pred_w-ori_w)//2:(pred_w-ori_w)//2+ori_w]
        else:
            ori_h,ori_w =ori_shape[0],ori_shape[1]
            pred_h,pred_w=pred_shape[0],pred_shape[1]
            return imgs[(pred_h-ori_h)//2:(pred_h-ori_h)//2+ori_h,(pred_w-ori_w)//2:(pred_w-ori_w)//2+ori_w]




















































































































































