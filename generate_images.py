import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import uuid
from keras.layers import Dense, Flatten, Convolution2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from loader import load_data
from os import listdir, path, makedirs
from params import img_rows, img_cols, image_dirs
from random import randrange
from shutil import rmtree
from scipy.misc import imsave, imresize

data_dir = '/data'+str(img_rows)+'/'

dna = {directory: [mh.imread(directory + '/data/' + f) for f in listdir(directory+'/data') if f[-3:] == 'tif']
        for directory in image_dirs}

def process_image(im, d, test = False, remove_bordering = True):
    plt.figure(1, frameon=False)
    sigma = 75
    dnaf = mh.gaussian_filter(im.astype(float), sigma)
    T_mean = dnaf.mean()
    bin_image = im > T_mean

    maxima = mh.regmax(mh.stretch(dnaf))
    maxima,_= mh.label(maxima)

    dist = mh.distance(bin_image)

    dist = 255 - mh.stretch(dist)
    watershed = mh.cwatershed(dist, maxima)
    
    _, old_nr_objects = mh.labeled.relabel(watershed)

    sizes = mh.labeled.labeled_size(watershed)
    min_size = 100000
    filtered = mh.labeled.remove_regions_where(watershed*bin_image, sizes < min_size)

    _, nr_objects = mh.labeled.relabel(filtered)    
    print('Removed', old_nr_objects - nr_objects, 'small regions')
    old_nr_objects = nr_objects
    
    if (remove_bordering):
        filtered = mh.labeled.remove_bordering(filtered)
    labeled,nr_objects = mh.labeled.relabel(filtered)
    
    print('Removed', old_nr_objects - nr_objects, 'bordering cells')
    
    print("Number of cells: {}".format(nr_objects))
    fin_weights = mh.labeled_sum(im.astype(np.uint32), labeled)
    fin_sizes = mh.labeled.labeled_size(labeled)
    fin_result = fin_weights/fin_sizes
    if (test):
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(labeled)
        for i in range(1, nr_objects+1):
            print("Cell {} average luminescence is {}".format(i, fin_result[i]))
            bbox = mh.bbox((labeled==i))
            plt.text((bbox[2]+bbox[3])/2, (bbox[0]+bbox[1])/2, str(i), fontsize=20, color='black')
        plt.show()
    else:
        for i in range(1, nr_objects+1):
            bbox = mh.bbox((labeled==i))
            cell = (im*(labeled==i))[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            hashed = hashlib.sha1(im).hexdigest()
            imsave(d+data_dir+hashed+'-'+str(i)+'.png', imresize(cell, (img_rows, img_cols)))

def gen_cropped(im, d, cnt=100):
    mean_brightness = np.mean(im)
    for i in range(cnt):
        y = randrange(0, im.shape[0]-img_rows)
        x = randrange(0, im.shape[1]-img_cols)
        part = im[y:y+img_rows, x:x+img_cols]
        part_mean = np.mean(part)
        if (part_mean > mean_brightness*2):
            hashed = uuid.uuid4().hex
            imsave(d+data_dir+hashed+'-'+str(i)+'.png', part)    
            
def augment(n = 4):
    for i in range(n):
        for image_dir in image_dirs:
            # Dummy model
            model = Sequential()
            model.add(Convolution2D(1, 1, 1, input_shape=(1, img_rows, img_cols)))     
            model.add(Flatten())
            model.add(Dense(4))
            
            model.compile(loss='mse', optimizer='SGD')
            
            datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=False, # Seems like not working at all!
            rotation_range=180,
            zca_whitening=False,
            #shear_range=0.3,
            #zoom_range=0.1,
            #width_shift_range=0.1,
            #height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
            
            (X_train, Y_train), (X_test, Y_test) = load_data(False, [image_dir])

            save_path = image_dir+data_dir[:-1]+'A'

            if not path.exists(save_path):
                makedirs(save_path)
        
            datagen.fit(X_train)
            model.fit_generator(datagen.flow(X_train, Y_train,
                                             save_to_dir=save_path,
                                             save_prefix='_'+str(i), save_format='png'),
                            samples_per_epoch=X_train.shape[0], nb_epoch=1)

if __name__ == "__main__":
    print("Enter operation type:")
    print("1 for cell detection with saving")
    print("2 for cell detection without saving")
    print("3 for augmentation of already generated data")
    op_type = int(input())
    if op_type == 1:
        for d, ims in dna.items():
            print('Process', d)
            if path.exists(d+data_dir):
                rmtree(d+data_dir[:-1])
            makedirs(d+data_dir)
            
            for im in ims:
                process_image(im, d, test=False)       
    elif op_type == 2:
        for d, ims in dna.items():
            for im in ims:
                process_image(im, d, test=True)       
    elif op_type == 3:
        augment()
    else:
        print("Incorrect operation type")