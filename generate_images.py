from os import listdir, path, makedirs
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
import hashlib
from scipy.misc import imsave, imresize
from params import img_size
from shutil import rmtree

image_dirs= ['images/01mkg', 'images/1mkg', 'images/5mkg', 'images/control']

data_dir = '/data'+str(img_size)+'/'

dna = {directory: [mh.imread(directory + '/data/' + f) for f in listdir(directory+'/data') if f[-3:] == 'tif']
        for directory in image_dirs}

def process_image(im, d, remove_bordering=True, test=False):
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
            print("Cell {} average luminescence is {}".format(i, fin_result[i]))
            plt.axis('off')
            bbox = mh.bbox((labeled==i))
            cell = (im*(labeled==i))[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            plt.imshow(cell)
            hashed = hashlib.sha1(im).hexdigest()

            imsave(d+data_dir+hashed+'-'+str(i)+'.png', imresize(cell, (img_size, img_size)))
            plt.axis('off')
            plt.show()

test = False
for d, ims in dna.items():
    if not test and path.exists(d+data_dir):
        rmtree(d+data_dir[:-1])
    makedirs(d+data_dir)
    
    for im in ims:
        process_image(im, d, test=test)