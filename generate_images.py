from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
import hashlib
from scipy.misc import imsave, imresize

SIZE = 100

image_dirs= ['images/01mkg', 'images/1mkg', 'images/5mkg', 'images/control', 'images/smallset']

dna = {directory: [mh.imread(directory + '/' + f) for f in listdir(directory) if f[-3:] == 'tif']
        for directory in image_dirs}

def process_image(im, folder, remove_bordering=True, test=False):
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

    sizes = mh.labeled.labeled_size(watershed)
    min_size = 15000
    filtered = mh.labeled.remove_regions_where(watershed*bin_image, sizes < min_size)
    if (remove_bordering):
        filtered = mh.labeled.remove_bordering(filtered)
    labeled,nr_objects = mh.labeled.relabel(filtered)
    
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
            imsave(d+'/data/'+hashed+'-'+str(i)+'.png', imresize(cell, (SIZE, SIZE)))
            plt.axis('off')
            plt.show()

for d, ims in dna.items():
    for im in ims:
        process_image(im, d, test=True)