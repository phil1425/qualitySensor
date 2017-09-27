import neuralnet as net
import numpy as np
import sys
import h5py

from tflearn.data_utils import image_preloader
from tflearn.data_flow import ArrayFlow
from tflearn import reshape

epochs = int(sys.argv[1])

X, Y = image_preloader('sleevesPhotos/training',
                        image_shape=(480, 640),
                        grayscale=False,
                        mode='folder',
                        categorical_labels=True,
                        normalize=True)

val_X, val_Y = image_preloader('sleevesPhotos/validation',
                                image_shape=(480, 640),
                                grayscale=False,
                                mode='folder',
                                categorical_labels=True,
                                normalize=True)
model = net.model

model.fit(np.array(X),np.array(Y),
            n_epoch=epochs,
            validation_set=(np.array(val_X),np.array(val_Y)),
            show_metric=True,
            run_id="deep_nn")

model.save('models/final-model.tflearn')
