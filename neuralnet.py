import tflearn
from tflearn.data_preprocessing import DataPreprocessing
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.data_augmentation import ImageAugmentation

acc = Accuracy()

augmentation = ImageAugmentation()
augmentation.add_random_blur()
augmentation.add_random_rotation(180)

network = input_data(shape=[None, 640, 480, 3], data_augmentation=augmentation)

network = conv_2d(network, 4, 5, strides=2, activation='relu', name = 'conv1')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 4, 5, strides=1, activation='relu', name = 'conv2')
network = conv_2d(network, 4, 3, strides=1, activation='relu', name = 'conv3')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, metric=acc)

model = tflearn.DNN(network, checkpoint_path='models/model-', best_checkpoint_path='models/best-model-')
