import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from rampwf.workflows.image_classifier import get_nb_minibatches
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClassifier(object):

    def __init__(self):
        self.batch_size = 5
        self.img_width, self.img_height = 32, 32
        self.model = Sequential()
        self._build_model()
    
    def _build_model(self):
        
        # Forme de l'input en fonction du backend
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, self.img_width, self.img_height)
        else:
            self.input_shape = (self.img_width, self.img_height, 3)

        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        # Compute the probabilities of the ten classes
        self.model.add(Dense(10)) 
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def _transform(self, x):
        x = resize(x, (self.img_width, self.img_height), preserve_range=True, anti_aliasing=False)
        # bringing input between 0 and 1
        x = x / 255.
        return x

    def _build_train_generator(self, img_loader, indices, batch_size,
                               shuffle=False):
        indices = indices.copy()
        nb = len(indices)
        X = np.zeros((batch_size, self.img_width, self.img_height, 3))
        Y = np.zeros((batch_size, 10))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                Y[:] = 0
                for i, img_index in enumerate(indices[start:stop]):
                    x, y = img_loader.load(img_index)
                    if len(x.shape)!=3 or x.shape[2]!=3:
                        print('Image passed')
                        Y[i, 0] = 1
                        continue
                    x = self._transform(x)
                    X[i] = x
                    Y[i, y] = 1
                yield X[:bs], Y[:bs]

    def _build_test_generator(self, img_loader, batch_size):
        nb = len(img_loader)
        X = np.zeros((batch_size, self.img_width, self.img_height, 3))
        while True:
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                for i, img_index in enumerate(range(start, stop)):
                    x = img_loader.load(img_index)
                    x = self._transform(x)
                    X[i] = x
                yield X[:bs]

    def fit(self, img_loader):
        np.random.seed(24)
        nb = len(img_loader)
        nb_train = int(nb * 0.9)
        nb_valid = nb - nb_train
        indices = np.arange(nb)
        np.random.shuffle(indices)
        ind_train = indices[0: nb_train]
        ind_valid = indices[nb_train:]

        gen_train = self._build_train_generator(
            img_loader,
            indices=ind_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        gen_valid = self._build_train_generator(
            img_loader,
            indices=ind_valid,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_valid, self.batch_size),
            epochs=1,
            max_queue_size=16,
            workers=1,
            use_multiprocessing=False,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, self.batch_size),
            verbose=1
        )

    def predict_proba(self, img_loader):
        nb_test = len(img_loader)
        gen_test = self._build_test_generator(img_loader, self.batch_size)
        return self.model.predict_generator(
            gen_test,
            steps=get_nb_minibatches(nb_test, self.batch_size),
            max_queue_size=16,
            workers=1,
            use_multiprocessing=False,
            verbose=0
        )
