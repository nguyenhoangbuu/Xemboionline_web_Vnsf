import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Flatten,Dense,Dropout,Input,Conv2D,AveragePooling2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import warnings
from joblib import dump, load
import pickle
import cv2

warnings.filterwarnings("ignore")

if __name__== '__main__':
    dirs = os.listdir('Data')
    X_datas = []
    labels = []
    for dir in dirs:
        images = os.listdir(f'Data/{dir}')
        if len(images) > 0:
            label = []
            label.append(dir)
            for image in images:
                root = os.path.join('Data', dir, image)
                print(root)
                img = load_img(root, target_size=(56, 56))
                img = img_to_array(img)
                img = img / 255
                X_datas.append(img)
                labels.append(label)
    enc = OneHotEncoder()
    labels = enc.fit_transform(labels)
    X_datas = np.array(X_datas, dtype='float32')
    labels = labels.toarray()

    gen = ImageDataGenerator()
    X_train, X_valid, y_train, y_valid = train_test_split(X_datas, labels, test_size=0.2, stratify=labels,
                                                          random_state=27)



    # int1 = Input(shape = (56,56,3))
    # out = Conv2D(64,  kernel_size = 3, activation = 'relu', padding = 'same', strides = 2)(int1)
    # out = AveragePooling2D(pool_size = 2, padding = 'same', strides = 1)(out)
    # out = Conv2D(256, kernel_size=3, activation='relu', padding='same', strides=2)(out)
    basemodel = DenseNet121(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (56,56,3)))
    for layer in basemodel.layers:
        layer.trainable = False
    out = basemodel.output
    out = MaxPooling2D(pool_size=2, padding='same', strides=1)(out)
    out = Flatten()(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.3)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(len(dirs), activation='softmax')(out)
    model = Model(inputs = basemodel.input, outputs = out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('model.keras',save_best_only = True)
    stopping = EarlyStopping(monitor='val_loss',patience = 13)
    reducelr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, min_lr = 0.0000001,patience = 3)
    model.fit(gen.flow(X_train, y_train, batch_size=64), epochs=99, validation_data=(X_valid, y_valid),
              callbacks=[checkpoint, stopping, reducelr])


    def predict(model, proof):
        img = load_img(proof, target_size=(56, 56))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        pred = enc.inverse_transform(pred)[0]
        print(pred)


    dump(enc, 'encoder.joblib')
    predict(model, r'Data\Buu\4_2.jpg')
    cv2.imshow('Image', X_valid[0])
    cv2.waitKey(0)
    cv2.imshow('Image', X_valid[1])
    cv2.waitKey(0)

