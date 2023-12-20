#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GET THE DATASET

!curl https://archive.ics.uci.edu/static/public/908/realwaste.zip --output realwaste.zip

!unzip realwaste.zip

!mv realwaste-main/RealWaste .

!rm -r realwaste-main

# SPLIT THE DATASET

def save_dataset(dataset,subgroup):
    i=0
    from PIL.Image import fromarray
    import os
    class_names = dataset.class_names
    index = [0 for i in class_names]
    try:
        os.mkdir(f'RealWaste-{subgroup}')
    except:
        pass
    for c in class_names:
        try:
            os.mkdir(f'RealWaste-{subgroup}/{c}')
        except:
            pass
    for content, categories in dataset:
        for (element,category) in zip(content,categories):
            i+=1
            index[category]+=1
            imgarr = np.array(element).round().astype('uint8')
            img = fromarray(imgarr)
            img.save(f'RealWaste-{subgroup}/{class_names[category]}/{str(index[category])}.jpeg')
    print(i)

batch_size=32
img_height = 524
img_width = 524
seed=0

full_train_ds = tf.keras.utils.image_dataset_from_directory(
  './RealWaste',
  validation_split=0.15,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  './RealWaste',
  validation_split=0.15,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

save_dataset(test_ds,'test')
save_dataset(full_train_ds,'full-train')

train_ds = tf.keras.utils.image_dataset_from_directory(
  './RealWaste-full-train',
  validation_split=0.15/0.85,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  './RealWaste-full-train',
  validation_split=0.15/0.85,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

save_dataset(train_ds,'train')
save_dataset(val_ds,'validation')


# TRAIN AND SAVE BEST MODEL

def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(9)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

input_size=299
learning_rate = 0.001
size=1000
droprate= 0

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                               shear_range=10,
                               rotation_range=45,
                               channel_shift_range=75.0)
train_ds = train_gen.flow_from_directory(
    './RealWaste-train',
    target_size=(input_size, input_size),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    './RealWaste-validation',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

ckpt = keras.callbacks.ModelCheckpoint(
    'waste_classification_model_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[ckpt])

