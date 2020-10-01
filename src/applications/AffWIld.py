import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from src.models.utility_funcs.preprocessor import *
from src.models.utility_funcs.flush_directory import *
import pandas as pd

flush_directory('cuda_utilities/caches/', exclude='gitkeep')

img_height = 96
img_width = 96
batch_size = 50

data_path = 'data/coins/*/*/single_faces/*.jpg',

cache_dir = 'cuda_utilities/caches/'

test_processor = Preprocessor(batch_size=batch_size, img_height=img_height, img_width=img_width,
                              pad=True, saturation=False, brightness=False, flip=False,
                              quality=False, normalize=False, resnet50=False, rotate=False, label_pos='none')

ds = tf.data.Dataset.list_files(data_path)
labeled_ds = ds.map(test_processor.combine_data, num_parallel_calls=AUTOTUNE)
final_ds = test_processor.prepare_for_training(labeled_ds, cache=False)


# Block 1
conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_1',
                                input_shape=(img_height, img_width, 3))
conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_2')
max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')

# Block 2
conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_3')
conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_4')
max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')

# Block3
conv_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_5')
conv_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_6')
conv_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_7')
max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')

# Block4
conv_8 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_8')
conv_9 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_9')
conv_10 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_10')
max_pooling_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')

# Block 5
conv_11 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_11')
conv_12 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_12')
conv_13 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_13')
max_pooling_5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool5')

# Top
flatten = tf.keras.layers.Flatten()
linear_1 = tf.keras.layers.Dense(4096, activation='linear', name='linear_1')
linear_2 = tf.keras.layers.Dense(2, activation='linear', name='linear2')
linear_3 = tf.keras.layers.Dense(2, activation='linear', name='linear3')

model = tf.keras.Sequential([
    # Block 1
    conv_1,
    conv_2,
    max_pooling_1,

    # Block 2
    conv_3,
    conv_4,
    max_pooling_2,

    # Block 3
    conv_5,
    conv_6,
    conv_7,
    max_pooling_3,

    # Block 4
    conv_8,
    conv_9,
    conv_10,
    max_pooling_4,

    # Block 5
    conv_11,
    conv_12,
    conv_13,
    max_pooling_5,

    # Top
    flatten,
    linear_1,
    linear_2,
    linear_3
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model_path = 'data/AffWIld/vggface/model.ckpt-975'

model.load_weights(model_path)

proba = model.predict(final_ds, verbose=1)
flow = (proba > 0.5).astype("int32")



