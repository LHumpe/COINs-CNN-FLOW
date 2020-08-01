'''
  Level | Level for Humans | Level Description
 -------|------------------|------------------------------------
  0     | DEBUG            | [Default] Print all messages
  1     | INFO             | Filter out INFO messages
  2     | WARNING          | Filter out INFO & WARNING messages
  3     | ERROR            | Filter out all messages
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utility_funcs.plotting import *
from utility_funcs.preprocessor import *
from utility_funcs.flush_directory import *

val_data_path = 'data/processed/validation_initial/img'
train_data_path = 'data/processed/train_initial/img'
cache_dir = 'cuda_utilities/caches/'
log_dir = 'cuda_utilities/logdir/'
log_add = 'final/presentation/'

img_height = 224
img_width = 224
batch_size = 50
init_epochs = 12
num_epochs = 15
fine_tune_at = 165

float_policy = '32bit'
flush_directories = True

################################################# Data Prep ############################################################
if float_policy == '16bit':
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

if flush_directories:
    flush_directory(cache_dir, exclude='gitkeep')

''' Training Data'''
training_processor = Preprocessor(batch_size=batch_size, img_height=img_height, img_width=img_width,
                                  pad=True, saturation=False, brightness=False, flip=True,
                                  quality=False, normalize=True, resnet50=True, rotate=False)

train_list_ds = tf.data.Dataset.list_files(train_data_path + '/*.jpg', shuffle=True, seed=123)
train_labeled_ds = train_list_ds.map(training_processor.combine_data, num_parallel_calls=AUTOTUNE)
train_ds = training_processor.prepare_for_training(train_labeled_ds,
                                                   cache=cache_dir + 'train.tfcache')
''' Validation Data'''
validation_processor = Preprocessor(batch_size=batch_size, img_height=img_height, img_width=img_width,
                                    pad=True, saturation=False, brightness=False, flip=True,
                                    quality=False, normalize=True, resnet50=True, rotate=False)

val_list_ds = tf.data.Dataset.list_files(val_data_path + '/*.jpg')
val_labeled_ds = val_list_ds.map(validation_processor.combine_data, num_parallel_calls=AUTOTUNE)
val_ds = validation_processor.prepare_for_training(val_labeled_ds,
                                                   cache=cache_dir + 'val.tfcache')

################################################# Callbacks ############################################################
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir + log_add, histogram_freq=1)

checkpoint_filepath = 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='auto',
    save_best_only=True)

################################################# Model Training #######################################################

base_model = tf.keras.applications.ResNet50(input_shape=(img_height, img_width, 3),
                                            include_top=False,
                                            weights='imagenet')
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dropout_layer = tf.keras.layers.Dropout(0.4)
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    dropout_layer,
    prediction_layer
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=True),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_ds,
                    epochs=init_epochs,
                    validation_data=val_ds,
                    callbacks=[tensorboard_callback, model_checkpoint_callback])

base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.0, nesterov=True),
              loss='binary_crossentropy',
              metrics=['acc'])

history_fine = model.fit(train_ds,
                         epochs=num_epochs,
                         initial_epoch=history.epoch[-1] + 1,
                         validation_data=val_ds,
                         callbacks=[tensorboard_callback, model_checkpoint_callback])
