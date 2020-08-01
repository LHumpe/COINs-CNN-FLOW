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
from utility_funcs.preprocessor import *
from utility_funcs.flush_directory import *
from tensorboard.plugins.hparams import api as hp

################################ CONFIG ################################################################################
img_height = 224
img_width = 224
batch_size = 50
val_data_path = 'data/processed/validation_initial/img'
train_data_path = 'data/processed/train_initial/img'
cache_dir = 'cuda_utilities/caches/'
log_dir = 'cuda_utilities/logdir/fit/'
log_add = 'initial/SGD/'

float_policy = '32bit'
flush_directories = True
fold_ind = 'initial'

############################### PRE-PROCESSING #########################################################################
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

############################# HPARAMS ##################################################################################
HP_INIT_LR = hp.HParam('LR', hp.Discrete([0.001]))
HP_LR_FACTOR = hp.HParam('LR_FACTOR', hp.Discrete([0.00001, 0.0000001]))
HP_MOMENTUM = hp.HParam('MOMENTUM', hp.Discrete([0.5, 0.4]))
HP_INIT_EPOCHS = hp.HParam('INIT_EPOCHS', hp.Discrete([10, 12]))
HP_FINE_EPOCHS = hp.HParam('FINE_EPOCHS', hp.Discrete([1, 2, 3]))
HP_TUNED_LAYERS = hp.HParam('TUNED_LAYERS', hp.Discrete([160, 165]))

############################# MODEL CONFIG #############################################################################
session_num = 0

for learning_rate in HP_INIT_LR.domain.values:
    for lr_factor in HP_LR_FACTOR.domain.values:
        for momentum in HP_MOMENTUM.domain.values:
            for init_epochs in HP_INIT_EPOCHS.domain.values:
                for fine_epochs in HP_FINE_EPOCHS.domain.values:
                    for tuned_layers in HP_TUNED_LAYERS.domain.values:
                        hparams = {
                            HP_INIT_LR: learning_rate,
                            HP_LR_FACTOR: lr_factor,
                            HP_MOMENTUM: momentum,
                            HP_INIT_EPOCHS: init_epochs,
                            HP_FINE_EPOCHS: fine_epochs,
                            HP_TUNED_LAYERS: tuned_layers
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})

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

                        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hparams[HP_INIT_LR],
                                                                        momentum=hparams[HP_MOMENTUM]),
                                      loss='binary_crossentropy',
                                      metrics=['acc'])

                        history = model.fit(train_ds,
                                            epochs=hparams[HP_INIT_EPOCHS],
                                            validation_data=val_ds,
                                            callbacks=[
                                                tf.keras.callbacks.TensorBoard(log_dir=log_dir + log_add + run_name,
                                                                               histogram_freq=1),
                                                hp.KerasCallback(log_dir + run_name, hparams)]
                                            )

                        base_model.trainable = True

                        fine_tune_at = hparams[HP_TUNED_LAYERS]
                        for layer in base_model.layers[:fine_tune_at]:
                            layer.trainable = False

                        model.compile(optimizer=tf.keras.optimizers.SGD(
                            learning_rate=hparams[HP_LR_FACTOR],
                            momentum=hparams[HP_MOMENTUM]),
                            loss='binary_crossentropy',
                            metrics=['acc'])

                        total_epochs = hparams[HP_INIT_EPOCHS] + hparams[HP_FINE_EPOCHS]

                        model.fit(train_ds,
                                  epochs=total_epochs,
                                  initial_epoch=history.epoch[-1] + 1,
                                  validation_data=val_ds,
                                  callbacks=[
                                      tf.keras.callbacks.TensorBoard(log_dir=log_dir + log_add + run_name,
                                                                     histogram_freq=1),
                                      hp.KerasCallback(log_dir + run_name, hparams)]
                                  )
                        session_num += 1
