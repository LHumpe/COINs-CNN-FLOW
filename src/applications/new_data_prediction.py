import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from models.utility_funcs.preprocessor import *
from models.utility_funcs.flush_directory import *

flush_directory('cuda_utilities/caches/', exclude='gitkeep')

data_path = 'data/processed/validation_initial/'
model_path = 'src/models/weights/final_model_weights.hdf5'
cache_dir = 'cuda_utilities/caches/'

test_processor = Preprocessor(batch_size=50, img_height=224, img_width=224,
                              pad=True, saturation=False, brightness=False, flip=False,
                              quality=False, normalize=True, resnet50=True, rotate=False)

ds = tf.data.Dataset.list_files(data_path + '*.jpg')
labeled_ds = ds.map(test_processor.combine_data, num_parallel_calls=AUTOTUNE)
final_ds = test_processor.prepare_for_training(labeled_ds,
                                               cache=cache_dir + 'test.tfcache')

model = tf.keras.models.load_model(model_path)
model.summary()

loss, acc = model.evaluate(final_ds)
print('Loss is: {}'.format(loss))
print('\n Accuracy is: {}'.format(acc))
