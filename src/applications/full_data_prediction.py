import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ..models.utility_funcs.preprocessor import *
from ..models.utility_funcs.flush_directory import *
import pandas as pd
flush_directory('cuda_utilities/caches/', exclude='gitkeep')

img_height = 224
img_width = 224
batch_size = 50

data_path = 'data/processed/YouTube/img/'
model_path = 'src/models/weights/final_model_weights.hdf5'
cache_dir = 'cuda_utilities/caches/'

test_processor = Preprocessor(batch_size=batch_size, img_height=224, img_width=224,
                              pad=True, saturation=False, brightness=False, flip=False,
                              quality=False, normalize=True, resnet50=True, rotate=False, label_pos='none')

ds = tf.data.Dataset.list_files(data_path + '*.jpg')
labeled_ds = ds.map(test_processor.combine_data, num_parallel_calls=AUTOTUNE)
final_ds = test_processor.prepare_for_training(labeled_ds, cache=False)

model = tf.keras.models.load_model(model_path)

proba = model.predict(final_ds, verbose=1)
flow = proba.astype("float32")

paths = []
for images, labels in final_ds.take(-1):
    paths.extend(labels.numpy().tolist())

result_frame = pd.DataFrame({'PROB': proba.reshape(-1), 'FILE': paths},
                            index=range(0, len(flow)))

result_frame.to_csv('data/yt_analysis/results.csv', index=False)
