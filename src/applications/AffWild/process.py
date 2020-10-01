import src.applications.AffWild.vggface_2 as net
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


network = net.VGGFace(batch_size)
final_ds = tf.keras.backend.placeholder(shape=(50, 96,96,3))
network.setup(final_ds)

network.