import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from src.models.utility_funcs.preprocessor import *
import pandas as pd

img_height = 197
img_width = 197
batch_size = 50

data_path = 'data/processed/YouTube/img/'
model_path = 'src/models/weights/FER_Model.h5'
cache_dir = 'cuda_utilities/caches/'

test_processor = Preprocessor(batch_size=batch_size, img_height=img_height, img_width=img_width,
                              pad=False, saturation=False, brightness=False, flip=False,
                              quality=False, normalize=True, resnet50=False, rotate=False, label_pos='none')

ds = tf.data.Dataset.list_files(data_path + '*.jpg')
labeled_ds = ds.map(test_processor.combine_data, num_parallel_calls=AUTOTUNE)
final_ds = test_processor.prepare_for_training(labeled_ds, cache=False)


model = tf.keras.models.load_model(model_path)
class_names = list(["Anger", "Disgust", "Fear", "Happinness", "Sadness", "Surprise", "Neutral"])

flows = []
paths = []
for images, labels in final_ds.take(-1):
    proba = model.predict(images, verbose=1)
    for prob in proba:
        flows.append(class_names[np.argmax(prob, axis=0)])
    paths.extend(labels.numpy().tolist())


result_frame = pd.DataFrame({'EMO': flows, 'FILE': paths},
                            index=range(0, len(flows)))

result_frame.to_csv('results/yt_analysis/FER_results_public.csv', index=False)
