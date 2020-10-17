import cv2
import os
from fastai.vision import Image, pil2tensor, load_learner
import numpy as np
from tqdm import tqdm
import pandas as pd

# Store the most likely emotion as a string
predictions = []

# Store the different probabilities for each emotion
probabilities = []

flows = []

vids = []

frames =[]

data_path = 'data/processed/YouTube/img/'
pytorch_model_path = 'src/models/weights/FER/'
pytorch_model_name = 'Six_Emotions_NoPrep_ResNet34_05_12_2020-13_39_47.pkl'
images = [img for img in os.listdir(data_path)]

model = load_learner(pytorch_model_path, pytorch_model_name)

for img in tqdm(images):
    vids.append(img.split('_')[0])
    frames.append(img.split('_')[1])
    flows.append(img.split('_')[3])
    img = cv2.imread(data_path + img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_fastai = Image(pil2tensor(img, dtype=np.float32).div_(255))
    pred_class, pred_idx, outputs = model.predict(face_fastai)

    predictions.append(str(pred_class))
    probabilities.append(outputs.data.numpy())

result_frame = pd.DataFrame({'VID': vids,
                             'FRAME': frames,
                             'FLOW':flows,
                             'EMO_C': predictions,
                             'EMO_P' : probabilities,
                             },
                            index=range(0, len(flows)))

result_frame.to_csv('results/yt_analysis/FER_results_private.csv', index=False)
