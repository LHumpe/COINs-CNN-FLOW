import os
import pandas as pd
from utility_funcs.vid_to_frame import convert_to_annotated_images_cvat

clip_list = [clip for clip in os.listdir('data/videos') if clip != '.gitkeep']
irr_conf = 1

trash_vids = []
df = pd.read_csv('data/frames.csv').dropna()
df = df.loc[~df['video'].isin(trash_vids)]

convert_to_annotated_images_cvat(clip_list, sense='YouTube', irr_confidence=irr_conf, balance='none',
                                 fold=1, n_parallel=5)
