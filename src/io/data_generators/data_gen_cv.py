import os
import pandas as pd
from utility_funcs.vid_to_frame import convert_to_annotated_images_cvat
from utility_funcs.stratified_train_test_split import stratified_group_k_fold

clip_list = [clip for clip in os.listdir('data/videos') if clip != '.gitkeep']
irr_conf = 3

trash_vids = [3, 39, 48]
df = pd.read_csv('data/frames.csv').dropna()
df = df.loc[~df['video'].isin(trash_vids)]
df['flow_class_str'] = df.FLOW_majority.astype(str)
df['final_class_comb'] = df[['flow_class_str', 'ETHNICITY', 'GENDER']].agg('_'.join, axis=1)

groups = df['video']
y = df['final_class_comb']

mapper = dict()
for i, class_type in enumerate(y.unique()):
    mapper[class_type] = i

nomvar = y.replace(mapper)

for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(nomvar, groups, k=5, seed=123)):
    train_vids = groups.iloc[dev_ind].unique()
    validation_vids = groups.iloc[val_ind].unique()
    print('Fold {}'.format(fold_ind))

    train_clips = [x for x in clip_list if int(x.split('.')[0]) in train_vids]
    print('Clips used for Training: ', train_clips)

    validation_clips = [x for x in clip_list if int(x.split('.')[0]) in validation_vids]
    print('Clips used for Validation:', validation_clips)

    convert_to_annotated_images_cvat(train_clips, sense='training', irr_confidence=irr_conf, balance='flow',
                                     fold=fold_ind, n_parallel=5)

    convert_to_annotated_images_cvat(validation_clips, sense='validation', irr_confidence=irr_conf, balance='flow',
                                     fold=fold_ind, n_parallel=5)
