import os
import pandas as pd
import sys
from utility_funcs.vid_to_frame import convert_to_annotated_images_cvat
from utility_funcs.stratified_train_test_split import stratified_group_k_fold

irr_conf = 3
if sys.argv[1] == 'split':
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

    for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(nomvar, groups, k=3, seed=123)):
        training = groups.iloc[dev_ind].unique()
        validation = groups.iloc[val_ind].unique()

        assert len(set(training) & set(validation)) == 0
    train_data = df.loc[df['video'].isin(training)]
    test_data = df.loc[df['video'].isin(validation)]

    print("Train", training)
    print(train_data['final_class_comb'].value_counts(normalize=False).sum())
    print(train_data['final_class_comb'].value_counts(normalize=True))
    print('Validation', validation)
    print(test_data['final_class_comb'].value_counts(normalize=False).sum())
    print(test_data['final_class_comb'].value_counts(normalize=True))
    print(df['final_class_comb'].value_counts(normalize=False).sum())

    training = [str(x) + '.mp4' for x in training]
    validation = [str(x) + '.mp4' for x in validation]
else:
    training = ['1.mp4', '11.mp4', '13.mp4', '14.mp4', '15.mp4', '16.mp4', '17.mp4', '18.mp4', '22.mp4', '23.mp4',
                '25.mp4', '27.mp4', '30.mp4', '31.mp4', '32.mp4', '34.mp4', '35.mp4', '38.mp4', '4.mp4', '43.mp4',
                '44.mp4', '46.mp4', '47.mp4', '49.mp4', '5.mp4', '50.mp4', '51.mp4', '53.mp4', '54.mp4', '55.mp4',
                '57.mp4', '58.mp4', '59.mp4', '60.mp4', '7.mp4']

    validation = ['28.mp4', '10.mp4', '12.mp4', '19.mp4', '2.mp4', '20.mp4', '21.mp4', '24.mp4', '26.mp4', '36.mp4',
                  '37.mp4',
                  '40.mp4', '41.mp4', '42.mp4', '45.mp4', '52.mp4', '6.mp4', '9.mp4']
    print('pre-defined split')

input('Check the proportions above and press enter to continue... \n' +
      'If the split is not sufficient enough press CTRL + C to stop this process.')

clip_list = [clip for clip in os.listdir('data/videos') if clip != '.gitkeep']

train_clips = [x for x in clip_list if x in training]
print('Clips used for training: ', train_clips)

validation_clips = [x for x in clip_list if x in validation]
print('Clips used for testing:', validation_clips)

convert_to_annotated_images_cvat(train_clips, sense='training', irr_confidence=irr_conf, balance='flow',
                                 fold='initial', n_parallel=5)

convert_to_annotated_images_cvat(validation_clips, sense='validation', irr_confidence=irr_conf, balance='flow',
                                 fold='initial', n_parallel=5)
