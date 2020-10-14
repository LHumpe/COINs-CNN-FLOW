import os
import cv2
import pandas as pd
import shutil
from tensorflow.keras.preprocessing.image import save_img, img_to_array
from multiprocessing import Pool


def extract_frames(vid_name, annotations, folder_name):
    vid_source = 'data/videos'

    try:
        vid_nr = vid_name.split('.')[0]
        vid_path = os.path.join(vid_source, vid_name)

        cap = cv2.VideoCapture(vid_path)
    except Exception as e:
        print(e)
        print('\n')
        print(
            'Do you have the files in the correct folder structure? look in line 37-44 for more info')

    vid_annotations = annotations[annotations['video'] == int(vid_nr)]

    frames_list = [i for i in range(
        0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))]

    for n_frame in frames_list:

        frame_annotation = vid_annotations[vid_annotations['frame'] == int(
            n_frame)]

        if len(frame_annotation) == 0:
            # no annotation was found for this frame, skip
            a, b = cap.read()
            continue

        flow = int(frame_annotation['FLOW_majority'].values[0])

        try:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            print('Could not read frame number {} of video{}'.format(
                n_frame, vid_name))
            continue

        # clip to bbox
        xtl = int(round(frame_annotation['xtl'].values[0]))
        ytl = int(round(frame_annotation['ytl'].values[0]))
        xbr = int(round(frame_annotation['xbr'].values[0]))
        ybr = int(round(frame_annotation['ybr'].values[0]))
        face = [(xtl, ytl), (xbr, ybr)]
        gray = gray[face[0][1]:face[1][1], face[0][0]:face[1][0]]

        img = img_to_array(gray)

        img_path = 'data/processed/{}/img/{}_{}_F_{}_.jpg'.format(
            folder_name, vid_nr, n_frame, flow)

        save_img(img_path, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('All frames from {} have been extracted \n'.format(vid_name))

    cap.release()


def convert_to_annotated_images_cvat(clip_list, sense='training', irr_confidence=2, balance='flow', fold=1,
                                     n_parallel=5):
    if sense == 'training':
        folder_name = 'train_{}'.format(fold)
    elif sense == 'validation':
        folder_name = 'validation_{}'.format(fold)
    elif sense == 'test':
        folder_name = 'test'
    elif sense == 'YouTube':
        folder_name = 'YouTube'
    else:
        raise Exception(
            'You have not chosen a proper sense. The possible choices are:[training, validation, test] ' +
            'and can be set by e.g. sense="training"')

    # clean old images
    shutil.rmtree('data/processed/{}/img'.format(folder_name), ignore_errors=True)
    os.makedirs('data/processed/{}/img'.format(folder_name))

    annotations = pd.read_csv('data/frames.csv')
    annotations = annotations[annotations['video'].isin([x.replace('.mp4', '') for x in clip_list])]

    # reduce dataset to annotations that have the required irr_confidence
    print('Annotated frames:', len(annotations))
    annotations = annotations[annotations['majority'] >= irr_confidence]
    print('Annotated frames that have the required IRR confidence of',
          irr_confidence, 'are:', len(annotations))

    if balance == 'flow':
        # balance annotations such that the dataframe is 50% flow, 50% no flow, disregard all other frames

        n_flow = annotations[annotations['FLOW_majority']
                             == 1]['FLOW_majority'].count()
        n_noflow = annotations[annotations['FLOW_majority']
                               == 0]['FLOW_majority'].count()
        print('Frames with flow:', n_flow, ', frames without flow:', n_noflow)
        print('balancing to the smaller number of both...')

        grouped = annotations[~annotations['FLOW_majority'].isna()].groupby(
            'FLOW_majority')
        balanced = grouped.apply(lambda x: x.sample(
            grouped.size().min()).reset_index(drop=True))
        annotations = balanced.reset_index(drop=True)

    elif balance == 'all':
        # balance annotations such that the dataframe is 50% flow, 50% no flow,
        # 50% male, 50% female
        # 25% asian, 25% african, 25% european, 25% other
        # in other words: 1/16 of each combination of classes.
        # disregard all other frames

        grouped = annotations[~annotations['FLOW_majority'].isna()].groupby(
            ['FLOW_majority', 'GENDER', 'ETHNICITY'])
        balanced = grouped.apply(lambda x: x.sample(
            grouped.size().min()).reset_index(drop=True))
        annotations = balanced.reset_index(drop=True)

    elif balance == 'none':
        pass

    task_list = [(vid_name, annotations, folder_name) for vid_name in clip_list]

    with Pool(n_parallel) as p:
        p.starmap(extract_frames, task_list)
    cv2.destroyAllWindows()
