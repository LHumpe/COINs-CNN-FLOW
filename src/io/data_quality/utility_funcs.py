import os
import pandas as pd


def get_image_info(file_path):
    data = {'video': [],
            'frame': [],
            'flow': []}
    for img in os.listdir(file_path):
        split_img = img.split('_')
        data['video'].append(int(split_img[0]))
        data['frame'].append(int(split_img[1]))
        data['flow'].append(int(split_img[3]))

    return pd.DataFrame.from_dict(data)
