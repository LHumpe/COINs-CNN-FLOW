import pandas as pd
import numpy as np

df = pd.read_csv('results/yt_analysis/results.csv')
df['filename'] = df['FILE'].apply(lambda x: x.strip(']').strip('[').split('/')[-1].strip("'"))
df['vidnr'] = df.FILE.apply(lambda x: int(x.split('_')[0].split('/')[-1]))
df['FLOW_C'] = df.FILE.apply(lambda x: int(x.split('_')[-2]))

maxes = df.groupby(by='vidnr', as_index=False).agg({'PROB': np.max})
flow_pool = maxes[maxes['PROB'] > 0.71]['vidnr'].values

flow_set = df[df.vidnr.isin(flow_pool)]
no_flow_set = df[~df.vidnr.isin(flow_pool)]

flow_set = flow_set[flow_set.PROB >= 0.56]

flow_frames = list()
for vid in flow_set.vidnr.unique():
    current_frames = flow_set[flow_set.vidnr == vid]

    selected_frames = list()
    while len(selected_frames) < 3:
        selected_frames = list()
        for row in current_frames.itertuples():
            if len(selected_frames) >= 3:
                break
            flow_val = row.FLOW_C
            if flow_val == 0:
                if np.random.uniform() > 0.99:
                    selected_frames.append(row.Index)
            else:
                if row.PROB > 0.70:
                    selected_frames.append(row.Index)
                else:
                    if np.random.uniform() > 0.80:
                        selected_frames.append(row.Index)

    flow_frames.extend(selected_frames)

selected_flow = flow_set[flow_set.index.isin(flow_frames)]

no_flow_set = no_flow_set[no_flow_set.PROB < 0.47]
no_flow_frames = list()
for vid in no_flow_set.vidnr.unique():
    current_frames = no_flow_set[no_flow_set.vidnr == vid]

    selected_frames = list()
    while len(selected_frames) < 3:
        selected_frames = list()
        for row in current_frames.itertuples():
            if len(selected_frames) >= 3:
                break
            flow_val = row.FLOW_C
            if flow_val == 1:
                if np.random.uniform() > 0.8:
                    selected_frames.append(row.Index)
            else:
                if row.PROB < 0.35:
                    selected_frames.append(row.Index)
                else:
                    if np.random.uniform() > 0.6:
                        selected_frames.append(row.Index)

    no_flow_frames.extend(selected_frames)

selected_no_flow = no_flow_set[no_flow_set.index.isin(no_flow_frames)]

print('Flow: ', selected_flow.FLOW_C.value_counts(normalize=True))

print('NoFlow: ', selected_no_flow.FLOW_C.value_counts(normalize=True))

complete = pd.concat([selected_flow, selected_no_flow], axis=0)
complete['correct'] = ((complete.FLOW_C == 1) & (complete.PROB >= 0.5)) | (
        (complete.FLOW_C == 0) & (complete.PROB < 0.5))

#
drop_indices = np.array(
    [930, 2528, 170247, 204601, 2015, 218314, 60, 1703, 1098, 201, 171, 120, 4336, 282, 91, 456, 150, 323,408])
complete_sub = complete.drop(drop_indices).sample(100)

print(complete_sub.correct.sum() / len(complete_sub))
print(complete_sub.vidnr.value_counts())
print(complete_sub.FLOW_C.value_counts())
print(len(complete_sub.vidnr.unique()))

import shutil

for row in complete_sub.itertuples():
    shutil.copy(eval(row.FILE)[0], 'sel_im/{}'.format(row.filename))

c_s_copy = complete_sub.copy()

no_to_rm = [1280,169197,1102]
c_s_copy = c_s_copy.drop(no_to_rm)

flow_to_take = [122310, 191275,25098]
obs = df.loc[flow_to_take]

test = pd.concat([c_s_copy, obs], axis=0)
test['correct'] = ((test.FLOW_C == 1) & (test.PROB >= 0.5)) | (
        (test.FLOW_C == 0) & (test.PROB < 0.5))

print(test.correct.sum() / len(test))
print(test.vidnr.value_counts())
print(test.FLOW_C.value_counts())
print(len(test.vidnr.unique()))

for row in test.itertuples():
    shutil.copy(eval(row.FILE)[0], 'sel_im/{}'.format(row.filename))

final = test.sample(frac=1).reset_index(drop=True).copy()
control_flow = final[final['FLOW_C']==1].sample(25)
control_flow.correct.sum()/25

control_no_flow = final[final['FLOW_C']==0].sample(25)
control_no_flow.correct.sum()/25

final['control_flow'] = False
final['control_no_flow'] = False

control_flow['control_flow'] = True
control_flow['control_no_flow'] = False

control_no_flow['control_flow'] = False
control_no_flow['control_no_flow'] = True

final_frame = pd.concat([final, control_flow, control_no_flow], axis=0)
final_frame_shuffle = final_frame.sample(frac=1).reset_index(drop=True)

final_frame_shuffle.to_csv('Selction_Frame.csv')