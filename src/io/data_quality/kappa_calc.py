import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

trash_vids = [3, 39, 48]
save_kappa_csv = False

df = pd.read_csv('data/frames.csv').dropna()
ratings = df[['FLOW_JMu', 'FLOW_LHu', 'FLOW_SFr', 'video']]

bad_dict = {'vid_nr': [], 'kappa': []}

for i in ratings.video.unique():
    sample = pd.DataFrame()
    sample['flow'] = ratings.loc[ratings['video'] == i, ['FLOW_JMu', 'FLOW_LHu', 'FLOW_SFr']].sum(axis=1).astype(int)
    sample['no_flow'] = (3 - sample['flow']).astype(int)
    bad_dict['vid_nr'].append(i)
    bad_dict['kappa'].append(fleiss_kappa(sample, method='unif'))

print(pd.DataFrame.from_dict(bad_dict).sort_values('vid_nr'))

if save_kappa_csv:
    pd.DataFrame.from_dict(bad_dict).sort_values('vid_nr').to_csv('Kappas.csv', index=False)

ratings_df = df.loc[~df['video'].isin(trash_vids)]

ratings = ratings_df[['FLOW_JMu', 'FLOW_LHu', 'FLOW_SFr']]

ratings['flow'] = ratings.sum(axis=1).astype(int)
ratings['no_flow'] = (3 - ratings['flow']).astype(int)

print("Fleiss:", round(fleiss_kappa(ratings[['flow', 'no_flow']], method='fleiss') * 100, 2))
print("Randolph’s:", round(fleiss_kappa(ratings[['flow', 'no_flow']], method='unif') * 100, 2))
