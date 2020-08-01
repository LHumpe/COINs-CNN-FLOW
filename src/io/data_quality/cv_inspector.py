from src.io.data_quality.utility_funcs import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_palette(sns.color_palette('Paired', n_colors=8))

irr_conf = 3
trash_vids = [3, 39, 48]

df = pd.read_csv('data/frames.csv').dropna()
df = df[df['majority'] >= irr_conf]
df = df.loc[~df['video'].isin(trash_vids)]
df['flow_class'] = df.FLOW_majority.astype(str)
df['flow'] = df.FLOW_majority.apply(lambda x: 'Flow' if x == 1 else 'NoFLow')
df['final_class'] = df[['flow', 'ETHNICITY', 'GENDER']].agg(' '.join, axis=1)

df_list = []
for i in range(0, 5):
    validation_path = 'data/processed/validation_{}/img'.format(i)
    training_path = 'data/processed/train_{}/img'.format(i)

    val_name = 'v_m{}'.format(i)
    train_name = 't_m{}'.format(i)

    validation = get_image_info(validation_path)
    training = get_image_info(training_path)

    exec("{} = validation.merge(df, left_on=['video', 'frame'], right_on=['video', 'frame'])".format(val_name))
    exec("{} = training.merge(df, left_on=['video', 'frame'], right_on=['video', 'frame'])".format(train_name))
    exec("{}['fold'] = {} ".format(val_name, i))
    exec("{}['fold'] = {} ".format(train_name, i))
    exec("{}['type'] = 'Validation' ".format(val_name))
    exec("{}['type'] = 'Training' ".format(train_name))

    df_list.append(eval(train_name))
    df_list.append(eval(val_name))

final_frame = pd.concat(df_list, ignore_index=True)

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(25, 17), sharex=True,
                         gridspec_kw={'top': 0.95, 'bottom': 0.15, 'left': 0.07, 'right': 0.99})
for i in range(0, 5):
    ax = axes[i]
    rel_df = final_frame.loc[final_frame['fold'] == i]
    df_plot = rel_df.sort_values(by='final_class').groupby(['final_class', 'type']).size().reset_index().pivot(
        columns='final_class', index='type', values=0)

    df_plot.plot(kind='barh', stacked=True, legend=False, ax=ax, rot=1, fontsize='large')
    ax.set_ylabel('Fold {}'.format(i + 1), fontsize='x-large')

    handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, ncol=8, loc='lower center', fontsize='large')
plt.show()
