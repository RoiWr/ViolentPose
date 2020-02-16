import glob
import os
import pandas as pd

# define the dataframe 'df'
LABELS_FILEPATH = r'C:\Users\Roi\Documents\ITC\PoseProject\in\clips\labels.csv'

df = pd.read_csv(LABELS_FILEPATH, header=None)
df.columns = ['category', 'video', 'set', 'start', 'end', 'label']

# helper functions
def compose_vid_name(row):
    name = '_'.join(['v', row['video'], str(int(row['start'])), str(int(row['end']))])
    return name

def compose_folder_list(folder, filetype):
    filelist = [os.path.basename(f) for f in glob.glob(folder + r'/*.' + filetype)]
    return ','.join(filelist)


# main function
def check_if_in_folder(df, category, data_folder, filetype):
    ''' given labels df, category name, data folder such proximity, tracks, etc.
    and filetype such as 'pkl', 'csv', 'mp4 , etc.' '''
    subset = df.loc[df.category == category, :]
    names = subset.apply(compose_vid_name, axis=1)
    filelist = compose_folder_list(data_folder, filetype)
    good_vids = []
    labels = []
    for name, label in zip(names, subset.label.values):
        if name in filelist:
            good_vids.append(name)
            labels.append(label)
    return good_vids, labels


# test
vids, lbs = check_if_in_folder(df, 'Doing karate', 'C:\\Users\\Roi\\Documents\\ITC\\PoseProject\\out\\tracks',
                                   'csv')