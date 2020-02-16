import os
import argparse
import glob
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import pickle

DANGER_JOINTS = ['nose', 'neck', 'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']


def find_main_persons(df, thresh=0.9):
    '''Returns the indexes of the object_ids that appear in over thresh of the frames
    given pandas DF of joint_tracking.py csv format '''
    n_frames = len(set(df.iloc[:, 0]))
    vc = df.object_id.value_counts() / n_frames
    idx = vc[vc >= thresh]
    return list(idx.index.sort_values().astype(int))


def get_location_matrix(df_joints, object_ids, danger_joints):
    persons = [pd.DataFrame(y).set_index('frame_id') for x, y in df_joints.groupby('object_id', as_index=True) if
               int(x) in object_ids]
    location_matrix = np.zeros((len(object_ids), len(danger_joints), int(df_joints.frame_id.max() + 1), 2))
    for i_person, person in enumerate(persons):
        tmp_location = np.ones((int(df_joints.frame_id.max() + 1), 2)) * np.nan
        for i_joint, dj in enumerate(danger_joints):
            tmp_location[person.index.astype(int).to_list(), :] = person.loc[:, [dj + '_x', dj + '_y']].to_numpy()
            location_matrix[i_person, i_joint] = tmp_location
    return location_matrix


def get_proximity_matrix(location_matrix, frames_to_calc):
    proxmat_size = location_matrix.shape[0] * location_matrix.shape[1]
    proxmat = np.zeros((len(frames_to_calc), proxmat_size, proxmat_size))
    for i, frame_id in enumerate(frames_to_calc):
        xy_vec  = location_matrix[:, :, int(frame_id), :].reshape(-1, 2)
        xy_vec[np.isnan(xy_vec)] = 100000
        proxmat[i, :, :] = pairwise_distances(xy_vec)
    proxmat[proxmat > 10000] = np.nan
    return proxmat


def plot_proximity_matrix(proximity_matrix,is_show):
    fig = plt.figure()
    ax = sns.heatmap(proximity_matrix[0, :, :])
    ax.set_title(f'proximity matrix for {filename}')

    def init():
        plt.clf()
        ax = sns.heatmap(proximity_matrix[0, :, :], vmin=0, vmax=350)
        ax.set_title(f'proximity matrix for {filename}')
        locs = ax.get_yticks()
        labels = [DANGER_JOINTS[i % len(DANGER_JOINTS)] for i in range(len(locs))]
        ax.set_yticklabels(labels)

    def animate(i):
        plt.clf()
        data = proximity_matrix[i, :, :]
        ax = sns.heatmap(data, vmin=0, vmax=350)
        ax.set_title(f'proximity matrix for {filename}')
        locs = ax.get_yticks()
        labels = [DANGER_JOINTS[i % len(DANGER_JOINTS)] for i in range(len(locs))]
        ax.set_yticklabels(labels,rotation='horizontal')

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=2, frames=len(proximity_matrix[:, 0, 0]))

    if is_show:
        plt.show()
    return anim


def main():
    pass


def test():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='input file name')
    parser.add_argument('--data_dir', type=str, default='',
                        help='input data directory name')
    parser.add_argument('--save_dir', type=str, default='',
                        help='output data directory name')
    parser.add_argument('--person_thresh', type=int, default=0.9,
                        help='configuration score of person threshold [0.0,1.0)')
    parser.add_argument('--animate', action='store_true',
                        help='show animation')
    parser.add_argument('--dontsave', action='store_true',
                        help='save file')
    parser.add_argument('--save_animation', action='store_true',
                        help='save animation video')

    args = parser.parse_args()

    data_dir = args.data_dir
    filename = args.file
    save_dir = args.save_dir
    person_thresh = args.person_thresh

    if filename == None:
        file_paths = glob.glob(data_dir + '/*.csv')
    else:
        file_paths = [os.path.join(data_dir, filename)]
    for filepath in file_paths:
        filename = filepath.split('/')[-1]
        df_joints = pd.read_csv(filepath)
        df_joints = df_joints.rename(columns={'# frame_id': 'frame_id'})
        df_joints = df_joints.replace(-1, np.nan)
        main_persons = find_main_persons(df_joints, person_thresh)
        if len(main_persons) < 2:
            print(f'Not enough objects for detection in file {filename}')
            continue
        danger_joints = DANGER_JOINTS
        location_matrix = get_location_matrix(df_joints, main_persons, danger_joints)
        proximity_matrix = get_proximity_matrix(location_matrix, df_joints.frame_id.unique().tolist())
        print(f"File {filename} successfully analyzed")
        if not args.dontsave:
            out_filename = filename.split('.')[0] + '.pickle'
            out_filepath = os.path.join(save_dir, out_filename)
            pickle_out = open(out_filepath, "wb")
            pickle.dump(proximity_matrix, pickle_out)
            pickle_out.close()

            print(f"Saved to {out_filepath}")

        if args.animate or args.save_animation:
            anim = plot_proximity_matrix(proximity_matrix,args.animate)
            if args.save_animation:
                video_filename = filename.split('.')[0] + '.mp4'
                video_filepath = os.path.join(save_dir, video_filename)
                anim.save(video_filepath,fps=10)
