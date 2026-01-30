import minari
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import numpy
import os


save_dir = 'frames/'
os.makedirs(save_dir, exist_ok=True)

images = []

dataset_id = "PointMaze_UMazeDense-v3_rand_20000_v0"
dataset = minari.load_dataset(dataset_id)
X = []
Y = []
G = []
for episode in dataset:
    locations = episode.observations["achieved_goal"]
    goals = episode.observations["desired_goal"]
    for point in locations:
        x = point[0]
        y = point[1]
        X.append(x)
        Y.append(y)
    for goal in goals:





for i in range(0, len(X), 100):
    plt.figure(figsize=(4,4))
    plt.scatter()
    plt.scatter(X[:i+1],Y[:i+1],color='b',s=10)
    plt.xlim(-2,2)
    plt.ylim(-2,2)


    frame_path = os.path.join(save_dir, f'frame_{i}.png')
    plt.savefig(frame_path)
    plt.close()

    images.append(imageio.imread(frame_path))

gif_path = os.path.join(save_dir, 'rand.gif')
imageio.mimsave(gif_path, images, fps=5)

