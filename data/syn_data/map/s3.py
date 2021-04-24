import scipy.io as spio
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


s_name = 's3'
orig_syn_data_path = 'D:\crowd\datasets\dataset_X/'
new_syn_data_path = '../../../datasets/syn_x/'
file_name = 'bottleneck-squeeze_005.mat'
mat_adata = spio.loadmat(os.path.join(orig_syn_data_path, file_name))
trajs  = mat_adata['x_opt']
obs_poly = mat_adata['obs']
num_poly = obs_poly.shape[0]
x_limits = mat_adata['xlims'][0]
y_limits = mat_adata['zlims'][0]

time_rng = range(2, len(trajs[0]))


#### load obstacles
obs_vertices = []
for i in range(num_poly):
   obstacle = obs_poly[i][0]
   vertices_obstacle_x = obstacle[0]
   vertices_obstacle_y = obstacle[1]
   obs_vertices.append([[vertices_obstacle_x[0], vertices_obstacle_y[0]],
                             [vertices_obstacle_x[1], vertices_obstacle_y[1]],
                             [vertices_obstacle_x[2], vertices_obstacle_y[2]],
                             [vertices_obstacle_x[3], vertices_obstacle_y[3]]])
   plt.scatter(vertices_obstacle_x[0], vertices_obstacle_y[0], c='black', s=5)
   plt.scatter(vertices_obstacle_x[1], vertices_obstacle_y[1], c='black', s=5)
   plt.scatter(vertices_obstacle_x[2], vertices_obstacle_y[2], c='black', s=5)
   plt.scatter(vertices_obstacle_x[3], vertices_obstacle_y[3], c='black', s=5)



## draw map for s2
#up
plt.plot(np.linspace(-11,20), np.linspace(2.0999999046325684, 2.0999999046325684), c='black', linewidth=0.5)
plt.plot(np.linspace(-11,20), np.linspace(-2.0999999046325684, -2.0999999046325684), c='black', linewidth=0.5)
plt.plot(np.linspace(-11,20), np.linspace(100, 100), c='black', linewidth=0.5)
plt.plot(np.linspace(-11,20), np.linspace(-100, -100), c='black', linewidth=0.5)
# exit
plt.plot(np.linspace(-11, -11), np.linspace(2.0999999046325684, 100), c='black', linewidth=0.5)
plt.plot(np.linspace(20, 20), np.linspace(2.0999999046325684, 100), c='black', linewidth=0.5)
plt.plot(np.linspace(-11, -11), np.linspace(-2.0999999046325684, -100), c='black', linewidth=0.5)
plt.plot(np.linspace(20, 20), np.linspace(-2.0999999046325684, -100), c='black', linewidth=0.5)

plt.scatter(-150, 0, c='w', s=1)
plt.scatter(150, 0, c='w', s=1)
plt.scatter(0, 150, c='w', s=1)
plt.scatter(0, -150, c='w', s=1)

plt.axis('off')






colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
for a in range(8):
    for t in time_rng:
        current_position = trajs[a][t].reshape((2,))
        plt.scatter(current_position[0], current_position[1], c=colors[a], s=1)



#### homography
pts_img = np.array([
[237, 299], [237, 359],
[243, 299],  [243, 359],
[95, 299], [95, 359],
[385, 299], [385, 359],
])


pts_wrd = np.array([
[-11, 2.0999999046325684], [20, 2.0999999046325684],
[-11, -2.0999999046325684], [20, -2.0999999046325684],
[-11, 100], [20, 100],
[-11, -100], [20, -100],
])


h, status = cv2.findHomography(pts_img, pts_wrd)
inv_h_t = np.linalg.pinv(np.transpose(h))


with open(os.path.join('D:\crowd\datasets/syn_x/map', s_name + '_H.txt'), 'w') as f:
    for elt in h:
        line = '\t'.join([str(e) for e in elt])
        f.write(line + '\n')


### img process
c = imageio.imread(os.path.join('D:\crowd\datasets/syn_x', s_name + '.png'))
plt.imshow(c)

c = c[:,:,0]
c = 255-c

idx = np.where((c > 0) & (c < 255))
c[idx] = 255
cv2.imwrite(os.path.join('D:\crowd\datasets/syn_x', s_name + '_map.png'), c)

### make unnavi. as 1
c = imageio.imread(os.path.join('D:\crowd\datasets/syn_x', s_name + '_map.png'))
c[95:237, 299:359] = 255
c[243:385, 299:359] = 255


cv2.imwrite(os.path.join('D:\crowd\datasets/syn_x/map', s_name + '_map.png'), c)

### validate
for a in range(len(pts_wrd)):
    target_pos = np.expand_dims(np.transpose(pts_wrd[a]), 0)
    target_pixel = np.matmul(np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1), inv_h_t)
    target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    target_pixel = target_pixel[:, :2]
    # plt.scatter(target_pixel[0][1], target_pixel[0][0], c=colors[a], s=1)
    plt.scatter(target_pixel[0][1], target_pixel[0][0], c='r', s=1)

plt.show()



colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
for a in range(8):
    for t in time_rng:
        target_pos = np.transpose(trajs[a][t])
        target_pixel = np.matmul(np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1), inv_h_t)
        target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
        target_pixel = target_pixel[:, :2]
        plt.scatter(target_pixel[0][1], target_pixel[0][0], c=colors[a], s=1)
        # plt.scatter(target_pixel[0][1], target_pixel[0][0], c='b', s=1)



