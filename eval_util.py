
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def ploot(gt_data, pred_data, frame, b):
    n_agent = gt_data.shape[0]
    n_frame = gt_data.shape[1]


    fig, ax = plt.subplots()

    ln_gt = []
    ln_pred = []
    # colors = ['red', 'magenta', 'lightgreen', 'slateblue', 'blue', 'darkgreen', 'darkorange',
    #      'gray', 'purple', 'turquoise', 'midnightblue', 'olive', 'black', 'pink', 'burlywood', 'yellow']

    colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
    for i in range(n_agent):
        ln_gt.append(ax.plot([], [], colors[i] + '--')[0])
        ln_pred.append(ax.plot([], [], colors[i] + ':')[0])

    def init():
        ax.imshow(frame)
        # ax.set_xlim(-10, 15)
        # ax.set_ylim(-10, 15)

    def update_dot(num_t):
        print(num_t)
        # if (num_t < n_frame):
        for i in range(n_agent):
            ln_gt[i].set_data(gt_data[i, :num_t, 0], gt_data[i, :num_t, 1])
            ln_pred[i].set_data(pred_data[i, :num_t, 0][:num_t], pred_data[i, :num_t, 1])



    ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=100, init_func=init())

    writer = PillowWriter(fps=60)
    ani.save("eth" + str(b) + ".gif", writer=writer)
    print('---------------')
    plt.close()