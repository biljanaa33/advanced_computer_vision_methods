import cv2
import time
import numpy as np
import matplotlib . pyplot as plt
from ex2_utils import generate_responses_1, get_patch
from ex1_utils import gausssmooth

def get_kernel(r, kernel_type = 'epanechnikov'): 

    if kernel_type == 'epanechnikov':
        return np.ones_like(r)
    elif kernel_type == 'gaussian':
        return np.exp(-r)
   

def mean_shift_algorithm(pdf, curr_pos, kernel_type='gaussian', bandwidth=3, max_iter=1000, adaptive=False):

    initial_bw = bandwidth 
    path = [np.array(curr_pos, dtype=float)]

    for i in range(max_iter):

        if adaptive:
            curr_bw = int(max(5, round(initial_bw * (0.95 ** i))))  
        else:
            curr_bw = bandwidth

        patch_size = (curr_bw, curr_bw)
        x_vals = np.arange(-curr_bw // 2, curr_bw // 2)
        y_vals = np.arange(-curr_bw // 2, curr_bw // 2)
        x_pos, y_pos = np.meshgrid(x_vals, y_vals)

        # kernel  recomputed each time
        r_sq = (x_pos**2 + y_pos**2) / ((curr_bw // 2)**2)
        g = get_kernel(r_sq, kernel_type=kernel_type)

        w, _ = get_patch(pdf, curr_pos, patch_size)
        w = w * g
        w = w / (np.sum(w) + 1e-10)

        x_shift = np.sum(w * x_pos)
        y_shift = np.sum(w * y_pos)

        shift_vec = np.array([x_shift, y_shift])
        if np.linalg.norm(shift_vec) < 1:
            break

        curr_pos = curr_pos + shift_vec
        path.append(curr_pos.copy())

    return curr_pos, path


def plot_response(pdf, start_point, kernel_type, bandwidth=27, show_label=False, adaptive = False): 
    start = start_point
    start_cp = start_point.copy()

    if kernel_type == 'gaussian':
        c = 'm'
        label = 'Gaussian kernel'
    elif kernel_type == 'epanechnikov': 
        c = 'c'
        label = 'Epanechnikov kernel'

    elif kernel_type == 'epanechnikov_adaptive': 
        c = 'w'
        kernel_type = 'epanechnikov'
        label = 'Epanechnikov kernel-adaptive'

    mode, path = mean_shift_algorithm(pdf, start, bandwidth=bandwidth, kernel_type=kernel_type, adaptive=adaptive)
    path = np.array(path)
    # print(mode)

    plt.plot(start_cp[0], start_cp[1], c + '.', alpha=0.8, markersize=18)
    plt.plot(mode[0], mode[1], c + '.', markersize=18, alpha=0.4)

    if show_label:
        plt.plot(path[:, 0], path[:, 1], c + '.-', label=label)
    else:
        plt.plot(path[:, 0], path[:, 1], c + '.-')


plt.figure(figsize=(6,6))
pdf = generate_responses_1()
plt.imshow(pdf, cmap='hot')
plt.axis('off')

# plot Gaussian
plot_response(pdf, [10, 70], 'gaussian', show_label=True)
plot_response(pdf, [80, 50], 'gaussian', bandwidth=57)
plot_response(pdf, [10, 10], 'gaussian', bandwidth=57)
plot_response(pdf, [80, 20], 'gaussian', bandwidth=57)
plot_response(pdf, [10, 70], 'gaussian', bandwidth=57)
plot_response(pdf, [40, 30], 'gaussian', bandwidth=57)

# plot Epanechnikov
plot_response(pdf, [10, 70], 'epanechnikov', show_label=True)
plot_response(pdf, [80, 50], 'epanechnikov', bandwidth=57)
plot_response(pdf, [10, 10], 'epanechnikov', bandwidth=57)
plot_response(pdf, [80, 20], 'epanechnikov', bandwidth=57)
plot_response(pdf,[10, 70], 'epanechnikov', bandwidth=57)
plot_response(pdf, [40, 30], 'epanechnikov', bandwidth=57)

# plot Epanechnikov adaptive
plot_response(pdf, [10, 70], 'epanechnikov_adaptive', show_label=True)
plot_response(pdf, [80, 50], 'epanechnikov_adaptive', bandwidth=57, adaptive=True)
plot_response(pdf, [10, 10], 'epanechnikov_adaptive', bandwidth=57, adaptive=True)
plot_response(pdf, [80, 20], 'epanechnikov_adaptive', bandwidth=57, adaptive=True)
plot_response(pdf, [10, 70], 'epanechnikov_adaptive', bandwidth=57, adaptive=True)
plot_response(pdf, [40, 30], 'epanechnikov_adaptive', bandwidth=57, adaptive=True)


plt.legend(loc='upper right', fontsize=8)
#plt.savefig("ep_bw_57_ad.pdf", bbox_inches='tight', pad_inches=0)
plt.show()




def generate_responses_2():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[25, 25] = 1.0
    responses[75, 75] = 1.0
    responses[25, 75] = 1.0
    responses[75, 25] = 1.0
    return gausssmooth(responses, 10)


plt.figure(figsize=(8, 6))
pdf1 = generate_responses_2()
pdf1 = pdf1 / np.max(pdf1)
plt.imshow(pdf1, cmap='hot')
plt.axis('off')
plot_response(pdf1, [40 , 40], 'gaussian', show_label=True)
plot_response(pdf1,[56, 43], 'gaussian', bandwidth=27)
plot_response(pdf1,[48, 52], 'gaussian', bandwidth=27)
plot_response(pdf1,[43, 60], 'gaussian', bandwidth=27)
plot_response(pdf1,[60, 60], 'gaussian', bandwidth=27)
plot_response(pdf1,[50, 75], 'gaussian', bandwidth=27)

# plot Epanechnikov
plot_response(pdf1, [40 , 40], 'epanechnikov', show_label=True)
plot_response(pdf1,[56, 43], 'epanechnikov', bandwidth=27)
plot_response(pdf1,[48, 52], 'epanechnikov', bandwidth=27)
plot_response(pdf1,[43, 60], 'epanechnikov', bandwidth=27)
plot_response(pdf1,[60, 60], 'epanechnikov', bandwidth=27)
plot_response(pdf1,[50, 75], 'epanechnikov', bandwidth=27)

# lot contour
plt.legend(loc='upper right', fontsize=8)
#plt.savefig("ep_my_fun_bd_7.pdf", bbox_inches='tight', pad_inches=0)

plt.xlabel("x")
plt.ylabel("y")
plt.show()

