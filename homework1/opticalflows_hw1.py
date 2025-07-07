import math
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import time

def lucaskanade(im1, im2, n = 3, improve = False, harris_thresh = 10e-9, sigma = 1):
    
    # apply check condition for normalization 
    if im1.max() > 1:     
        im1 = im1.astype(np.float32) / 255.0
        im2 = im2.astype(np.float32) / 255.0

    # apply gaussian smoothing before calculating the derrivatives     
    I_xt, I_yt = gaussderiv(img=im1, sigma=1)
    I_x1, I_y1 = gaussderiv(img=im2, sigma=1)
    I_x, I_y = 1/2*(I_xt + I_x1), 1/2*(I_yt + I_y1) 
    
    epsilon = 1e-6
    kernel_sum = np.ones((n,n), np.float32)
    
    # compture temporal derrivative 
    I_t = im2 - im1
    I_t = gausssmooth(I_t, sigma=sigma)
    
    if improve == True: 
        
         harris_response = cv2.cornerHarris(im1, blockSize=3, ksize=3, k=0.02)
         mask = harris_response > harris_thresh
         # print(harris_thresh)
         # print(harris_response.max())
         # print(harris_response.min())
         #print(mask)

    else: 
        mask = np.ones_like(I_x, dtype=bool)
        
    # calculate second order derrivatives 
    # and summing neighboring derrivatives
    I_xx_conv = cv2.filter2D(np.multiply(I_x, I_x), -1, kernel_sum)
    I_yy_conv = cv2.filter2D(np.multiply(I_y, I_y), -1, kernel_sum)
    I_xy_conv = cv2.filter2D(np.multiply(I_x, I_y), -1, kernel_sum)
    I_xt_conv = cv2.filter2D(np.multiply(I_x, I_t), -1, kernel_sum)
    I_yt_conv = cv2.filter2D(np.multiply(I_y, I_t), -1, kernel_sum)
    
    det = np.multiply(I_xx_conv, I_yy_conv) - np.multiply(I_xy_conv, I_xy_conv) + epsilon
    delta_x = (-np.multiply(I_yy_conv, I_xt_conv) + np.multiply(I_xy_conv, I_yt_conv))/ det
    delta_y = (np.multiply(I_xy_conv, I_xt_conv)-np.multiply(I_xx_conv, I_yt_conv))/ det
    delta_x[~mask] = 0
    delta_y[~mask] = 0
    # delta_x = gausssmooth(delta_x, sigma = 1)
    # delta_y = gausssmooth(delta_y, sigma = 1)   
     
    return delta_x, delta_y
       
def hornschunck(im1, im2, n_iters = 1000, lmbd = 0.5, speed_up = False, u_init = None, v_init = None, convergence = False, sigma = 1):
    
    # add normalization condtion
    # print(im1)
    if im1.max() > 1:     
        im1 = im1.astype(np.float32) / 255.0
        im2 = im2.astype(np.float32) / 255.0
        

    convg_err_x = []
    convg_err_y = []

    u_k, v_k = np.zeros(im1.shape, np.float32), np.zeros(im1.shape, np.float32)
    if speed_up: 
        u_k, v_k = u_init, v_init

    kernel_average = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]])

    
    I_xt, I_yt = gaussderiv(img=im1, sigma=sigma)
    I_x1, I_y1 = gaussderiv(img=im2, sigma=sigma)
    I_x, I_y = 1/2*(I_xt + I_x1), 1/2*(I_yt + I_y1)
    I_t = im2 - im1
    I_t = gausssmooth(I_t, sigma=sigma)
    
    for i in range(n_iters): 
        
        u_avg = cv2.filter2D(u_k, -1, kernel_average)
        v_avg = cv2.filter2D(v_k, -1, kernel_average)
        P = (I_t + np.multiply(I_x, u_avg) + np.multiply(I_y, v_avg))/(np.multiply(I_x, I_x) + np.multiply(I_y, I_y) + lmbd)
        
        u_prev = u_k
        v_prev = v_k
        # print(i)
        u_k = u_avg - np.multiply(I_x, P)
        v_k = v_avg - np.multiply(I_y, P)
        # print(np.linalg.norm(u_prev-u_k, 2))
            # print(i)
          
        if convergence: 

            convg_err_x.append(np.linalg.norm(u_prev - u_k, 2))
            convg_err_y.append(np.linalg.norm(v_prev - v_k, 2))
        
    if convergence:  
        return convg_err_x, convg_err_y
    
    return u_k, v_k

def gaussderiv(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    
    D = -2 * (x * np.exp(-x**2 / (2 * sigma**2))) / (np.sqrt(2 * math.pi) * sigma**3)
    D = D / (np.sum(np.abs(D)) / 2)
    
    Dx = cv2.sepFilter2D(img, -1, D, G)
    Dy = cv2.sepFilter2D(img, -1, G, D)

    return Dx, Dy

def gausssmooth(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)
    
def show_flow(U, V, ax, type='field', set_aspect=False):
    
    if type == 'field':
        scaling = 0.1
        u = cv2.resize(gausssmooth(U, 1.5), (0, 0), fx=scaling, fy=scaling)
        v = cv2.resize(gausssmooth(V, 1.5), (0, 0), fx=scaling, fy=scaling)
        
        x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
        y_ = -(np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
        x, y = np.meshgrid(x_, y_)
        
        ax.quiver(x, y, -u * 5, v * 5)
        if set_aspect:
            ax.set_aspect(1.)
        
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
            
    elif type == 'magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        ax.imshow(np.minimum(1, magnitude))
    elif type == 'angle':
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    elif type == 'angle_magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.minimum(1, magnitude), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    else:
        print('Error: unknown optical flow visualization type.')
        exit(-1)

def rotate_image(img, angle):
    
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

def image_pyramid(img1, levels = 5): 
    
    pyramid = [img1]
    for i in range(1, levels): 
        
        img1 = gausssmooth(img1, sigma=1)
        img1 = cv2.resize(img1, (img1.shape[1] //2, img1.shape[0] //2), interpolation=cv2.INTER_LINEAR )
        img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        pyramid.append(img1)

    return pyramid
      
def warp_image(img, flow_x, flow_y):

    h, w = img.shape[:2]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = np.arange(w) + flow_x  
    flow[:, :, 1] = np.arange(h)[:, np.newaxis] + flow_y  
    warped_img = cv2.remap(img, flow, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_img

def pyramidial_lucas_kanade(img1, img2, iterations = 1):
     
     # build image pyramid for each of the images
     img1_pyramid = image_pyramid(img1)
     img2_pyramid = image_pyramid(img2)
    
     img1, img2 = img1_pyramid[-1], img2_pyramid[-1]
     delta_xi, delta_yi = lucaskanade(img1, img2)
 
     # start from the smallest size image, last level 
     for i, (img1, img2) in enumerate(zip(img1_pyramid[::-1], img2_pyramid[::-1])):
         
         if i == 0: 
             continue
         
         new_size = (img1.shape[1], img1.shape[0])
         delta_x_up = cv2.resize(delta_xi, new_size) * 2
         delta_y_up = cv2.resize(delta_yi, new_size) * 2

         # warp the second image using the estimated 
         warped_img2 = warp_image(img2, delta_x_up, delta_y_up)
         # plt.imshow(warped_img2)
         # plt.show()
         for _ in range(iterations):
            delta_rx, delta_ry = lucaskanade(img1, warped_img2, 3)

            delta_x_up += delta_rx
            delta_y_up += delta_ry

            warped_img2 = warp_image(img2, delta_x_up, delta_y_up)

         delta_xi = delta_x_up
         delta_yi = delta_y_up

         
         # inverse warping
     return delta_xi, delta_yi
# def testing both approaches and comparing them

def test_on_synthetic_images():

    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2,-1)
    
    U_lk, V_lk = lucaskanade(im1, im2, 7)
    
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow')
    plt.savefig("lucas_kanade_synthetic.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()


    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2,-1)
    
    U_lk, V_lk = hornschunck(im1, im2, lmbd=0.5)
    
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn-Schunck Optical Flow')
    plt.savefig("horn_schunck_synthetic.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    

def test_on_different_images():
    
    
    image_1a = cv2.imread("lab2/001.jpg", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("lab2/002.jpg", cv2.IMREAD_GRAYSCALE)
    
    U_lk, V_lk = lucaskanade(image_1a, image_1b, 3, False)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow')
    plt.savefig("lucas_kanade_1.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    U_lk, V_lk = hornschunck(image_1a, image_1b)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn−Schunck Optical Flow')
    plt.savefig("horn_schunck_1.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    
    
    image_2a = cv2.imread("disparity/cporta_left.png", cv2.IMREAD_GRAYSCALE)
    image_2b = cv2.imread("disparity/cporta_right.png", cv2.IMREAD_GRAYSCALE)
    U_lk, V_lk = lucaskanade(image_2a, image_2b, 3, False)
    #U_lk, V_lk = hornschunck(image_t, image_t1)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_2a)
    ax1_12.imshow(image_2b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow')
    plt.savefig("lucas_kanade_2.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    U_lk, V_lk = hornschunck(image_2a, image_2b)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_2a)
    ax1_12.imshow(image_2b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn−Schunck Optical Flow')
    plt.savefig("horn_schunck_2.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    

    image_3a = cv2.imread("collision/00000172.jpg", cv2.IMREAD_GRAYSCALE)
    image_3b = cv2.imread("collision/00000173.jpg", cv2.IMREAD_GRAYSCALE)
    U_lk, V_lk = lucaskanade(image_3a, image_3b, 3, False)
    #U_lk, V_lk = hornschunck(image_t, image_t1)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_3a)
    ax1_12.imshow(image_3b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow')
    plt.savefig("lucas_kanade_3.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    U_lk, V_lk = hornschunck(image_3a, image_3b)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_3a)
    ax1_12.imshow(image_3b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn−Schunck Optical Flow')
    plt.savefig("horn_schunck_3.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
def lk_improvement(): 
    
    image_1a = cv2.imread("disparity/cporta_left.png", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("disparity/cporta_right.png", cv2.IMREAD_GRAYSCALE)
    
    U_lk, V_lk = lucaskanade(image_1a, image_1b, 3, False)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow')
    plt.savefig("lucas_kanade_nimp.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    U_lk, V_lk = lucaskanade(image_1a, image_1b, 3, True)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas-Kanade Optical Flow + Improvement')
    plt.savefig("lucas_kanade_imp.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
   
def methods_parameters_lk():
    
    # lucas kanade
    # Number of neighboring pixels considered
    image_1a = cv2.imread("collision/00000172.jpg", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("collision/00000173.jpg", cv2.IMREAD_GRAYSCALE)
    
    U_lk3, V_lk3 = lucaskanade(image_1a, image_1b, 3, False)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk3, V_lk3, ax1_21, type='angle')
    show_flow (U_lk3, V_lk3, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow neighboring pixles: 3x3')
    plt.savefig("lucas_kanade_3x3.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    U_lk5, V_lk5 = lucaskanade(image_1a, image_1b, 5, False)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk5, V_lk5, ax1_21, type='angle')
    show_flow (U_lk5, V_lk5, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow neighboring pixles: 5x5')
    plt.savefig("lucas_kanade_5x5.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
   
    U_lk7, V_lk7 = lucaskanade(image_1a, image_1b, 7, False)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk7, V_lk7, ax1_21, type='angle')
    show_flow (U_lk7, V_lk7, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow neighboring pixles: 7x7')
    plt.savefig("lucas_kanade_7x7.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    U_lk9, V_lk9 = lucaskanade(image_1a, image_1b, 9, False)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk9, V_lk9, ax1_21, type='angle')
    show_flow (U_lk9, V_lk9, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow neighboring pixles: 9x9')
    plt.savefig("lucas_kanade_9x9.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    fig1,(ax1_11, ax1_12,ax1_21, ax1_22) = plt.subplots(1, 4)
    show_flow(U_lk3, V_lk3, ax1_11, type='field', set_aspect=True)
    ax1_11.set_title('N: 3x3', fontsize = 10)
    show_flow(U_lk5, V_lk5, ax1_12, type='field', set_aspect=True)
    ax1_12.set_title('N: 5x5', fontsize = 10)
    show_flow(U_lk7, V_lk7, ax1_21, type='field', set_aspect=True)
    ax1_21.set_title('N: 7x7', fontsize = 10)
    show_flow(U_lk9, V_lk9, ax1_22, type='field', set_aspect=True)
    ax1_22.set_title('N: 9x9', fontsize = 10)
    plt.savefig("different_neighborhood.pdf",  bbox_inches='tight', pad_inches=0.05)
    plt.show()

    # sigma value considered for smoothing the neighborhood
    # Number of neighboring pixels considered
    image_1a = cv2.imread("collision/00000172.jpg", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("collision/00000173.jpg", cv2.IMREAD_GRAYSCALE)
    
    U_lk3, V_lk3 = lucaskanade(image_1a, image_1b, 3, False, sigma=1)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk3, V_lk3, ax1_21, type='angle')
    show_flow (U_lk3, V_lk3, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow sigma = 1')
    plt.savefig("lucas_kanade_s1.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    U_lk5, V_lk5 = lucaskanade(image_1a, image_1b, 3, False, sigma=3)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk5, V_lk5, ax1_21, type='angle')
    show_flow (U_lk5, V_lk5, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow sigma = 3')
    plt.savefig("lucas_kanade_s3.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
   
    U_lk7, V_lk7 = lucaskanade(image_1a, image_1b, 3, False, sigma=5)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk7, V_lk7, ax1_21, type='angle')
    show_flow (U_lk7, V_lk7, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow sigma = 5')
    plt.savefig("lucas_kanade_s5.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    U_lk9, V_lk9 = lucaskanade(image_1a, image_1b, 3, False, sigma= 7)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_lk9, V_lk9, ax1_21, type='angle')
    show_flow (U_lk9, V_lk9, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow sigma = 7')
    plt.savefig("lucas_kanade_s7.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
    
def test_pyramidial_lucas_kanade():

    image_t = cv2.imread("disparity/cporta_left.png", cv2.IMREAD_GRAYSCALE)
    image_t1 = cv2.imread("disparity/cporta_right.png", cv2.IMREAD_GRAYSCALE)

    U_lk, V_lk = pyramidial_lucas_kanade(image_t, image_t1)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_t)
    ax1_12.imshow(image_t1)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Pyramidial Lucas-Kanade Optical Flow')
    plt.savefig('pyramidial_lucas_kanade.pdf')
    plt.show()

    U_lk, V_lk = lucaskanade(image_t, image_t1)
    
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_t)
    ax1_12.imshow(image_t1)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow (U_lk, V_lk, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Non-Pyramidial Lucas-Kanade Optical Flow')
    plt.savefig('non_pyramidial_lucas_kanade.pdf')
    plt.show()
    
def methods_parameters_hs():

    # Number of iterations
       
    image_1a = cv2.imread("collision/00000172.jpg", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("collision/00000173.jpg", cv2.IMREAD_GRAYSCALE)
    """

    conv_err_x, conv_err_y = hornschunck(image_1a, image_1b, n_iters = 5000, convergence=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  
    plt.plot(conv_err_x,  color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Convergence in X')
    # plt.title('Convergence in X')
    plt.grid()

    plt.subplot(1, 2, 2)  
    plt.plot(conv_err_y, color='r')
    plt.xlabel('Iterations')
    plt.ylabel('Convergence in Y')
    # plt.title('Convergence in Y')
    plt.grid()

    # Show the figure
    plt.tight_layout()
    plt.savefig("convergence_error.pdf")
    plt.show()

    """
    U_hss1, V_hss1 = hornschunck(image_1a, image_1b, n_iters = 1000, sigma = 1)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hss1, V_hss1, ax1_21, type='angle')
    show_flow (U_hss1, V_hss1, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow sigma: 1')
    plt.savefig("horn_schucnk_s1.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    U_hss3, V_hss3 = hornschunck(image_1a, image_1b, n_iters = 1000, sigma = 3)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hss3, V_hss3, ax1_21, type='angle')
    show_flow (U_hss3, V_hss3, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow sigma: 3')
    plt.savefig("horn_schucnk_s3.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    U_hss5, V_hss5 = hornschunck(image_1a, image_1b, n_iters = 1000, sigma= 5)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hss5, V_hss5, ax1_21, type='angle')
    show_flow (U_hss5, V_hss5, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow sigma: 5')
    plt.savefig("horn_schucnk_s5.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    
    #### FOR THE FIGURE PLOTTED ON THE REPORT 
    U_hs1000, V_hs1000 = hornschunck(image_1a, image_1b, n_iters = 1000)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs1000, V_hs1000, ax1_21, type='angle')
    show_flow (U_hs1000, V_hs1000, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow iter: 1000')
    plt.savefig("horn_schucnk_lambda_iter_1000.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    U_hs2000, V_hs2000 = hornschunck(image_1a, image_1b, n_iters = 2000)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs1000, V_hs1000, ax1_21, type='angle')
    show_flow (U_hs1000, V_hs1000, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow iter: 5000')
    plt.savefig("horn_schucnk_lambda_iter_2000.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    fig1,(ax1_11, ax1_12) = plt.subplots(2, 1)
    show_flow(U_hs1000, V_hs1000, ax1_11, type='field', set_aspect=True)
    ax1_11.set_title('iter: 1000', fontsize = 10)
    show_flow(U_hs2000, V_hs2000, ax1_12, type='field', set_aspect=True)
    ax1_12.set_title('iter: 2000', fontsize = 10)
    plt.savefig("convergence_hs_iter.pdf",  bbox_inches='tight', pad_inches=0.05)
    plt.show()


    ############################################ LAMBDA #################################################
    """
    
    image_1a = cv2.imread("collision/00000172.jpg", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("collision/00000173.jpg", cv2.IMREAD_GRAYSCALE)
    # def hornschunck(im1, im2, n_iters = 1000, lmbd = 0.5, speed_up = False, u_init = None, v_init = None):

    # HOW DOES LAMBDA AFFECTS THE COMPUTATION
    # LAMBDA = 0.1
    
    U_hs01, V_hs01 = hornschunck(image_1a, image_1b, lmbd=0.1)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs01, V_hs01, ax1_21, type='angle')
    show_flow (U_hs01, V_hs01, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 0.1')
    plt.savefig("horn_schucnk_lambda_01.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    

    # LAMBDA = 0.3
    U_hs03, V_hs03 = hornschunck(image_1a, image_1b, lmbd=0.3)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs03, V_hs03, ax1_21, type='angle')
    show_flow (U_hs03, V_hs03, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 0.3')
    plt.savefig("horn_schucnk_lambda_03.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    # LAMBDA = 0.5
    """
    """
    U_hs05, V_hs05 = hornschunck(image_1a, image_1b,n_iters=200, lmbd=0.5)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs05, V_hs05, ax1_21, type='angle')
    show_flow (U_hs05, V_hs05, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 0.5')
    plt.savefig("horn_schucnk_lambda_05_200.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    """
    """
    # LAMBDA = 0.7
    """
    """
    U_hs07, V_hs07 = hornschunck(image_1a, image_1b,n_iters=200, lmbd=0.7)

    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs07, V_hs07, ax1_21, type='angle')
    show_flow (U_hs07, V_hs07, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 0.7')
    plt.savefig("horn_schucnk_lambda_07_200.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    """
    """
    # LAMBDA = 0.9
    """
    """
    U_hs09, V_hs09 = hornschunck(image_1a, image_1b,n_iters=200, lmbd=0.9)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs09, V_hs09, ax1_21, type='angle')
    show_flow (U_hs09, V_hs09, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 0.9')
    plt.savefig("horn_schucnk_lambda_09_200.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    """
    """
    image_1a = cv2.imread("collision/00000172.jpg", cv2.IMREAD_GRAYSCALE)
    image_1b = cv2.imread("collision/00000173.jpg", cv2.IMREAD_GRAYSCALE)
    # def hornschunck(im1, im2, n_iters = 1000, lmbd = 0.5, speed_up = False, u_init = None, v_init = None):

    # LAMBDA = 3
    
    U_hs3, V_hs3 = hornschunck(image_1a, image_1b, lmbd=3)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs3, V_hs3, ax1_21, type='angle')
    show_flow (U_hs3, V_hs3, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 3')
    plt.savefig("horn_schucnk_lambda_3.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    """
    """
    U_hs5, V_hs5 = hornschunck(image_1a, image_1b,n_iters=200, lmbd=5)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs5, V_hs5, ax1_21, type='angle')
    show_flow (U_hs5, V_hs5, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 5')
    plt.savefig("horn_schucnk_lambda_5_200.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    """
    """
    U_hs7, V_hs7 = hornschunck(image_1a, image_1b, lmbd=7)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs7, V_hs7, ax1_21, type='angle')
    show_flow (U_hs7, V_hs7, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 7')
    plt.savefig("horn_schucnk_lambda_7.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

    """
    """
    U_hs10, V_hs10 = hornschunck(image_1a, image_1b,n_iters=200, lmbd=10)
    fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(image_1a)
    ax1_12.imshow(image_1b)
    show_flow(U_hs10, V_hs10, ax1_21, type='angle')
    show_flow (U_hs10, V_hs10, ax1_22 , type='field' , set_aspect=True)
    fig1.suptitle('Horn Schunck Optical Flow lambda: 10')
    plt.savefig("horn_schucnk_lambda_10_200.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    """
    """

    fig1,(ax1_11, ax1_12,ax1_21, ax1_22) = plt.subplots(1, 4)
    show_flow(U_hs01, V_hs01, ax1_11, type='angle', set_aspect=True)
    ax1_11.set_title('λ: 0.1', fontsize = 10)
    show_flow(U_hs03, V_hs03, ax1_12, type='angle', set_aspect=True)
    ax1_12.set_title('λ: 0.3', fontsize = 10)
    show_flow(U_hs3, V_hs3, ax1_21, type='angle', set_aspect=True)
    ax1_21.set_title('λ: 3', fontsize = 10)
    show_flow(U_hs7, V_hs7, ax1_22, type='angle', set_aspect=True)
    ax1_22.set_title('λ: 7', fontsize = 10)
    plt.savefig("different_lambdas_hs.pdf",  bbox_inches='tight', pad_inches=0.05)
    plt.show()
    """


def measure_time(dataset = "lab2"):

    # lucas kanade 
    if dataset == "collision":
        prefix = "000000"
    if dataset == "lab2":
        prefix = "0"

    elapsed_time = []
    for i in range(10,41):
        
        image_t = cv2.imread(f"{dataset}/{prefix}{i}.jpg", cv2.IMREAD_GRAYSCALE)
        image_t1 = cv2.imread(f"{dataset}/{prefix}{i+1}.jpg", cv2.IMREAD_GRAYSCALE)
        # print(f"{dataset}/{prefix}{i}.png")
        time_start = time.time()
        U_lk, V_lk = lucaskanade(image_t, image_t1)
        time_end = time.time()
        elapsed_time.append(time_end-time_start)

    elapsed_time = np.array(elapsed_time)
    print(f"Average execution time on {dataset} LK: {np.mean(elapsed_time)} +- {np.std(elapsed_time)}")


     # horn schunck accelerated
    elapsed_time = []
    for i in range(10,41):
        
        image_t = cv2.imread(f"{dataset}/{prefix}{i}.jpg", cv2.IMREAD_GRAYSCALE)
        image_t1 = cv2.imread(f"{dataset}/{prefix}{i+1}.jpg", cv2.IMREAD_GRAYSCALE)
        # print(f"{dataset}/{prefix}{i}.png")
        time_start = time.time()
        U_lk, V_lk = lucaskanade(image_t, image_t1)
        U_hs, V_hs = hornschunck(image_t, image_t1, speed_up=True, u_init= U_lk, v_init=V_lk)
        time_end = time.time()
        elapsed_time.append(time_end-time_start)

    elapsed_time = np.array(elapsed_time)
    print(f"Average execution time on {dataset} HS-accelerated: {np.mean(elapsed_time)} +- {np.std(elapsed_time)}")

    # horn schunck
    elapsed_time = []
    for i in range(10,41):
        
        image_t = cv2.imread(f"{dataset}/{prefix}{i}.jpg", cv2.IMREAD_GRAYSCALE)
        image_t1 = cv2.imread(f"{dataset}/{prefix}{i+1}.jpg", cv2.IMREAD_GRAYSCALE)
        # print(f"{dataset}/{prefix}{i}.png")
        time_start = time.time()
        U_hs, V_hs = hornschunck(image_t, image_t1)
        time_end = time.time()
        elapsed_time.append(time_end-time_start)

    elapsed_time = np.array(elapsed_time)
    print(f"Average execution time on: {dataset} HS: {np.mean(elapsed_time)}  ± {np.std(elapsed_time)}")



def hs_improvement(dataset = "collision"):

    if dataset == "collision":
        prefix = "00000"
    if dataset == "lab2":
        prefix = "0"


    for i in range(177,182):
        
        image_t = cv2.imread(f"{dataset}/{prefix}{i}.jpg", cv2.IMREAD_GRAYSCALE)
        image_t1 = cv2.imread(f"{dataset}/{prefix}{i+1}.jpg", cv2.IMREAD_GRAYSCALE)

        # U_lk, V_lk = lucaskanade(image_t, image_t1)
        # U_hs, V_hs = hornschunck(image_t, image_t1, speed_up=True, u_init= U_lk, v_init=V_lk)
        U_hs, V_hs = hornschunck(image_t, image_t1)

        fig1,((ax1_11, ax1_12),(ax1_21, ax1_22)) = plt.subplots(2, 2)
        ax1_11.imshow(image_t)
        ax1_12.imshow(image_t1)
        show_flow(U_hs, V_hs, ax1_21, type='angle')
        show_flow (U_hs, V_hs, ax1_22 , type='field' , set_aspect=True)
        fig1.suptitle('Horn Schunck')
        plt.savefig(f"horn_schucnk_impr_{prefix}{i}_basic.pdf", bbox_inches='tight', pad_inches=0.05)
        plt.show()

    
if __name__ == "__main__":
    
    
    # TEST ON SYNTETIC IMAGES
    # test_on_synthetic_images()

    # TEST ON THREE IMAGES BOTH METHODS WIHOUT ADDITIONAL IMPROVEMENTS
    # test_on_different_images()

    # TEST IMRPOVED LK, FILTERING PIXELS USING HARRIS CORNER DETECTOR
    # lk_improvement()

    # TEST HS, INITIALIZED WITH LK RESULTS 
    # hs_improvement()

    # TEST METHODS PARAMETERS 
    # methods_parameters_lk()
    # methods_parameters_hs()

    # MEASURE TIME
    #  measure_time()
    
    # TEST PYRAMIDIAL LUCAS KANADE
    test_pyramidial_lucas_kanade() 
    