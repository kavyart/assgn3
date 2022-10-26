import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import cv2


def bilateral_filter(A, F, sigma_s, sigma_r):
    A_base = np.zeros_like(A)
    lamb = 0.01
    I_intensities = np.sum(A, axis=-1) / 3
    minI = np.min(I_intensities) - lamb
    maxI = np.max(I_intensities) + lamb
    NB_SEGMENTS = math.ceil((maxI - minI) / sigma_r)
    J = np.zeros((A.shape[0], A.shape[1], 3, NB_SEGMENTS + 1))
    i = np.zeros(NB_SEGMENTS + 1)

    (x, y) = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    channel_interpn = tuple()

    for c in range(A.shape[2]):
        for j in range(NB_SEGMENTS + 1):
        
            i_j = minI + j * ((maxI - minI) / NB_SEGMENTS)
            i[j] = i_j

            G_j = (1 / (sigma_r * np.sqrt(2 * np.pi))) * np.exp((-1/2) * np.power(((F[:, :, c] - i_j) / sigma_r), 2))
            K_j = gaussian_filter(G_j, sigma_s)
            H_j = G_j * A[:,:,c]
            H_star_j = gaussian_filter(H_j, sigma_s)
            J_j = H_star_j / K_j
            J[:,:,c,j] = J_j
        
        coords = np.dstack([y, x, A[:,:,c]])
        points = (np.arange(A.shape[0]), np.arange(A.shape[1]), i)
        channel_interpn += (interpolate.interpn(points, J[:,:,c,:], coords),)

    A_base = np.dstack(channel_interpn)
    return A_base


def detail_transfer(A_NR, F, F_base):
    epsilon = 0.01

    A_detail = A_NR * ((F + epsilon) / (F_base + epsilon))

    return A_detail

def linearize(C):
    C_linear = np.where(C <= 0.0404482, C / 12.92, ((C + 0.055) / 1.055) ** 2.4)
    return C_linear


def mask_merging(A_base, A_detail, A_lin, F_lin):
    t_shad = 0.1
    difference = F_lin - A_lin
    M = np.where(difference <= t_shad, 1, 0)

    M_detail = (1 - M) * A_detail
    M_base = M * A_base

    A_final = M_detail + M_base

    return A_final




# GRADIENT DOMAIN PROCESSING
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])
def gradient(I):
    d_x = cv2.Sobel(I, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    d_y = cv2.Sobel(I, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return d_x, d_y

def divergence(I_x, I_y):
    d_x = np.diff(I_x, 1, axis=1)
    d_y = np.diff(I_y, 1, axis=0)
    return d_x + d_y

def laplacian(I):
    return signal.convolve2d(I, kernel, mode='same', boundary='fill', fillvalue=0)



def gradient_descent_channel(D, I):
    B = np.ones_like(I)
    B[0,:] = 0
    B[:,0] = 0
    B[I.shape[0]-1,:] = 0
    B[:,I.shape[1]-1] = 0

    I_star_init = np.zeros_like(I)

    I_star_boundary = np.zeros_like(I)
    I_star_boundary[0,:] = I[0,:]
    I_star_boundary[:,0] = I[:,0]
    I_star_boundary[I.shape[0]-1,:] = I[I.shape[0]-1,:]
    I_star_boundary[:,I.shape[1]-1] = I[:,I.shape[1]-1]

    epsilon = .01
    N = 300

    # Initialization
    I_star = (B * I_star_init) + ((1 - B) * I_star_boundary)
    r = B * (D - laplacian(I_star))
    d = r
    delta_new = np.linalg.norm(r) ** 2
    n = 0

    # Conjugate gradient descent iteration
    while np.sqrt(delta_new) > epsilon and n < N:
        q = laplacian(d)
        eta = delta_new / np.dot(np.ndarray.flatten(d), np.ndarray.flatten(q))
        I_star = I_star + (B * (eta * d))
        r = B * (r - (eta * q))
        delta_old = delta_new
        delta_new = np.linalg.norm(r) ** 2
        beta = delta_new / delta_old
        d = r + (beta * d)
        n = n + 1

    return I_star

def gradient_descent(I):
    channel_gd = tuple()
    print(I)

    for c in range(I.shape[2]):
        D = laplacian(I[:,:,c])
        channel_gd += (gradient_descent_channel(D, I[:,:,c]),)

    I_star = np.dstack(channel_gd)
    I_star = (I_star - np.min(I_star)) / (np.max(I_star) - np.min(I_star))
    print(np.min(I_star))
    print(np.max(I_star))
    return I_star
 

def orientation_coherency_map(A, F):
    A_x, A_y = gradient(A)
    F_x, F_y = gradient(F)

    numerator = np.abs((F_x * A_x) + (F_y * A_y))
    denominator = np.sqrt(np.power(F_x, 2) + np.power(F_y, 2)) * np.sqrt(np.power(A_x, 2) + np.power(A_y, 2))
    
    M = np.nan_to_num(numerator / denominator)
    return M

def saturation_weight_map(F):
    sigma = 40
    t_s = 0.6

    w_s = np.tanh(sigma * (F - t_s))
    w_s = (w_s - np.min(w_s)) / (np.max(w_s) - np.min(w_s))
    return w_s

def fused_gradient_field(A, F):
    channel_gd = tuple()

    for c in range(3):
        A_x, A_y = gradient(A[:,:,c])
        F_x, F_y = gradient(F[:,:,c])

        M = orientation_coherency_map(A[:,:,c], F[:,:,c])
        w_s = saturation_weight_map(F[:,:,c])

        F_star_x = (w_s * A_x) + ((1 - w_s) * ((M * F_x) + ((1 - M) * A_x)))
        F_star_y = (w_s * A_y) + ((1 - w_s) * ((M * F_y) + ((1 - M) * A_y)))

        F_star_xx = cv2.Sobel(F_star_x, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        F_star_yy = cv2.Sobel(F_star_y, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        
        channel_gd += (gradient_descent_channel(F_star_xx + F_star_yy, A[:,:,c]),)

    I_star = np.dstack(channel_gd)
    I_star = (I_star - np.min(I_star)) / (np.max(I_star) - np.min(I_star))

    return I_star


    


def main():
   
    print("Initializing variables...")
    N = 1
    ISO_A = 1600
    ISO_F = 200

    im_lamp_amb = io.imread('assgn3/data/lamp/lamp_ambient.tif')[::N, ::N] / 255
    im_lamp_flash = io.imread('assgn3/data/lamp/lamp_flash.tif')[::N, ::N] / 255

    A = im_lamp_amb
    F = im_lamp_flash


    A_lin = linearize(A) * (ISO_F / ISO_A)
    F_lin = linearize(F)

    A_a = np.clip(A, 0, 1) * 255
    Aa = A_a.astype(np.ubyte)
    
    print("Finished initialization!")

    A_base = bilateral_filter(A, A, 5, 0.05)

    A_NR = bilateral_filter(A, F, 1, 0.15)

    F_base = bilateral_filter(F, F, 2, 0.05)
    A_detail = detail_transfer(A_NR, F, F_base)

    A_final = mask_merging(A_base, A_detail, A_lin, F_lin)

    im_museum_amb = io.imread('assgn3/data/museum/museum_ambient.png')[::N, ::N] / 255
    im_museum_flash = io.imread('assgn3/data/museum/museum_flash.png')[::N, ::N] / 255


    grad = gradient_descent(im_museum_amb)
    fused = fused_gradient_field(im_museum_amb, im_museum_flash)


 


    

if __name__ == "__main__":
    main()