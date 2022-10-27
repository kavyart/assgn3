import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import cv2

def normalize(I):
    return (I - np.min(I)) / (np.max(I) - np.min(I))
 

def bilateral_filter(A, F, sigma_s, sigma_r):
    A_base = np.zeros_like(A)
    lamb = 0.01
    I_intensities = A
    # I_intensities = np.sum(A, axis=-1) / 3
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
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

def gradient(I):
    d_x = np.diff(I, 1, axis=1, prepend=0)
    d_y = np.diff(I, 1, axis=0, prepend=0)
    return d_x, d_y

def divergence(I_x, I_y):
    d_x = np.diff(I_x, 1, axis=1, append=0)
    d_y = np.diff(I_y, 1, axis=0, append=0)
    return d_x + d_y

def laplacian(I):
    return signal.convolve2d(I, laplacian_kernel, mode='same', boundary='fill', fillvalue=0)



def gradient_descent_channel(D, I):
    epsilon = .000001
    N = 1400

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

    for c in range(I.shape[2]):
        D = laplacian(I[:,:,c])
        channel_gd += (gradient_descent_channel(D, I[:,:,c]),)

    I_star = np.dstack(channel_gd)
    I_star = normalize(I_star)
    return I_star
 

def orientation_coherency_map(A, F):
    A_x, A_y = gradient(A)
    F_x, F_y = gradient(F)

    numerator = np.abs((F_x * A_x) + (F_y * A_y))
    denominator = np.sqrt(np.power(F_x, 2) + np.power(F_y, 2)) * np.sqrt(np.power(A_x, 2) + np.power(A_y, 2))
    
    M = np.nan_to_num(numerator / denominator)
    return M

def saturation_weight_map(F):
    sigma = 50
    t_s = 0.7

    w_s = np.tanh(sigma * (F - t_s))
    w_s = normalize(w_s)
    return w_s


def fused_gradient_field(A, F):
    channel_gd = tuple()
    # M = orientation_coherency_map(A, F)
    # w_s = saturation_weight_map(F)

    for c in range(3):
        A_x, A_y = gradient(A[:,:,c])
        F_x, F_y = gradient(F[:,:,c])

        M = orientation_coherency_map(A[:,:,c], F[:,:,c])
        w_s = saturation_weight_map(F[:,:,c])

        F_star_x = (w_s * A_x) + ((1 - w_s) * ((M * F_x) + ((1 - M) * A_x)))
        F_star_y = (w_s * A_y) + ((1 - w_s) * ((M * F_y) + ((1 - M) * A_y)))
        
        channel_gd += (gradient_descent_channel(divergence(F_star_x, F_star_y), A[:,:,c]),)

    I_star = np.dstack(channel_gd)
    I_star = normalize(I_star)

    return I_star



# OBSERVATION DECK
def underexposed_map(F):
    sigma = 100
    t_ue = 0.5

    I = np.sum(F, axis=-1) / 3

    w_ue = 1 - np.tanh(sigma * (I - t_ue))
    w_ue = normalize(w_ue)
    return w_ue

def projection(H, A):
    adotb = np.dot(H.flatten(), A.flatten())
    bdotb = np.dot(A.flatten(), A.flatten())
    return (adotb / bdotb) * A

def observation_deck_gradient_field(A, F):
    channel_gd = tuple()

    H = A + F
    w_ue = underexposed_map(A)

    for c in range(3):
        A_x, A_y = gradient(A[:,:,c])
        H_x, H_y = gradient(H[:,:,c])

        # w_ue = underexposed_map(A[:,:,c])

        F_star_x = (w_ue * H_x) + ((1 - w_ue) * projection(H_x, A_x))
        F_star_y = (w_ue * H_y) + ((1 - w_ue) * projection(H_y, A_y))

        channel_gd += (gradient_descent_channel(divergence(F_star_x, F_star_y), A[:,:,c]),)

    I_star = np.dstack(channel_gd)
    I_star = normalize(I_star)

    return I_star



# def edges(I):
#     # I = np.sum(I, axis=-1) / 3

#     # gaussian_filter(I, sigma_s)
#     # I_uint = np.uint8(I)

#     # edges = cv2.Canny(I_uint, 1000, 1000)
#     # edges = normalize(edges)

#     blur = cv2.GaussianBlur(I, (5,5), 0.4)

#     intensity = np.sum(blur, axis=-1) / 3

#     # intensities
#     d_x, d_y = gradient(intensity)
#     gaussian = np.hypot(d_x, d_y)
#     gaussian = normalize(gaussian)

#     # edge directions
#     theta = np.arctan2(d_y, d_x)

#     return gaussian, theta



# def stylize(I):
#     edges, _ = edges(I)

#     u = I
#     e = edges




def main():
   
    print("Initializing variables...")
    N = 1
    ISO_A = 100
    ISO_F = 100

    # im_lamp_amb = io.imread('assgn3/data/lamp/lamp_ambient.JPG')[20::N, :-11:N] / 255
    # im_lamp_flash = io.imread('assgn3/data/lamp/lamp_flash.JPG')[:-20:N, 11::N] / 255

    # A = im_lamp_amb#[:,:,0:3]
    # F = im_lamp_flash#[:,:,0:3]
    # print(A.shape)
    # print(F.shape)

    # A_a = np.clip(A, 0, 1) * 255
    # Aa = A_a.astype(np.ubyte)

    # A_lin = linearize(A) * (ISO_F / ISO_A)
    # F_lin = linearize(F)
    
    # print("Finished initialization!")

    # A_base = bilateral_filter(A, A, 5, 0.05)
    # fig = plt.figure()
    # plt.imshow(A_base)
    # A1 = np.clip(A_base, 0, 1) * 255
    # A1_ubyte = A1.astype(np.ubyte)
    # io.imsave('base.png', A1_ubyte)

    # A_NR = bilateral_filter(A, F, 2, 0.20)
    # fig = plt.figure()
    # plt.imshow(A_NR)
    # A2 = np.clip(A_NR, 0, 1) * 255
    # A2_ubyte = A2.astype(np.ubyte)
    # io.imsave('NR.png', A2_ubyte)

    # F_base = bilateral_filter(F, F, 2, 0.10)
    # A_detail = detail_transfer(A_NR, F, F_base)
    # fig = plt.figure()
    # plt.imshow(A_detail)
    # A3 = np.clip(A_detail, 0, 1) * 255
    # A3_ubyte = A3.astype(np.ubyte)
    # io.imsave('detail.png', A3_ubyte)

    # A_final = mask_merging(A_base, A_detail, A_lin, F_lin)
    # fig = plt.figure()
    # plt.imshow(A_final)
    # A4 = np.clip(A_final, 0, 1) * 255
    # A4_ubyte = A4.astype(np.ubyte)
    # io.imsave('final.png', A4_ubyte)

    im_museum_amb = io.imread('assgn3/data/museum/mick_ambient.JPG')[::N, ::N] / 255
    im_museum_flash = io.imread('assgn3/data/museum/mick_flash.JPG')[::N, ::N] / 255

    # image = np.arange(100).reshape((10,10))
    # f_x, f_y = gradient(image)
    # fused1 = divergence(f_x, f_y)

    # fused2 = laplacian(image)

    # print(fused1 - fused2)


    # fig = plt.figure()
    # plt.imshow(im_museum_amb)
    # fig = plt.figure()
    # plt.imshow(im_museum_flash)


    # fused = gradient_descent(im_museum_amb)
    # f_x, f_y = gradient(im_museum_amb)
    # fused = divergence(f_x, f_y)
    # fused = fused_gradient_field(im_museum_amb, im_museum_flash)
    # fig = plt.figure()
    # plt.imshow(fused)
    # fused = fused * 255
    # fused_ubyte = fused.astype(np.ubyte)
    # io.imsave('newfused_channels.png', fused_ubyte)

    # fused = laplacian(im_museum_amb)
    # fused = fused_gradient_field(im_museum_amb, im_museum_flash)
    # fig = plt.figure()
    # plt.imshow(fused)

    # N = 1
    # im_window_amb = io.imread('assgn3/data/window/groot_ambient.JPG')[::N, ::N] / 255
    # im_window_flash = io.imread('assgn3/data/window/groot_flash.JPG')[::N, ::N] / 255

    # A = im_window_amb
    # F = im_window_flash

    # obdeck = observation_deck_gradient_field(A, F)
    # fig = plt.figure()
    # plt.imshow(obdeck)
    # obdeck = obdeck * 255
    # obdeck_ubyte = obdeck.astype(np.ubyte)
    # io.imsave('obdeck.png', obdeck_ubyte)

    # gaussian_edges, theta = edges(im_lamp_amb)
    # print(theta)
    # fig = plt.figure()
    # plt.imshow(gaussian_edges, cmap='gray')
    # fig = plt.figure()
    # plt.imshow(theta, cmap='gray')
    # gaussian_edges = gaussian_edges * 255
    # gaussian_edges_ubyte = gaussian_edges.astype(np.ubyte)
    # io.imsave('edges.png', gaussian_edges_ubyte)


    plt.show()


    

if __name__ == "__main__":
    main()