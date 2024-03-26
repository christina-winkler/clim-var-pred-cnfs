from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.metrics_esrgan import calculate_ssim, calculate_psnr, bgr2ycbcr
import torchvision
import numpy as np
import pdb
import torch
import torch.nn as nn

def ssim(im1, im2):
    """
    Computes the similarity index between two images measuring
    the similarity between the two images. SSIM has a maximum value of 1, indicating that the two signals are perfectly structural similar
    while a value of 0 indicates no structural similarity.

    Args:
        im1 (tensor):
        im2 (tensor):
    Returns:
        ssim (list):
    """

    im1 = im1.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()

    ssim = []

    # Compute ssim over samples in mini-batch
    for i in range(im1.shape[0]):
        ssim.append(calculate_ssim(im1[i, :, :, :] * 255, im2[i, :, :, :] * 255))
    return ssim

def psnr(im1, im2):
    """

    Args:
        im1 (tensor)
        im2 (tensor)
    Returns:
        psnr (list)
    """
    im1 = im1.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    psnr = []

    # Compute psnr over samples in mini-batch
    # pdb.set_trace()
    for i in range(im1.shape[0]):
        # psnr.append(calculate_psnr(im1[i, :, :, :] * 255, im2[i, :, :, :] * 255))
        psnr.append(calculate_psnr(im1[i, :, :, :], im2[i, :, :, :]))
    return psnr

def compute_latitude_bounds(latitude, longitude, num_lat_bins, num_lon_bins):
    """
    Compute latitude bounds for each grid cell based on latitude and longitude coordinates.

    Parameters:
    - latitude: Numpy array of latitude coordinates for each pixel.
    - longitude: Numpy array of longitude coordinates for each pixel.
    - num_lat_bins: Number of latitude bins (grid cells). i.e. height
    - num_lon_bins: Number of longitude bins (grid cells). i.e. width

    Returns:
    - latitude_bounds: List of tuples containing upper and lower latitude bounds for each grid cell.
    """

    lat_min = torch.min(latitude)
    lat_max = torch.max(latitude)
    lon_min = torch.min(longitude)
    lon_max = torch.max(longitude)

    lat_step = (lat_max - lat_min) / num_lat_bins
    lon_step = (lon_max - lon_min) / num_lon_bins

    latitude_bounds = []
    for lat_bin in range(num_lat_bins):
        lat_lower = lat_min + lat_bin * lat_step
        lat_upper = lat_min + (lat_bin + 1) * lat_step

        for lon_bin in range(num_lon_bins):
            lon_lower = lon_min + lon_bin * lon_step
            lon_upper = lon_min + (lon_bin + 1) * lon_step

            latitude_bounds.append((lat_upper, lat_lower))

    return latitude_bounds

def compute_latitude_weights(latitude, longitude):
    """
    Compute latitude weights based on the provided latitude bounds.

    Parameters:
    - latitude_bounds: List or array of tuples containing upper and lower latitude bounds for each grid cell.

    Returns:
    - weights: Numpy array of latitude weights.
    """
    num_lat_bins = latitude.shape[1]
    num_lon_bins = longitude.shape[1]
    latitude_bounds = compute_latitude_bounds(latitude, longitude, num_lat_bins, num_lon_bins)

    # init array to store latitude weights
    weights = torch.zeros(len(latitude_bounds))

    # compute sum of differences in sine of latitude bounds
    sum_sin_diff = 0 
    for i, item in enumerate(latitude_bounds):
        sum_sin_diff += torch.sin(torch.deg2rad(item[0])) - torch.sin(torch.deg2rad(item[1]))
    mean_sin_diff = (sum_sin_diff/len(latitude_bounds))

    # compute latitude weights for each grid cell 
    for i, item in enumerate(latitude_bounds):
        weight = (torch.sin(torch.deg2rad(item[0]))-torch.sin(torch.deg2rad(item[1])))/mean_sin_diff  
        weights[i] = weight
    return weights

def weighted_RMSE(yhat, y, latitude, longitude):
    weights = compute_latitude_weights(latitude, longitude)
    # pdb.set_trace()
    _,_,w,h = y.size()
    sq_diff = (weights.view(w,h) * (yhat-y)**2)
    mean = sq_diff.mean(dim=[1,2,3])
    return torch.sqrt(mean)

def RMSE(yhat,y):
    sq_diff = (yhat-y)**2
    mean = sq_diff.mean(dim=[1,2,3])
    return torch.sqrt(mean)

def MSE(y_hat, y):
    _,_,h,w=y.shape
    diff = (y_hat - y)**2
    sum = (diff).sum(dim=[1,2,3])
    return sum/(h*w)

def MAE(y_hat, y):
    out = torch.abs(y_hat - y)
    return out.mean((1,2,3))

def nrmse(im1, im2):
    """

    Args:

    Returns:

    """
    im1 = im1.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    im2 = im2.permute(0, 3, 2, 1).contiguous().cpu().detach().numpy()
    nrmse = []
    # Compute ssim over samples in mini-batch
    for i in range(im1.shape[0]):
        nrmse.append(compare_nrmse(im1[i, :, :, :], im2[i, :, :, :]))
    return np.mean(nrmse)

def MMD(x, y, kernel= lambda x, y: rbf_kernel(x, y, gamma=1.0), sqrtMMD=False):
    """
    :param x:
    :param y:
    :param gamma:
    :param sqrtMMD:
    :return:
    """
    result = kernel(x, x).mean(dim=[1,2,3]) - 2 * kernel(x, y).mean(dim=[1,2,3]) + kernel(y, y).mean(dim=[1,2,3])
    if sqrtMMD == 2:
        result = torch.sqrt(result)
    elif sqrtMMD == 3:
        result = result ** (1 / 3)
    elif sqrtMMD == 4:
        result = result ** (1 / 4)
    return result

def rbf_kernel(x, y, gamma=None):
    """Radial basis (Gaussian) kernel between x and y. Exp(-gamma*|x-y|^2).

    Input:
        x: Tensor (n_samples_x, n_features).
        y: Tensor (n_samples_y, n_features).
        gamma: Default: 1.0 / n_features. Gamma can also be a list, then the cost function is
                                evaluated overall entries of those lists for gamma realizations
    """
    gamma = gamma or (1.0 / x.shape[1])
    if not isinstance(gamma, list):
        return torch.exp(-gamma * euclidean_distances(x, y, squared=True))
    else:
        reVal = torch.zeros((x.shape[0], y.shape[0]))
        euclDist = euclidean_distances(x, y, squared=True)
        for g in gamma:
            reVal = reVal + torch.exp(-g * euclDist)
        return reVal

def euclidean_distances(x, y, squared=False):
    """Euclidean distance.

    Input:
        x: Tensor.
        y: Tensor.
        squared: Compute squared distance? Default: False.

    Returns:
        Tensor (n_samples_x, n_samples_y).
    """
    # (a-b)^2 = -2ab + a^2 + b^2
    # distances = -2 * x @ y.mT

    # pdb.set_trace()
    # distances += torch.einsum('ij,ij->i', x, x)[:, None]
    # distances += torch.einsum('ij,ij->i', y, y)[None, :]
    # distances += torch.cdist(x,y,p=2)
    distances = ((x-y)**2)

    if squared:
        return distances
    else:
        return distances.sqrt()

# def EMD(x, y):
#     """Computes EMD / Wasserstein Distance"""
#     emd = neuralnet_pytorch.metrics.emd_loss(x,y)
#     return emd
