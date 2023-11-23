from typing import Tuple, Callable
import xarray as xr
from dataclasses import dataclass
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from torchvision import transforms, datasets

class XRToTensor:
    def __call__(self, zarr_array):
        # unsqueeze to have a tensor of shape (CxWxH)
        return torch.from_numpy(zarr_array.values).unsqueeze(0)

@dataclass # TODO: needs to be adapted!
class MinMaxScaler:
    max_value: float = 315.91873
    min_value: float = 241.22385
    values_range: Tuple[int, int] = (-1, 1)

    def __call__(self, x):
        x = (x - self.min_value) / (self.max_value - self.min_value)
        return x * (self.values_range[1] - self.values_range[0]) + self.values_range[0]

@dataclass # TODO: needs to be adapted!
class InverseMinMaxScaler:
    max_value: float = 315.91873
    min_value: float = 241.22385
    values_range: Tuple[int, int] = (0, 1)

    def __call__(self, y):
        x = y * (self.max_value - self.min_value) + self.min_value
        return x

@dataclass
class WeatherBenchData(Dataset):
    """
    WeatherBench: A benchmark dataset for data-driven weather forecasting.
    Description of data: https://arxiv.org/pdf/2002.00469.pdf
    """

    data_path: str
    transform: Callable = None
    window_size: int = 2

    def __post_init__(self):

        self.data = xr.open_mfdataset(self.data_path + '*.nc', combine='by_coords')['z'] #.sel(time=slice('2016', '2016')).mean('time').load()

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),  #XRToTensor(),
                                                 MinMaxScaler(values_range=(0, 1))])
            # self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx:idx+self.window_size+1]

        # x = self.data.isel(time=idx)
        # print(x.values.min()-273.15, x.values.max()-273.15), converting min and max temp to Celsius
        time = np.array(x.coords['time'])
        latitude = np.array(x.coords['lat'])
        longitude = np.array(x.coords['lon'])

        # resize frames
        x_resh = np.zeros((x.shape[0], x.shape[1]//1, x.shape[2]//1))

        for i in range(self.window_size+1): # resize all three channels
            x_resh[i,...] = resize(x[i,...], (x.shape[1]//1, x.shape[2]//1),
                              anti_aliasing=True)

        x=x_resh
        return self.transform(x).permute(1,0,2).unsqueeze(1), str(time), latitude, longitude
