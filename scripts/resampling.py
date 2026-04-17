
import torch
import xarray
import rasterio

import loss

RESAMPLINGS = {
    'nearest' : rasterio.enums.Resampling.nearest,
    'bilinear' : rasterio.enums.Resampling.bilinear,
    'cubic' : rasterio.enums.Resampling.cubic,
    'cubic_spline' : rasterio.enums.Resampling.cubic_spline,
    'lanczos' : rasterio.enums.Resampling.lanczos,
    'average' : rasterio.enums.Resampling.average
}

LOSSES = {
    'conservation mae' : loss.ConservationLoss(16, 'mae'),
    'conservation mse' : loss.ConservationLoss(16, 'mse'),
    'simple gradient mae' : loss.SoftSimpleGradientLoss(16, 'mae'),
    'simple gradient mse' : loss.SoftSimpleGradientLoss(16, 'mse'),
    'continuity mae' : loss.SoftContinuityLoss(16, 'mae'),
    'continuity mse' : loss.SoftContinuityLoss(16, 'mse'),
    'Sobel mae' : loss.SoftSobelGradientLoss(16, 'mae'),
    'Sobel mse' : loss.SoftSobelGradientLoss(16, 'mse'),
    'G-loss mae' : loss.SoftGLoss(16, 'mae'),
    'G-loss mse' : loss.SoftGLoss(16, 'mse'),
    'direction continuity mae' : loss.SoftDirectionContinuityLoss(16, 'mae'),
    'direction continuity mse' : loss.SoftDirectionContinuityLoss(16, 'mse'),
    'Sobel direction mae' : loss.SoftSobelDirectionContinuityLoss(16, 'mae'),
    'Sobel direction mse' : loss.SoftSobelDirectionContinuityLoss(16, 'mse'),
}

class ReKIS(torch.utils.data.Dataset):

    def __init__(self, Y: xarray.DataArray, scale_factor : int, resampling):
        self.scale_factor = scale_factor

        Y = Y.isel(easting=slice(0, 400), northing=slice(0, 400))
        Y.rio.set_spatial_dims("easting", "northing")

        resolution = 1000 * self.scale_factor
        X = Y.rio.reproject(
            Y.rio.crs, resolution=(resolution, resolution), resampling=resampling
        )
        self.X = torch.from_numpy(X.values).unsqueeze(1)
        self.Y = torch.from_numpy(Y.values).unsqueeze(1)


    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]

if __name__ == "__main__":
    PATH = '/mnt/data/climate/ReKIS/KlimRefDS_v3.1_1961-2023/Raster/Tag/GK4/'
    # PATH = '/home/tomas/ctu/current/rci_data/climate/ReKIS/KlimRefDS_v3.1_1961-2023/Raster/Tag/GK4/'
    print(PATH + "TM/*.nc")
    Y_raw = xarray.open_mfdataset(PATH + "TM/*.nc", decode_coords="all")
    Y_raw = Y_raw["TM"]

    for resampling_name, resampling in RESAMPLINGS.items():
        dataset = ReKIS(Y_raw.sel(time=slice('1961', '1962')).copy(), 16, resampling=resampling)

        X, Y = dataset.X.reshape((-1, 1, 400 // 16, 400 // 16)), dataset.Y.reshape((-1, 1, 400, 400))

        print(50 * '=')
        print(resampling_name)

        X_upscaled = torch.kron(X, torch.ones(16, 16))
        print(X_upscaled.shape)

        print('RMSE:', ((X_upscaled - Y) ** 2).mean().sqrt())
        print('MAE:', (X_upscaled - Y).abs().mean())

        for loss_name, loss_fn in LOSSES.items():
            print(f'{loss_name}: {loss_fn(Y, None, X)}')
