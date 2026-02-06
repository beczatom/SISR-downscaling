# https://github.com/podondra/downscaling/

import lightning
import rasterio
import torch
import xarray


# TODO verify with experiments
RESAMPLING = rasterio.enums.Resampling.cubic_spline


class ReKIS(torch.utils.data.Dataset):
    def __init__(self, Y):
        Y = Y.isel(easting=slice(0, 400), northing=slice(0, 400))
        # TODO verify order of dimensions
        # spatial dimensions are set here because DataArray.sel loses them
        Y.rio.set_spatial_dims("easting", "northing")
        # TODO how is it with resolution/shape?
        X = Y.rio.reproject(
            Y.rio.crs, resolution=(2_000, 2_000), resampling=RESAMPLING
        )
        self.X = torch.from_numpy(X.values).unsqueeze(1)
        self.Y = torch.from_numpy(Y.values).unsqueeze(1)
        # TODO standardise

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class ReKISDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size, path):
        super().__init__()
        self.batch_size = batch_size
        self.path = path

    def setup(self, stage):
        variable = "TM"  # TODO
        Y = xarray.open_mfdataset(self.path + variable + "/*.nc", decode_coords="all")
        Y = Y[variable]
        self.trainset = ReKIS(Y.sel(time=slice("1961", "1965")))
        self.valset = ReKIS(Y.sel(time=slice("1966", "1967")))
        self.testset = ReKIS(Y.sel(time=slice("1968", "1968")))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=16
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, num_workers=16)
