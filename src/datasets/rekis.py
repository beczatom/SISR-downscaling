"""
Based on: Ondřej Podsztavek
Source: https://github.com/podondra/downscaling/
Dataset division with parameter was added.
"""

import lightning
import rasterio
import torch
import xarray

# Cubic spline resampling used to reduce resolution
RESAMPLING = rasterio.enums.Resampling.cubic_spline


class ReKIS(torch.utils.data.Dataset):
    """Dataset for ReKIS."""

    def __init__(self, Y: xarray.DataArray):
        """
        Initialize the dataset with target.
        The target is upscaled and used as input for the model.
        Args:
            Y : xarray.DataArray    target variable
        """
        Y = Y.isel(easting=slice(0, 400), northing=slice(0, 400))
        # TODO verify order of dimensions
        # spatial dimensions are set here because DataArray.sel loses them
        Y.rio.set_spatial_dims("easting", "northing")
        # TODO how is it with resolution/shape?
        X = Y.rio.reproject(
            Y.rio.crs, resolution=(10_000, 10_000), resampling=RESAMPLING
        )
        self.X = torch.from_numpy(X.values).unsqueeze(1)
        self.Y = torch.from_numpy(Y.values).unsqueeze(1)
        # TODO standardise

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]


class ReKISDataModule(lightning.LightningDataModule):
    """DataModule for ReKIS."""

    def __init__(self, batch_size: int, path: str, sets_years: list):
        """
        Initialize the DataModule.
        Args:
            batch_size : int
            path : str              path to ReKIS
            sets_years : list[int]  division of dataset into train, val and test sets
        """
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.sets_years = sets_years

    def setup(self, stage):
        """
        Loads, splits and saves data.
        """
        variable = "TM"  # TODO
        Y = xarray.open_mfdataset(self.path + variable + "/*.nc", decode_coords="all")
        Y = Y[variable]
        self.trainset = ReKIS(Y.sel(time=slice(self.sets_years[0], self.sets_years[1])))
        self.valset = ReKIS(Y.sel(time=slice(self.sets_years[2], self.sets_years[3])))
        self.testset = ReKIS(Y.sel(time=slice(self.sets_years[4], self.sets_years[5])))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a DataLoader for the training set."""
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a DataLoader for the validation set."""
        return torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, num_workers=8)
