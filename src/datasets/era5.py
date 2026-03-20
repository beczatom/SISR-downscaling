
import lightning
import rasterio
import torch
import xarray

# Average resampling used to reduce resolution
RESAMPLING = rasterio.enums.Resampling.average


class Era5(torch.utils.data.Dataset):
    """Dataset for ERA5."""

    def __init__(self, Y: xarray.DataArray, scale_factor : int):
        """
        Initialize the dataset with target.
        The target is upscaled and used as input for the model.
        Args:
            Y : xarray.DataArray    target variable
            scale_factor : int
        """
        self.scale_factor = scale_factor

        Y = Y.isel(latitude=slice(0, 50), longitude=slice(0, 90))

        shape = (50 // self.scale_factor, 90 // self.scale_factor)
        X = Y.rio.reproject(
            Y.rio.crs, shape=shape, resampling=RESAMPLING
        )
        self.X = torch.from_numpy(X.values).unsqueeze(1)
        self.Y = torch.from_numpy(Y.values).unsqueeze(1)


    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]


class Era5DataModule(lightning.LightningDataModule):
    """DataModule for Era5."""

    def __init__(self, batch_size: int, path: str, sets_years: list, scale_factor : int):
        """
        Initialize the DataModule.
        Args:
            batch_size : int
            path : str              path to Era5
            sets_years : list[int]  division of dataset into train, val and test sets
            scale_factor : int
        """
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.sets_years = sets_years
        self.scale_factor = scale_factor

    def setup(self, stage):
        """
        Loads, splits and saves data.
        """
        Y = xarray.open_mfdataset(self.path, decode_coords="all", decode_timedelta=False)
        Y = Y.rio.write_crs('EPSG:4326')
        Y = Y.isel(latitude=slice(0, 50), longitude=slice(0, 90))
        Y = Y.t2m.mean(dim="step") - 273.15

        self.trainset = Era5(Y.sel(time=slice(self.sets_years[0], self.sets_years[1])), self.scale_factor)
        self.valset = Era5(Y.sel(time=slice(self.sets_years[2], self.sets_years[3])), self.scale_factor)
        self.testset = Era5(Y.sel(time=slice(self.sets_years[4], self.sets_years[5])), self.scale_factor)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a DataLoader for the training set."""
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a DataLoader for the validation set."""
        return torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, num_workers=8)
