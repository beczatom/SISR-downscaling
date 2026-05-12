import lightning
import rasterio
import torch
import xarray
import affine

# Average resampling used to reduce resolution
RESAMPLING = rasterio.enums.Resampling.average


class CORDEX(torch.utils.data.Dataset):
    """EURO-CORDEX dataset."""

    def __init__(self, variable, scale_factor : int):
        res = float(scale_factor * 1000)
        shape = 400 // scale_factor

        # reproject to ReKIS CRS and bounds
        X = variable.rio.reproject(
            dst_crs="EPSG:31468",
            transform=affine.Affine(res, 0.0, 4335000.0, 0.0, -res, 5955000.0),
            shape=(shape, shape),
            resampling=RESAMPLING,
        )

        # K -> deg C
        X -= 273.15
        self.X = torch.from_numpy(X.values).unsqueeze(1)

        # in Y the date of X will be stored, it will help with downscaling later
        self.Y = torch.from_numpy(X.time.values.astype('datetime64[s]').astype(int)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class CORDEXDataModule(lightning.LightningDataModule):
    """DataModule for EURO-CORDEX."""

    def __init__(self, batch_size : int, path_cordex : str, scale_factor : int):
        super().__init__()
        self.batch_size = batch_size
        self.path_cordex = path_cordex
        self.scale_factor = scale_factor

    def setup(self, stage):
        """Sets up the datamodule."""
        cordex = xarray.open_mfdataset(
            self.path_cordex
            + "/tas/tas_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_GERICS-REMO2015_v1_day_*.nc",
            decode_coords="all",
        )
        cordex = cordex["tas"]
        self.trainset = CORDEX(cordex.sel(time=slice("1979-01-02", "1992")), self.scale_factor)
        self.valset = CORDEX(cordex.sel(time=slice("1993", "2002")), self.scale_factor)
        self.testset = CORDEX(cordex.sel(time=slice("2003", "2012")), self.scale_factor)

    def train_dataloader(self):
        """Returns a DataLoader for the training set."""
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self):
        """Returns a DataLoader for the validation set."""
        return torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        """Returns a DataLoader for the test set."""
        return torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=8)
