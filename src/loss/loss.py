import torch


class ConservationLoss(torch.nn.Module):
    """
    Implements the conservation loss function.
    Measures the difference between the mean of downscaled pixel and the origin pixel.
    """

    def __init__(self, scale_factor: int):
        """
        Initializes the conservation loss.
        Notes:
            We calculate the mean of downscaled pixels by applying convolution.
            That is why we need scale_factor.
        Args:
            scale_factor: int   downscaling factor, used for kernel initialization
        """
        super().__init__()
        self.scale_factor = scale_factor

        # kernel full of ones and then divided by its size
        kernel = torch.ones((1, 1, self.scale_factor, self.scale_factor)).div(self.scale_factor ** 2)

        self.register_buffer('kernel', kernel)

    def forward(self, y_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the conservation loss.
        Args:
            y_hat:  downscaled image
            x:      origin image
        Returns:
            loss
        """
        # the output of convolution of tensor y_hat, with the above kernel,
        # will be the mean value in the scale_factor x scale_factor region
        # then this "tensor of means" has the same size as the net input x,
        # and we want them to be the same, so we penalize the difference between them
        upscaled = torch.conv2d(y_hat, self.kernel, stride=self.scale_factor, padding=0)

        # this corresponds to MAE(1/n \sum(y_hat), x)
        # TODO try MSE, like Harder et al. section 5.3
        return (upscaled - x).abs().mean()


class LossCombination(torch.nn.Module):
    """
    Loss Combination is a loss function, that is a linear combination of some supported loss functions.
    """

    def __init__(self, weights: dict, scale_factor: int):
        """
        Initializes the loss.
        Args:
            weights : dict      the items should be pairs of (type of loss, weight)
            scale_factor : int  downscaling factor, important for some losses (e.g. ConservationLoss)
        """
        super().__init__()
        weights_sum = sum(weights.values())

        if weights_sum != 1:
            raise ValueError("Weights sum must be 1")

        self.weights = weights
        self.scale_factor = scale_factor

        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.conservation = ConservationLoss(scale_factor)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the loss.
        Args:
            y_hat : torch.Tensor    downscaled image
            y : torch.Tensor        target image
            x : torch.Tensor        origin image
        Returns:
            loss
        """
        loss = torch.zeros(1, dtype=y_hat.dtype, device=y_hat.device)

        if "mae" in self.weights.keys():
            loss += self.weights["mae"] * self.mae(y_hat, y)

        if "mse" in self.weights.keys():
            loss += self.weights["mse"] * self.mse(y_hat, y)

        if "rmse" in self.weights.keys():
            loss += self.weights["rmse"] * self.mse(y_hat, y).sqrt()

        if "conservation" in self.weights.keys():
            loss += self.weights["conservation"] * self.conservation(y_hat, x)

        return loss
