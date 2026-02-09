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



class ContinuityLoss(torch.nn.Module):
    """
    Implements the continuity loss function.
    Measures the difference between the model output gradient and the target gradient.
    """
    def __init__(self):
        """
        Initializes the continuity loss.
        """
        super().__init__()

        # x derivation kernel
        x_kernel = torch.tensor([1, -1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # y derivation kernel
        y_kernel = torch.tensor([1, -1], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 2, 1))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the continuity loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """
        y_hat_x_cont = torch.conv2d(y_hat, self.x_kernel, padding=0)
        y_x_cont = torch.conv2d(y, self.x_kernel, padding=0)

        y_hat_y_cont = torch.conv2d(y_hat, self.y_kernel, padding=0)
        y_y_cont = torch.conv2d(y, self.y_kernel, padding=0)

        # MAE of x-derivation and y-derivation
        loss = (y_hat_x_cont - y_x_cont).abs().mean() + (y_hat_y_cont - y_y_cont).abs().mean()
        return loss


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
        self.continuity = ContinuityLoss()

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

        if "continuity" in self.weights.keys():
            loss += self.weights["continuity"] * self.continuity(y_hat, y)

        return loss
