import torch

EPS = 1e-8


class MSELoss(torch.nn.Module):
    """
    Classical MSE loss function.
    Defined only to be compatible with 3 attribute forward pass.
    """

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        return self.mse(sr, hr)


class L1Loss(torch.nn.Module):
    """
    Classical MAE loss function.
    Defined only to be compatible with 3 attribute forward pass.
    """

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        return self.l1(sr, hr)


def dilated_derivative(image: torch.Tensor, derivative_kernels: list[torch.Tensor] | torch.Tensor, scale_factor: int):
    """
    Calculates the dilated derivative with average resampling kernel.

    See Section 1.2.1.2

    Args:
        image: torch.Tensor - batch of images
        derivative_kernels: list[torch.Tensor | torch.Module] | torch.Tensor - for each a derivative will be appended to the result
        scale_factor: int - downscaling factor

    Returns:
        list[torch.Tensor] the dilated derivatives for each kernel
    """
    res = []
    for ker in derivative_kernels:
        derivative = torch.conv2d(input=image, weight=ker, dilation=(scale_factor, scale_factor))
        mean_derivative = torch.nn.functional.avg_pool2d(derivative, kernel_size=scale_factor, stride=scale_factor)
        res.append(mean_derivative)
    return res


class ConservationLoss(torch.nn.Module):
    """
    Implements the conservation loss function.
    Measures the difference between the mean of downscaled pixel and the origin pixel.
    See Section 1.3.2.1
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the conservation loss.
        Notes:
            We calculate the mean of downscaled pixels by applying AvgPool2d to downscaled pixels.
            That is why we need scale_factor.
        Args:
            scale_factor: int   downscaling factor, used for kernel initialization
        """
        super().__init__()
        self.scale_factor = scale_factor

        if penalization == 'mae':
            self.penalization = torch.nn.L1Loss(reduction='mean')
        else:
            self.penalization = torch.nn.MSELoss(reduction='mean')

        # will compute the means in downscaled LR pixels
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor, padding=0)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the conservation loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: low resolution image
        Returns:
            loss
        """
        # the output of convolution of tensor sr, with the average pool,
        # will be the mean value in the scale_factor x scale_factor region
        # then this "tensor of means" has the same size as the net input lr,
        # and we want them to be the same, so we penalize the difference between them
        means = self.avg_pool(sr)

        return self.penalization(means, lr)


class SoftSimpleDerivativeLoss(torch.nn.Module):
    """
    Measures the difference between the dilated model output derivatives and the LR derivatives.
    We use only primitive kernels to calculate the derivatives.
    See Section 1.3.2.2
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the simple derivative loss.
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(scale_factor)

        if penalization == 'mae':
            self.penalization = torch.nn.L1Loss(reduction='mean')
        else:
            self.penalization = torch.nn.MSELoss(reduction='mean')

        # dx
        x_kernel = torch.tensor([1, -1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # dy
        y_kernel = torch.tensor([1, -1], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 2, 1))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the simple derivative loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: low resolution image

        Returns:
            loss
        """

        # SR dilated derivatives
        sr_d = dilated_derivative(sr, [self.x_kernel, self.y_kernel], self.scale_factor)

        # LR derivatives
        lr_dx = torch.conv2d(lr, self.x_kernel, padding=0)
        lr_dy = torch.conv2d(lr, self.y_kernel, padding=0)

        # MAE/MSE of dx and dy
        loss = self.penalization(sr_d[0], lr_dx) + self.penalization(sr_d[1], lr_dy)
        return loss


# Zhengyang Lu, Ying Chen. 2020. Single image super-resolution based on a modified U-net with mixed gradient loss.
# Signal, Image and Video Processing (2022). https://doi.org/10.1007/s11760-021-02063-5.
class SoftSobelGradientMagnitudeLoss(torch.nn.Module):
    """
    Modifies the Sobel gradient magnitude loss function to match the definition of soft constraint.
    Measures the difference between the dilated model output gradient magnitude and the LR gradient magnitude.
    We use Sobel operator to calculate the gradient.
    See Section 1.3.2.3
    """

    def __init__(self, scale_factor: int, penalization: str = 'mean'):
        """
        Initializes the Soft Sobel gradient magnitude loss.
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(self.scale_factor)

        if penalization == 'mae':
            self.penalization = torch.nn.L1Loss(reduction='mean')
        else:
            self.penalization = torch.nn.MSELoss(reduction='mean')

        # dx
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 3, 3))
        self.register_buffer('x_kernel', x_kernel)

        # dy
        y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 3, 3))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the soft Sobel gradient magnitude loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: low resolution image

        Returns:
            loss
        """

        # SR derivatives
        sr_d = dilated_derivative(sr, [self.x_kernel, self.y_kernel], self.scale_factor)

        # LR derivatives
        lr_dx = torch.conv2d(lr, self.x_kernel, padding=0)
        lr_dy = torch.conv2d(lr, self.y_kernel, padding=0)

        # LR and SR gradient calculation
        lr_grad = (lr_dx.pow(2) + lr_dy.pow(2) + EPS).sqrt()
        sr_grad = (sr_d[0].pow(2) + sr_d[1].pow(2) + EPS).sqrt()

        # MAE/MSE of gradients
        return self.penalization(sr_grad, lr_grad)


# Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
# J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
class SoftContinuityLoss(torch.nn.Module):
    """
    Modifies the Continuity loss function to match the definition of soft constraint.
    Measures the difference between the sum of dilated model output derivatives and the sum of LR derivatives.
    We use only primitive kernels to calculate the derivatives.
    See Section 1.3.2.4
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the soft continuity loss.
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(self.scale_factor)

        self.penalization = penalization

        # dx
        x_kernel = torch.tensor([1, -1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # dy
        y_kernel = torch.tensor([1, -1], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 2, 1))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the soft continuity loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: target image

        Returns:
            loss
        """

        # SR derivatives
        sr_d = dilated_derivative(sr, [self.x_kernel, self.y_kernel], self.scale_factor)

        dilated_sr_dx = sr_d[0]
        dilated_sr_dy = sr_d[1]

        # LR derivatives
        lr_dx = torch.conv2d(lr, self.x_kernel, padding=0)
        lr_dy = torch.conv2d(lr, self.y_kernel, padding=0)

        # AE/SE of sums of derivatives
        loss = torch.zeros(1, dtype=sr.dtype, device=sr.device)
        if self.penalization == 'mae':
            loss = (
                        dilated_sr_dx.abs().mean() + dilated_sr_dy.abs().mean() - lr_dx.abs().mean() - lr_dy.abs().mean()).abs()
        elif self.penalization == 'mse':
            loss = (dilated_sr_dx.pow(2).mean() + dilated_sr_dy.pow(2).mean() - lr_dx.pow(2).mean() - lr_dy.pow(
                2).mean()).abs()

        return loss


# Inspired by : Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
# J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
class SoftGradientDirectionLoss(torch.nn.Module):
    """
    Modifies the Gradient Direction Loss function to match the definition of soft constraint.
    See Section 1.3.2.5
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the gradient direction loss function.
        """
        super().__init__()

        self.scale_factor = scale_factor

        self.penalization = penalization

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(scale_factor)

        # dx kernel
        x_kernel = torch.tensor([-1, 1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # dy kernel
        y_kernel = torch.tensor([1, -1], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 2, 1))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradient direction loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: low resolution image

        Returns:
            loss
        """

        sr_d = dilated_derivative(sr, [self.x_kernel, self.y_kernel], self.scale_factor)

        sr_dx = sr_d[0]
        sr_dx = sr_dx[:, :, :-1, :]

        sr_dy = sr_d[1]
        sr_dy = sr_dy[:, :, :, :-1]

        # LR derivatives
        lr_dx = torch.conv2d(lr, self.x_kernel, padding=0)
        lr_dy = torch.conv2d(lr, self.y_kernel, padding=0)

        # dropping a row, resp. column to match dy, resp. dx
        lr_dx = lr_dx[:, :, :-1, :]
        lr_dy = lr_dy[:, :, :, :-1]

        # SR and LR gradients
        sr_grad = torch.stack([sr_dx, sr_dy], dim=1)
        lr_grad = torch.stack([lr_dx, lr_dy], dim=1)

        # cosine similarity
        sim = torch.cosine_similarity(sr_grad, lr_grad, dim=1)

        # scaling with magnitude, we don't want to punish large differences in angles when the gradient magnitude is close to zero
        loss = torch.zeros(1, dtype=sr.dtype, device=sr.device)
        if self.penalization == 'mae':
            loss = ((1 - sim) * (lr_dx.pow(2) + lr_dy.pow(2) + EPS).sqrt()).abs().mean()
        elif self.penalization == 'mse':
            loss = ((1 - sim) * (lr_dx.pow(2) + lr_dy.pow(2) + EPS).sqrt()).pow(2).mean()
        return loss


class SoftSobelGradientDirectionLoss(torch.nn.Module):
    """
    Modifies the Soft Gradient Direction loss function to use Sobel kernels for the derivatives computation.
    See Section 1.3.2.5
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the Soft Sobel Gradient Direction loss function.
        """
        super().__init__()

        self.scale_factor = scale_factor

        self.penalization = penalization

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(scale_factor)

        # Sobel dx
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 3, 3))
        self.register_buffer('x_kernel', x_kernel)

        # Sobel dy
        y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 3, 3))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Soft Sobel Gradient Direction loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: low resolution image

        Returns:
            loss
        """

        sr_d = dilated_derivative(sr, [self.x_kernel, self.y_kernel], self.scale_factor)

        sr_dx = sr_d[0]
        sr_dy = sr_d[1]

        # LR derivatives
        lr_dx = torch.conv2d(lr, self.x_kernel, padding=0)
        lr_dy = torch.conv2d(lr, self.y_kernel, padding=0)

        # SR and LR gradients
        sr_grad = torch.stack([sr_dx, sr_dy], dim=1)
        lr_grad = torch.stack([lr_dx, lr_dy], dim=1)

        # cosine similarity
        sim = torch.cosine_similarity(sr_grad, lr_grad, dim=1)

        # scaling with magnitude, we don't want to punish large differences in angles when the gradient magnitude is close to zero
        loss = torch.zeros(1, dtype=sr.dtype, device=sr.device)
        if self.penalization == 'mae':
            loss = ((1 - sim) * (lr_dx.pow(2) + lr_dy.pow(2) + EPS).sqrt()).abs().mean()
        elif self.penalization == 'mse':
            loss = ((1 - sim) * (lr_dx.pow(2) + lr_dy.pow(2) + EPS).sqrt()).pow(2).mean()
        return loss


# Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
# Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
class SoftGLoss(torch.nn.Module):
    """
    Modifies the G-Loss to match the definition of soft constraint.
    See Section 1.3.2.6
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the G-Loss.
        """
        super().__init__()

        self.scale_factor = scale_factor

        if penalization == 'mae':
            self.penalization = torch.nn.L1Loss(reduction='mean')
        else:
            self.penalization = torch.nn.MSELoss(reduction='mean')

        # derivative kernels
        kernel = torch.tensor([
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
        ], dtype=torch.float32)
        kernel = kernel.unsqueeze(1)
        self.register_buffer('kernel', kernel)

        self.unshuffle = torch.nn.PixelUnshuffle(scale_factor)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the G-Loss.
        Args:
            sr: downscaled image
            hr: unused
            lr: low resolution image

        Returns:
            loss
        """

        # SR derivative
        sr_d = dilated_derivative(sr, self.kernel.unsqueeze(0), self.scale_factor)[0]

        print(sr_d.shape)

        lr_d = torch.conv2d(lr, self.kernel, padding=0)

        # MAE/MSE of derivatives
        return self.penalization(sr_d, lr_d)


class LossCombination(torch.nn.Module):
    """
    Loss Combination is a loss function, that is a weighted sum of some supported loss functions.
    """

    def __init__(self, weights: list[float], losses: list[torch.nn.Module]):
        """
        Initializes the loss.
        Args:
            weights: list[float]            the items should be the weights of losses
            losses: list[torch.nn.Module]   loss functions
        """
        super().__init__()
        weights_sum = sum(weights)

        if weights_sum != 1:
            raise ValueError("Sum of weights must be 1")

        self.weights = weights
        self.losses = torch.nn.ModuleList(losses)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the loss.
        Args:
            sr : torch.Tensor   downscaled image
            hr : torch.Tensor   target image
            lr : torch.Tensor   low resolution image
        Returns:
            loss
        """
        loss = torch.zeros(1, dtype=sr.dtype, device=sr.device)

        for i in range(len(self.weights)):
            loss += self.weights[i] * self.losses[i](sr, hr, lr)

        return loss
