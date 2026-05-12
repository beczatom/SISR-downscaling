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


class SimpleGradientLoss(torch.nn.Module):
    """
    Implements the simple gradient loss function.
    Measures the difference between the model output gradient and the target gradient.
    We use only primitive kernels to calculate the gradient.
    """

    def __init__(self):
        """
        Initializes the simple gradient loss.
        """
        super().__init__()

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
        Calculates the simple gradient loss.
        Args:
            sr: output image
            hr: target image
            lr: unused

        Returns:
            loss
        """

        # SR and HR dx
        sr_dx = torch.conv2d(sr, self.x_kernel, padding=0)
        hr_dx = torch.conv2d(hr, self.x_kernel, padding=0)

        # SR and HR dy
        sr_dy = torch.conv2d(sr, self.y_kernel, padding=0)
        hr_dy = torch.conv2d(hr, self.y_kernel, padding=0)

        # MAE of dx and dy
        loss = (sr_dx - hr_dx).abs().mean() + (sr_dy - hr_dy).abs().mean()
        return loss


class SoftSimpleGradientLoss(torch.nn.Module):
    """
    Modifies the Simple gradient loss function to match the definition of soft constraint.
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the simple gradient loss.
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
        Calculates the simple gradient loss.
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
class SobelGradientLoss(torch.nn.Module):
    """
    Implements the Sobel gradient loss function.
    Measures the difference between the model output gradient and the target gradient.
    We use Sobel operator to calculate the gradient.
    """

    def __init__(self):
        """
        Initializes the Sobel gradient loss.
        """
        super().__init__()

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
        Calculates the Sobel gradient loss.
        Args:
            sr: downscaled image
            hr: target image
            lr: unused

        Returns:
            loss
        """

        # SR and HR dx
        sr_dx = torch.conv2d(sr, self.x_kernel, padding=0)
        hr_dx = torch.conv2d(hr, self.x_kernel, padding=0)

        # SR and HR dy
        sr_dy = torch.conv2d(sr, self.y_kernel, padding=0)
        hr_dy = torch.conv2d(hr, self.y_kernel, padding=0)

        # SR and HR gradient calculation
        hr_grad = (hr_dx.pow(2) + hr_dy.pow(2) + EPS).sqrt()
        sr_grad = (sr_dx.pow(2) + sr_dy.pow(2) + EPS).sqrt()

        # MAE of gradients
        loss = (hr_grad - sr_grad).abs().mean()
        return loss


# Zhengyang Lu, Ying Chen. 2020. Single image super-resolution based on a modified U-net with mixed gradient loss.
# Signal, Image and Video Processing (2022). https://doi.org/10.1007/s11760-021-02063-5.
class SoftSobelGradientLoss(torch.nn.Module):
    """
    Modifies the Sobel gradient loss function to match the definition of soft constraint.
    """

    def __init__(self, scale_factor: int, penalization: str = 'mean'):
        """
        Initializes the Soft Sobel gradient loss.
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
        Calculates the soft Sobel gradient loss.
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
class ContinuityLoss(torch.nn.Module):
    """
    Implements the continuity loss function.
    Measures the difference between the sum of model output gradient and the sum of target gradient.
    We use only primitive kernels to calculate the gradient.
    """

    def __init__(self):
        """
        Initializes the continuity loss.
        """
        super().__init__()

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
        Calculates the continuity loss.
        Args:
            sr: downscaled image
            hr: target image
            lr: unused

        Returns:
            loss
        """

        # SR and HR dx
        sr_dx = torch.conv2d(sr, self.x_kernel, padding=0)
        hr_dx = torch.conv2d(hr, self.x_kernel, padding=0)

        # SR and HR dy
        sr_dy = torch.conv2d(sr, self.y_kernel, padding=0)
        hr_dy = torch.conv2d(hr, self.y_kernel, padding=0)

        # AE of sums of derivatives
        loss = (sr_dx.abs().mean() + sr_dy.abs().mean() - hr_dx.abs().mean() - hr_dy.abs().mean()).abs()
        return loss


class SoftContinuityLoss(torch.nn.Module):
    """
    Modifies the Continuity loss function to match the definition of soft constraint.
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
            loss = (dilated_sr_dx.abs().mean() + dilated_sr_dy.abs().mean() - lr_dx.abs().mean() - lr_dy.abs().mean()).abs()
        elif self.penalization == 'mse':
            loss = (dilated_sr_dx.pow(2).mean() + dilated_sr_dy.pow(2).mean() - lr_dx.pow(2).mean() - lr_dy.pow(2).mean()).abs()

        return loss


# Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
# Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
class GLoss(torch.nn.Module):
    """
    Implements the G-Loss function.
    """

    def __init__(self, scale_factor: int):
        """
        Initializes the G-Loss.
        """
        super().__init__()

        self.scale_factor = scale_factor

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
            hr: target image
            lr: unused

        Returns:
            loss
        """

        # HR and SR derivatives
        hr_d = torch.conv2d(hr, self.kernel, padding=0)
        sr_d = torch.conv2d(sr, self.kernel, padding=0)

        # MAE of HR and SR derivatives
        loss = (hr_d - sr_d).abs().mean()

        # SR and HR unshuffle
        u_hr = self.unshuffle(hr)
        u_sr = self.unshuffle(sr)

        # SR and HR reshaping
        _, _, h, w = u_hr.shape
        u_hr = u_hr.reshape(-1, 1, h, w)
        u_sr = u_sr.reshape(-1, 1, h, w)

        # SR and HR unshuffled derivatives
        u_hr_d = torch.conv2d(u_hr, self.kernel, padding=0)
        u_sr_d = torch.conv2d(u_sr, self.kernel, padding=0)

        # MAE of SR and HR unshuffled derivatives
        loss += (u_hr_d - u_sr_d).abs().mean()
        return loss


# Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
# Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
class SoftGLoss(torch.nn.Module):
    """
    Modifies the G-Loss to match the definition of soft constraint.
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


if __name__ == '__main__':
    # sgl = GradientVarianceLoss(scale_factor=3)
    # y = torch.tensor([[[[0., 0., 0., 0., 0., 0.],
    #       [1., 1., 0., 1., 1., 0.],
    #       [0., 1., 1., 0., 0., 0.],
    #       [0., 1., 1., 1., 1., 0.],
    #       [0., 1., 1., 1., 0., 1.],
    #       [1., 1., 1., 0., 0., 0.]]]]).float()
    # y_hat = torch.tensor([[[[0., 1., 1., 1., 1., 1.],
    #       [0., 0., 1., 1., 1., 0.],
    #       [1., 0., 0., 1., 1., 1.],
    #       [1., 1., 1., 0., 0., 0.],
    #       [0., 1., 1., 1., 0., 0.],
    #       [0., 0., 1., 1., 0., 1.]]]]).float()
    #
    loss = SoftGLoss(3)

    x = torch.randn(2, 1, 4, 4)
    y_hat = torch.randn(2, 1, 12, 12)

    # x = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).float()
    # y_hat = torch.tensor([[[[1, 2, 3, 4, 5, 6],
    #                         [7, 1, 2, 3, 4, 5],
    #                         [6, 7, 1, 2, 3, 4],
    #                         [5, 6, 7, 1, 2, 3],
    #                         [4, 5, 6, 7, 1, 2],
    #                         [3, 4, 5, 6, 7, 1]]],
    #                       [[[1, 1, 2, 2, 3, 3],
    #                         [1, 1, 2, 2, 3, 3],
    #                         [4, 4, 5, 5, 6, 6],
    #                         [4, 4, 5, 5, 6, 6],
    #                         [7, 7, 8, 8, 9, 9],
    #                         [7, 7, 8, 8, 9, 9]]]]).float()

    print(loss(y_hat, None, x))

    print(x)
    print(y_hat)


# Abrahamyan. 2022. Gradient Variance Loss for Structure-Enhanced Image Super-Resolution.
# ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
# https://doi.org/10.1109/ICASSP43922.2022.9747387.
class GradientVarianceLoss(torch.nn.Module):
    """
    Implements the gradient variance loss function (GVL).
    Measures the difference between the variance of patches of model output gradient and target gradient.
    We use Sobel operator to calculate the gradient.
    """

    def __init__(self, scale_factor: int):
        """
        Initializes the GVL.
        """
        super().__init__()

        self.scale_factor = scale_factor

        # dx
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 3, 3))
        self.register_buffer('x_kernel', x_kernel)

        # dy
        y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 3, 3))
        self.register_buffer('y_kernel', y_kernel)

        self.unfold = torch.nn.Unfold(kernel_size=self.scale_factor, padding=0, stride=self.scale_factor)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Calculates the GVL.
        Args:
            sr: downscaled image
            hr: target image
            lr: unused

        Returns:
            loss
        """

        # added padding
        sr_p = torch.nn.functional.pad(sr, pad=(1, 1, 1, 1), mode='replicate')
        hr_p = torch.nn.functional.pad(hr, pad=(1, 1, 1, 1), mode='replicate')

        # sobel kernels
        sr_dx = torch.conv2d(sr_p, self.x_kernel)
        hr_dx = torch.conv2d(hr_p, self.x_kernel)

        sr_dy = torch.conv2d(sr_p, self.y_kernel)
        hr_dy = torch.conv2d(hr_p, self.y_kernel)

        # unfold into patches
        u_hr_dx = self.unfold(hr_dx)
        u_hr_dy = self.unfold(hr_dy)

        u_sr_dx = self.unfold(sr_dx)
        u_sr_dy = self.unfold(sr_dy)

        # calculate the variance of dx and dy in patches
        var_hr_dx = torch.var(u_hr_dx, dim=1, unbiased=True)
        var_hr_dy = torch.var(u_hr_dy, dim=1, unbiased=True)

        var_sr_dx = torch.var(u_sr_dx, dim=1, unbiased=True)
        var_sr_dy = torch.var(u_sr_dy, dim=1, unbiased=True)

        # obtain final loss
        loss = (var_sr_dx - var_hr_dx).abs().mean() + (var_sr_dy - var_hr_dy).abs().mean()
        return loss


# Inspired by : Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
# J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
# Xiong et al. used the atan2s on output image.
# But that does NOT capture the gradient orientation.
# Example: suppose we have an image [[1, 1], [1, 1]], a flat space, where gradient is 0 everywhere and the orientation
# should also be 0. But with applying atan2 on the image shifted by one to right, we get atan2(1, 1) != 0.
# Moreover, the atan2 is not the best to later measure difference between these values,
# (179 and -179 degrees are close, but the difference is large).
# So we will use cosine_similarity on gradient.
class DirectionContinuityLoss(torch.nn.Module):
    """
    Implements the direction continuity gradient loss function.
    Measures the difference between the orientation of gradient.
    """

    def __init__(self):
        """
        Initializes the orientation continuity loss function.
        """
        super().__init__()

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
        Calculates the orientation continuity gradient loss.
        Args:
            sr: downscaled image
            hr: target image
            lr: unused

        Returns:
            loss
        """

        # dx and dropping the last row to match sizes with dy
        sr_dx = torch.conv2d(sr, self.x_kernel, padding=0)
        hr_dx = torch.conv2d(hr, self.x_kernel, padding=0)
        sr_dx = sr_dx[:, :, :-1, :]
        hr_dx = hr_dx[:, :, :-1, :]

        # dy and dropping the last column to match sizes with dx
        sr_dy = torch.conv2d(sr, self.y_kernel, padding=0)
        hr_dy = torch.conv2d(hr, self.y_kernel, padding=0)
        sr_dy = sr_dy[:, :, :, :-1]
        hr_dy = hr_dy[:, :, :, :-1]

        # SR and HR gradients
        sr_grad = torch.stack([sr_dx, sr_dy], dim=1)
        hr_grad = torch.stack([hr_dx, hr_dy], dim=1)

        # cosine similarity
        sim = torch.cosine_similarity(sr_grad, hr_grad, dim=1)

        # scaling with magnitude, we don't want to punish large differences in angles when the gradient magnitude is close to zero
        loss = ((1 - sim) * (hr_dx.pow(2) + hr_dy.pow(2) + EPS).sqrt()).abs().mean()
        return loss


class SoftDirectionContinuityLoss(torch.nn.Module):
    """
    Modifies the Direction continuity loss function to match the definition of soft constraint.
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the orientation continuity loss function.
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
        Calculates the orientation continuity gradient loss.
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


class SoftSobelDirectionContinuityLoss(torch.nn.Module):
    """
    Modifies the Direction continuity loss function to match the definition of soft constraint.
    """

    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        """
        Initializes the orientation continuity loss function.
        """
        super().__init__()

        self.scale_factor = scale_factor

        self.penalization = penalization

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(scale_factor)

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
        Calculates the orientation continuity gradient loss.
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

class VarLoss(torch.nn.Module):
    def __init__(self, scale_factor: int, penalization: str = 'mae'):
        super().__init__()
        self.scale_factor = scale_factor
        self.penalization = penalization

    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        b, c, h, w = hr.shape

        div = h // self.scale_factor

        hr_unfolded = hr.unfold(2, self.scale_factor, self.scale_factor).unfold(3, self.scale_factor, self.scale_factor)
        hr_unfolded = hr_unfolded.reshape(b, div ** 2, self.scale_factor ** 2)
        hr_var = hr_unfolded.var(dim=2)
        hr_var /= torch.clamp_min(hr_var.sum(dim=0), 1e-6)

        error = (hr - sr).abs() if self.penalization == 'mae' else (hr - sr).pow(2)
        mean_error = torch.nn.AvgPool2d(self.scale_factor)(error)
        mean_error = mean_error.reshape(b, div ** 2)

        return (hr_var * mean_error).sum(dim=1).mean()





class LossCombination(torch.nn.Module):
    """
    Loss Combination is a loss function, that is a linear combination of some supported loss functions.
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
