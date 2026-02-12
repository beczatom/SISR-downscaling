import torch


class ConservationLoss(torch.nn.Module):
    # TODO AveragePool?
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
        Calculates the simple gradient loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """
        y_hat_dx = torch.conv2d(y_hat, self.x_kernel, padding=0)
        y_dx = torch.conv2d(y, self.x_kernel, padding=0)

        y_hat_dy = torch.conv2d(y_hat, self.y_kernel, padding=0)
        y_dy = torch.conv2d(y, self.y_kernel, padding=0)

        # MAE of x-derivation and y-derivation
        loss = (y_hat_dx - y_dx).abs().mean() + (y_hat_dy - y_dy).abs().mean()
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
        Initializes the simple gradient loss.
        """
        super().__init__()

        # x derivation kernel
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 3, 3))
        self.register_buffer('x_kernel', x_kernel)

        # y derivation kernel
        y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 3, 3))
        self.register_buffer('y_kernel', y_kernel)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the simple gradient loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """

        y_hat_dx = torch.conv2d(y_hat, self.x_kernel, padding=0)
        y_dx = torch.conv2d(y, self.x_kernel, padding=0)

        y_hat_dy = torch.conv2d(y_hat, self.y_kernel, padding=0)
        y_dy = torch.conv2d(y, self.y_kernel, padding=0)

        G = (y_dx.pow(2) + y_dy.pow(2)).sqrt()
        G_hat = (y_hat_dx.pow(2) + y_hat_dy.pow(2)).sqrt()

        # MAE from gradients
        # TODO try MSE
        loss = (G - G_hat).abs().mean()
        return loss


# Xiong, M. Q., 2025: Impact of physical constraints on deep learning-based downscaling prediction of temperature.
# J. Meteor. Res., 39(4), 904–919, https://doi.org/10.1007/s13351-025-4061-1.
class ContinuityLoss(torch.nn.Module):
    """
    Implements the continuity gradient loss function.
    Measures the difference between the sum of model output gradient and the sum of target gradient.
    We use only primitive kernels to calculate the gradient.
    """
    def __init__(self):
        """
        Initializes the simple gradient loss.
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
        Calculates the continuity gradient loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """
        y_hat_dx = torch.conv2d(y_hat, self.x_kernel, padding=0)
        y_dx = torch.conv2d(y, self.x_kernel, padding=0)

        y_hat_dy = torch.conv2d(y_hat, self.y_kernel, padding=0)
        y_dy = torch.conv2d(y, self.y_kernel, padding=0)

        # AE of sums of gradients
        loss = (y_hat_dx.abs().mean() + y_hat_dy.abs().mean() - y_dx.abs().mean() - y_dy.abs().mean()).abs()
        return loss


# Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
# Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
class GLoss(torch.nn.Module):
    """
    Implements the sum gradient loss function.
    Measures the difference between the sum of model output gradient and the sum of target gradient.
    We use only primitive kernels to calculate the gradient.
    """
    def __init__(self, scale_factor: int):
        """
        Initializes the G-Loss.
        """
        super().__init__()

        self.scale_factor = scale_factor

        # derivation kernels
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


    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the G-Loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """
        D_y = torch.conv2d(y, self.kernel, padding = 0)
        D_y_hat = torch.conv2d(y_hat, self.kernel, padding = 0)

        # AE of sums of gradients
        loss = (D_y - D_y_hat).abs().mean()

        u_y = self.unshuffle(y)
        u_y_hat = self.unshuffle(y_hat)

        _, _, h, w = u_y.shape

        u_y = u_y.view(-1, 1, h, w)
        u_y_hat = u_y_hat.view(-1, 1, h, w)

        D_u_y = torch.conv2d(u_y, self.kernel, padding = 0)
        D_u_y_hat = torch.conv2d(u_y_hat, self.kernel, padding = 0)

        loss += (D_u_y - D_u_y_hat).abs().mean()

        return loss


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

        # x derivation kernel
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 3, 3))
        self.register_buffer('x_kernel', x_kernel)

        # y derivation kernel
        y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 3, 3))
        self.register_buffer('y_kernel', y_kernel)

        self.unfold = torch.nn.Unfold(kernel_size=self.scale_factor, padding=0, stride=self.scale_factor)


    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the GVL.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """

        # added padding
        y_hat_p = torch.nn.functional.pad(y_hat, pad=(1, 1, 1, 1), mode='replicate')
        y_p = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode='replicate')

        # sobel kernels
        y_hat_dx = torch.conv2d(y_hat_p, self.x_kernel)
        y_dx = torch.conv2d(y_p, self.x_kernel)

        y_hat_dy = torch.conv2d(y_hat_p, self.y_kernel)
        y_dy = torch.conv2d(y_p, self.y_kernel)


        # unfold into patches
        G_x = self.unfold(y_dx)
        G_y = self.unfold(y_dy)

        G_hat_x = self.unfold(y_hat_dx)
        G_hat_y = self.unfold(y_hat_dy)

        # calculate variance in patches
        v_x = torch.var(G_x, dim = 1, unbiased=True)
        v_y = torch.var(G_y, dim = 1, unbiased=True)

        v_hat_x = torch.var(G_hat_x, dim = 1, unbiased=True)
        v_hat_y = torch.var(G_hat_y, dim = 1, unbiased=True)

        # obtain final loss
        loss = (v_hat_x - v_x).abs().mean() + (v_hat_y - v_y).abs().mean()
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

        # x derivation kernel
        x_kernel = torch.tensor([-1, 1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # y derivation kernel
        y_kernel = torch.tensor([1, -1], dtype=torch.float32)
        y_kernel = torch.reshape(y_kernel, (1, 1, 2, 1))
        self.register_buffer('y_kernel', y_kernel)


    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the orientation continuity gradient loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """

        # x derivation and dropping the last row to match sizes with y derivation
        y_hat_dx = torch.conv2d(y_hat, self.x_kernel, padding=0)
        y_dx = torch.conv2d(y, self.x_kernel, padding=0)
        y_hat_dx = y_hat_dx[:, :, :-1, :]
        y_dx = y_dx[:, :, :-1, :]

        # y derivation and dropping the last column to match sizes with x derivation
        y_hat_dy = torch.conv2d(y_hat, self.y_kernel, padding=0)
        y_dy = torch.conv2d(y, self.y_kernel, padding=0)
        y_hat_dy = y_hat_dy[:, :, :, :-1]
        y_dy = y_dy[:, :, :, :-1]

        # cosine similarity
        g_y_hat_xy = torch.stack([y_hat_dx, y_hat_dy], dim=1)
        g_y_xy = torch.stack([y_dx, y_dy], dim=1)
        sim = torch.cosine_similarity(g_y_hat_xy, g_y_xy, dim=1)

        loss = (1 - sim).abs().mean()
        return loss


# if __name__ == '__main__':
#     # sgl = GradientVarianceLoss(scale_factor=3)
#     # y = torch.tensor([[[[0., 0., 0., 0., 0., 0.],
#     #       [1., 1., 0., 1., 1., 0.],
#     #       [0., 1., 1., 0., 0., 0.],
#     #       [0., 1., 1., 1., 1., 0.],
#     #       [0., 1., 1., 1., 0., 1.],
#     #       [1., 1., 1., 0., 0., 0.]]]]).float()
#     # y_hat = torch.tensor([[[[0., 1., 1., 1., 1., 1.],
#     #       [0., 0., 1., 1., 1., 0.],
#     #       [1., 0., 0., 1., 1., 1.],
#     #       [1., 1., 1., 0., 0., 0.],
#     #       [0., 1., 1., 1., 0., 0.],
#     #       [0., 0., 1., 1., 0., 1.]]]]).float()
#
#     dcloss = DirectionContinuityLoss()
#
#     y = torch.tensor([[[[2, 4, 6], [8, 10, 12], [14, 16, 18]]], [[[2, 4, 6], [8, 10, 12], [14, 16, 18]]]]).float()
#     y_hat = torch.tensor([[[[2, 4, 5], [8, 10, 12], [14, 16, 18]]], [[[2, 4, 6], [8, 10, 12], [14, 16, 18]]]]).float()
#
#     print(dcloss(y, y_hat))
#     # print(y)
#     # print(y_hat)
#     # loss = sgl.forward(y_hat, y)


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
        self.simple_gradient = SimpleGradientLoss()
        self.continuity = ContinuityLoss()
        self.sobel_gradient = SobelGradientLoss()
        self.gloss = GLoss(scale_factor)
        self.gvl = GradientVarianceLoss(scale_factor)
        self.dcl = DirectionContinuityLoss()

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

        if "simple_gradient" in self.weights.keys():
            loss += self.weights["simple_gradient_loss"] * self.simple_gradient(y_hat, y)

        if "continuity" in self.weights.keys():
            loss += self.weights["continuity"] * self.continuity(y_hat, y)

        if "sobel_gradient" in self.weights.keys():
            loss += self.weights["sobel_gradient"] * self.sobel_gradient(y_hat, y)

        if "gloss" in self.weights.keys():
            loss += self.weights["gloss"] * self.gloss(y_hat, y)

        if "gvl" in self.weights.keys():
            loss += self.weights["gvl"] * self.gvl(y_hat, y)

        if "dcl" in self.weights.keys():
            loss += self.weights["dcl"] * self.dcl(y_hat, y)

        return loss
