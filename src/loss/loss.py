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
            We calculate the mean of downscaled pixels by applying AvgPool2d to downscaled pixels.
            That is why we need scale_factor.
        Args:
            scale_factor: int   downscaling factor, used for kernel initialization
        """
        super().__init__()
        self.scale_factor = scale_factor

        # will compute the means in downscaled LR pixels
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor, padding=0)


    def forward(self, y_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the conservation loss.
        Args:
            y_hat:  downscaled image
            x:      origin image
        Returns:
            loss
        """
        # the output of convolution of tensor y_hat, with the average pool,
        # will be the mean value in the scale_factor x scale_factor region
        # then this "tensor of means" has the same size as the net input x,
        # and we want them to be the same, so we penalize the difference between them
        means = self.avg_pool(y_hat)

        # this corresponds to MAE(1/n \sum(y_hat), x)
        # TODO try MSE, like Harder et al. section 5.3
        return (means - x).abs().mean()



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

        # MAE of dx and dy
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

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Sobel gradient loss.
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

        eps = 1e-8
        G = (y_dx.pow(2) + y_dy.pow(2) + eps).sqrt()
        G_hat = (y_hat_dx.pow(2) + y_hat_dy.pow(2) + eps).sqrt()

        # MAE of gradients
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

        # dx
        x_kernel = torch.tensor([1, -1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # dy
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
    Implements the G-Loss function.
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

        # MAE of sums of gradients
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

        # dx
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 3, 3))
        self.register_buffer('x_kernel', x_kernel)

        # dy
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

        # dx kernel
        x_kernel = torch.tensor([-1, 1], dtype=torch.float32)
        x_kernel = torch.reshape(x_kernel, (1, 1, 1, 2))
        self.register_buffer('x_kernel', x_kernel)

        # dy kernel
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

        # dx and dropping the last row to match sizes with dy
        y_hat_dx = torch.conv2d(y_hat, self.x_kernel, padding=0)
        y_dx = torch.conv2d(y, self.x_kernel, padding=0)
        y_hat_dx = y_hat_dx[:, :, :-1, :]
        y_dx = y_dx[:, :, :-1, :]

        # dy and dropping the last column to match sizes with dx
        y_hat_dy = torch.conv2d(y_hat, self.y_kernel, padding=0)
        y_dy = torch.conv2d(y, self.y_kernel, padding=0)
        y_hat_dy = y_hat_dy[:, :, :, :-1]
        y_dy = y_dy[:, :, :, :-1]

        # cosine similarity, scaling with magnitude, we don't want to punish large differences in angles
        # when the gradient magnitude is close to zero
        g_y_hat_xy = torch.stack([y_hat_dx, y_hat_dy], dim=1)

        g_y_xy = torch.stack([y_dx, y_dy], dim=1)

        sim = torch.cosine_similarity(g_y_hat_xy, g_y_xy, dim=1)

        loss = ((1 - sim) * (y_dx.pow(2) + y_dy.pow(2)).sqrt()).abs().mean()
        return loss


# Lei Ge, Lei Dou. 2023. G-Loss: A loss function with gradient information for super-resolution.
# Optik - International Journal for Light and Electron Optics. https://doi.org/10.1016/j.ijleo.2023.170750.
class SoftGLoss(torch.nn.Module):
    """
    Modifies the G-Loss to match the definition of soft constraint.
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

    def forward(self, y_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the G-Loss.
        Args:
            y_hat:  output image
            y:      target image

        Returns:
            loss
        """


        # we will do gradients on distance of scale_factor
        # y_hat = (B, 1, H, W)

        # a particular image is divided into pieces, where in all scale_factor^2 channels, there are
        # H / scale_factor x W / scale_factor pixels, originally in distance of scale_factor from each other
        # u_y_hat = (B, scale_factor^2, H / scale_factor, W / scale_factor)
        u_y_hat = self.unshuffle(y_hat)

        _, _, h, w = u_y_hat.shape

        # for easier convolution, we will move the channels into batches
        # u_y_hat = (B * scale_factor^2, 1, H / scale_factor, W / scale_factor)
        u_y_hat = u_y_hat.view(-1, 1, h, w)

        # now the gradient calculation takes place
        # for each batch we will now inspect how the value changed in 8 directions
        # this corresponds to 8 directions in distance scale_factor in original image
        # D_u_y_hat = (B * scale_factor^2, 8, H / scale_factor - 2, W / scale_factor - 2)
        D_u_y_hat = torch.conv2d(u_y_hat, self.kernel, padding=0)

        # now for each image, there are scale_factor^2 images with stride scale_factor
        # and 8 gradients for 8 different directions
        # D_u_y_hat = (B, scale_factor^2, 8, H / scale_factor - 2, W / scale_factor - 2)
        D_u_y_hat = D_u_y_hat.view(-1, self.scale_factor ** 2, 8, h - 2, w - 2)

        # we will calculate the mean differences through dimension 1, that leads to 8 channels for 8 direction,
        # where in each direction we have the mean difference between two groups (scale_factor x scale_factor
        # regions) of SR pixels downscaled from two LR pixels, where these means and the difference in LR image should
        # be the same
        # D_u_y_hat_mean = (B, 8, H / scale_factor - 2, W / scale_factor - 2)
        D_u_y_hat_mean = D_u_y_hat.mean(dim=1, keepdim=True)

        # x = (B, 1, H / scale_factor, W / scale_factor)
        # D_x = (B, 8, H / scale_factor - 2, W / scale_factor - 2)
        D_x = torch.conv2d(x, self.kernel, padding=0)

        loss = (D_x - D_u_y_hat_mean).abs().mean()

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
#     #
#     dcloss = SoftGLoss(3)
#     #
#     x = torch.randn(2, 1, 4, 4)
#     y_hat = torch.randn(2, 1, 12, 12)
#
#     # x = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).float()
#     # y_hat = torch.tensor([[[[1, 2, 3, 4, 5, 6],
#     #                         [7, 1, 2, 3, 4, 5],
#     #                         [6, 7, 1, 2, 3, 4],
#     #                         [5, 6, 7, 1, 2, 3],
#     #                         [4, 5, 6, 7, 1, 2],
#     #                         [3, 4, 5, 6, 7, 1]]],
#     #                       [[[1, 1, 2, 2, 3, 3],
#     #                         [1, 1, 2, 2, 3, 3],
#     #                         [4, 4, 5, 5, 6, 6],
#     #                         [4, 4, 5, 5, 6, 6],
#     #                         [7, 7, 8, 8, 9, 9],
#     #                         [7, 7, 8, 8, 9, 9]]]]).float()
#
#     print(dcloss(y_hat, x))
#     print(y_hat)


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
            raise ValueError("Sum of weights must be 1")

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
        self.soft_gloss = SoftGLoss(scale_factor)

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
            loss += self.weights["simple_gradient"] * self.simple_gradient(y_hat, y)

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

        if "soft_gloss" in self.weights.keys():
            loss += self.weights["soft_gloss"] * self.soft_gloss(y_hat, x)

        return loss
