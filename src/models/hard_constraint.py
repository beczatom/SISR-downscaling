"""
The layer types are from Harder et al. https://www.jmlr.org/papers/v24/23-0158.html
"""

import torch

from .model import AbstractModel

EPS = 1e-8


class ConstraintLayer(torch.nn.Module):
    """
    Abstract constraint layer.
    """

    def __init__(self, scale_factor: int):
        """
        Initialize the constraint layer.
        """
        super().__init__()
        self.scale_factor = scale_factor


# Section 1.3.3.2
class AddConstraintLayer(ConstraintLayer):
    """
    Enforces conservation law using sums.

    Suppose an input image X in R^n, and a model output image Y_tilde in R^m.
    For the sake of simplicity the indexation is only one dimensional,
    allowing us to state, that for j-th pixel of X,
    the model makes mapping to a set I_j, which is a subset of {1, ..., m}.
    These I_j are pairwise disjoint sets and for the scale factor of s, their sizes are s^2.
    (Thus the model maps X_j to {Y_tilde_i | i in I_j})

    For each i-th pixel of Y_tilde, where i in I_j, we calculate the constrained output Y_hat_i as:
    Y_hat_i = Y_tilde_i + X_j - s^-2 * sum_{k in I_j} Y_tilde_k
    """

    def __init__(self, scale_factor: int):
        super().__init__(scale_factor)

        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = torch.nn.AvgPool2d(scale_factor)

    def forward(self, x: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
        means = self.avg_pool(y_tilde)
        mean_diff = x - means
        mean_diff_upsampled = self.upsample(mean_diff)
        return y_tilde + mean_diff_upsampled


# Section 1.3.3.2
class MultConstraintLayer(ConstraintLayer):
    """
    Enforces conservation law using product.

    The setup is the same as in AddConstraintLayer.

    For each i-th pixel of Y_tilde, where i in I_j, we calculate the constrained output Y_hat_i as:
    Y_hat_i = Y_tilde_i * X_j / (s^-2 * sum_{k in I_j} Y_tilde_k)

    Divisions around zero, will be unstable.
    """

    def __init__(self, scale_factor: int):
        super().__init__(scale_factor)

        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = torch.nn.AvgPool2d(scale_factor)

    def forward(self, x: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
        means = self.avg_pool(y_tilde)

        # preventing division by zero
        means = means.sign() * torch.clamp(means.abs(), min=EPS)

        mean_ratio = x.div(means)
        mean_ratio = self.upsample(mean_ratio)
        return y_tilde.mul(mean_ratio)


# Section 1.3.3.2
class SmConstraintLayer(ConstraintLayer):
    """
    Enforces conservation law using softmax.

    The setup is the same as in AddConstraintLayer.

    For each i-th pixel of Y_tilde, where i in I_j, we calculate the constrained output Y_hat_i as:
    Y_hat_i = exp(Y_tilde_i) * X_j / (s^-2 * sum_{k in I_j} exp(Y_tilde_k))

    From the equation it is obvious that it enforces same sign as X_j,
    this can in some cases be useful.
    However in temperature in degrees C, near zero the water freezes, fine,
    but the air dynamics doesn't change rapidly, meaning it is far less interesting point than 0K.
    Additionally, when Y_tilde_i and the mean in the denominator are near 0 deg C,
    we divide by 1e-8, which can output a large number, and the exp of zero will be near 1,
    so this large output can be yielded.
    """

    def __init__(self, scale_factor: int):
        super().__init__(scale_factor)

        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = torch.nn.AvgPool2d(scale_factor)

    def forward(self, x: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
        y_tilde_exp = torch.exp(y_tilde)
        means = self.avg_pool(y_tilde_exp)

        # always positive means
        means = torch.clamp(means.abs(), min=EPS)

        mean_diff = x.div(means)
        mean_diff_upsampled = self.upsample(mean_diff)
        return y_tilde_exp.mul(mean_diff_upsampled)


class ConstrainedModel(AbstractModel):
    """
    Hard constrained model.
    """

    def __init__(self, model: AbstractModel, constraint: ConstraintLayer):
        """
        Initializes the hard constrained model.
        Args:
            model : AbstractModel           model which output will be the input into constraint layer
            constraint : ConstraintLayer    hard constraint layer
        """
        super().__init__()
        self.model = model
        self.constraint = constraint
        self.loss = self.model.loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hard constrained model.
        Args:
            x : torch.Tensor    Input image
        Returns:
            y_hat: torch.Tensor Predicted image
        """

        # output of the model, y_tilde, will be passed into constraint layer
        y_tilde = self.model(x)

        # constraint layer to obtain y_hat = final prediction
        y_hat = self.constraint(x, y_tilde)
        return y_hat
