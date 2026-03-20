import torch

from .model import AbstractModel


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


class AddConstraintLayer(ConstraintLayer):
    """
    Enforces conservation law using sums.
    y_hat_j = y_tilde_j + sum x_i - sum y_tilde_i
    Harder et al.
    """

    def __init__(self, scale_factor: int):
        super().__init__(scale_factor)

        self.upscale = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = torch.nn.AvgPool2d(scale_factor)

    def forward(self, x: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
        means = self.avg_pool(y_tilde)
        mean_diff = x - means
        mean_diff_upscaled = self.upscale(mean_diff)
        return y_tilde + mean_diff_upscaled


class MultConstraintLayer(ConstraintLayer):
    """
    Enforces conservation law using product.
    y_hat_j = y_tilde_j * (sum x_i / sum y_tilde_i)
    Harder et al.
    """

    def __init__(self, scale_factor: int):
        super().__init__(scale_factor)

        self.upscale = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = torch.nn.AvgPool2d(scale_factor)

    def forward(self, x: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
        means = self.avg_pool(y_tilde)
        eps = 1e-8
        means = means.sign() * torch.clamp(means.abs(), min=eps)
        mean_diff = x.div(means)
        mean_diff_upscaled = self.upscale(mean_diff)
        return y_tilde.mul(mean_diff_upscaled)


class SmConstraintLayer(ConstraintLayer):
    """
    Enforces conservation law using softmax.
    Also enforces positivity, do not use on possible negative y-s.
    y_hat_j = exp(y_tilde_j) * (sum x_i / sum exp(y_tilde_i))
    Harder et al.
    """

    def __init__(self, scale_factor: int):
        super().__init__(scale_factor)

        self.upscale = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = torch.nn.AvgPool2d(scale_factor)

    def forward(self, x: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
        y_tilde_exp = torch.exp(y_tilde)
        means = self.avg_pool(y_tilde_exp)

        # always positive means
        eps = 1e-8
        means = torch.clamp(means.abs(), min=eps)

        mean_diff = x.div(means)
        mean_diff_upscaled = self.upscale(mean_diff)
        return y_tilde_exp.mul(mean_diff_upscaled)


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
        y_tilde = self.model(x)

        # constraint layer
        y_hat = self.constraint(x, y_tilde)
        return y_hat
