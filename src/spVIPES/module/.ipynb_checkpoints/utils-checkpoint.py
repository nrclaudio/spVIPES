import torch
import torch.nn as nn


def get_kl(mu: torch.Tensor, logsigma: torch.Tensor):
    """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
    Args:
        mu: the mean of the q distribution.
        logsigma: the log of the standard deviation of the q distribution.
    Returns:
        KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
    """
    logsigma = 2 * logsigma
    return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(1)


def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float("inf")) | (xm == float("-inf")),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)),
    )
    return x if keepdim else x.squeeze(dim)


def mutual_information(matrix1, matrix2, sigma=0.1, num_bins=256, normalize=True):
    """
    Calculate the mutual information between two input 2D matrices.

    matrix1: torch.Tensor
        Input matrix 1 of shape (B, D), where B is the batch size and D is the dimensionality of each matrix.
    matrix2: torch.Tensor
        Input matrix 2 of shape (B, D), where B is the batch size and D is the dimensionality of each matrix.
    sigma: float, optional
        Bandwidth for the Gaussian kernel (default: 0.1).
    num_bins: int, optional
        Number of bins for histogram approximation (default: 256).
    normalize: bool, optional
        Whether to normalize the mutual information (default: True).

    Returns:
    mutual_information: torch.Tensor
        A tensor of shape (B,) containing the mutual information between the input matrices for each batch element.
    """
    epsilon = 1e-10

    # Calculate the bin centers based on the minimum and maximum values in the input matrices
    min_value_1 = torch.min(matrix1).item()
    max_value_1 = torch.max(matrix1).item()
    min_value_2 = torch.min(matrix2).item()
    max_value_2 = torch.max(matrix2).item()

    def marginalPdf(values, min_value, max_value, num_bins, device):
        bins = torch.linspace(min_value, max_value, num_bins, device=device).float()
        residuals = values.unsqueeze(1) - bins.unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def jointPdf(kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    device = matrix1.device
    pdf_matrix1, kernel_values1 = marginalPdf(matrix1, min_value_1, max_value_1, matrix1.shape[1], device)
    pdf_matrix2, kernel_values2 = marginalPdf(matrix2, min_value_2, max_value_2, matrix2.shape[1], device)
    pdf_matrix1x2 = jointPdf(kernel_values1, kernel_values2)

    H_matrix1 = -torch.sum(pdf_matrix1 * torch.log2(pdf_matrix1 + epsilon), dim=1)
    H_matrix2 = -torch.sum(pdf_matrix2 * torch.log2(pdf_matrix2 + epsilon), dim=1)
    H_matrix1x2 = -torch.sum(pdf_matrix1x2 * torch.log2(pdf_matrix1x2 + epsilon), dim=(1, 2))

    mutual_information = H_matrix1 + H_matrix2 - H_matrix1x2

    if normalize:
        mutual_information = 2 * mutual_information / (H_matrix1 + H_matrix2)

    return mutual_information
