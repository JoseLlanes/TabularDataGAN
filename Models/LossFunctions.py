import torch


def covariance_matrix(x):
    """
    Computes the upper-triangular values of the covariance matrix for a given 2D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, num_features].

    Returns:
        torch.Tensor: Upper-triangular values (excluding diagonal) of the covariance matrix.
    """
    # Ensure input is 2D
    if x.dim() != 2:
        raise ValueError("Input tensor must have shape [batch_size, num_features]")

    # Compute mean and center the data
    mean_x = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mean_x

    # Compute covariance matrix
    cov_matrix = torch.mm(x_centered.T, x_centered) / (x.shape[0] - 1)

    # Extract upper triangular values (excluding diagonal)
    num_features = cov_matrix.shape[0]
    upper_indices = torch.triu_indices(num_features, num_features, offset=1)
    upper_diag_values = cov_matrix[upper_indices[0], upper_indices[1]]
    
    return upper_diag_values


def correlation_matrix(x):
    # Ensure y_pred is 2D: [batch_size, num_features]
    if x.dim() != 2:
        raise ValueError("Input tensor must have shape [batch_size, num_features]")

    mean_x = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mean_x

    cov_matrix = torch.mm(x_centered.T, x_centered) / (x.shape[0] - 1)
    std_dev = torch.sqrt(torch.diag(cov_matrix) + 1e-8)
    correlation_matrix = cov_matrix / (std_dev[:, None] * std_dev[None, :])

    num_features = correlation_matrix.shape[0]
    upper_indices = torch.triu_indices(num_features, num_features, offset=1)
    upper_diag_values = correlation_matrix[upper_indices[0], upper_indices[1]]
    
    return upper_diag_values


def autocorrelation(x, epsilon=1e-8):
    x_mean = torch.mean(x, dim=0)
    x_shifted = torch.roll(x, shifts=1, dims=0)
    numerator = torch.sum((x - x_mean) * (x_shifted - x_mean), dim=0)
    denominator = torch.sum((x - x_mean) ** 2, dim=0)
    return numerator / (denominator + epsilon)


def custom_loss(y_pred, data_target, method="all"):
    median_pred = torch.median(y_pred, dim=0).values
    median_target = torch.median(data_target, dim=0).values

    q1_pred = torch.quantile(y_pred, 0.25, dim=0)
    q3_pred = torch.quantile(y_pred, 0.75, dim=0)
    iqr_pred = q3_pred - q1_pred

    q1_target = torch.quantile(data_target, 0.25, dim=0)
    q3_target = torch.quantile(data_target, 0.75, dim=0)
    iqr_target = q3_target - q1_target

    autocorr_pred = autocorrelation(y_pred)
    autocorr_target = autocorrelation(data_target)

    if method == "all":
        custom_loss_value = (
            # torch.sum(torch.abs(mean_pred - mean_target)) +
            torch.sum(torch.abs(median_pred - median_target)) +
            torch.sum(torch.abs(iqr_pred - iqr_target)) +
            torch.sum(torch.abs(autocorr_pred - autocorr_target)) +
            torch.sum(torch.abs(correlation_matrix(y_pred) - correlation_matrix(data_target))) + 
            torch.sum(torch.abs(covariance_matrix(y_pred) - covariance_matrix(data_target)))
        )
    else:
        custom_loss_value = 0
        if "median" in method:
            custom_loss_value += torch.sum(torch.abs(median_pred - median_target))
        if "iqr" in method:
            custom_loss_value += torch.sum(torch.abs(iqr_pred - iqr_target))
        if "autocorr" in method:
            custom_loss_value += torch.sum(torch.abs(autocorr_pred - autocorr_target))
        if "corrmat" in method:
            custom_loss_value += torch.sum(torch.abs(correlation_matrix(y_pred) - correlation_matrix(data_target)))
        if "covmat" in method:
            custom_loss_value += torch.sum(torch.abs(covariance_matrix(y_pred) - covariance_matrix(data_target)))
    
    return custom_loss_value