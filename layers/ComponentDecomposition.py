import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _sanitize_kernel(kernel_size: int, length: int) -> int:
    if length <= 1:
        return 1
    kernel = max(1, int(kernel_size))
    max_kernel = length if (length % 2 == 1) else max(1, length - 1)
    kernel = min(kernel, max_kernel)
    if kernel % 2 == 0:
        kernel -= 1
    return max(1, kernel)


def moving_average_same_length(x, kernel_size: int):
    """
    Moving average that keeps sequence length unchanged.
    Supported shapes:
    - [B, L, C]
    - [N, L, C]
    - [L, C]
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    squeeze_batch = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_batch = True

    if x.dim() != 3:
        raise ValueError(f"moving_average_same_length expects 2D/3D input, got {tuple(x.shape)}")

    bsz, seq_len, channels = x.shape
    kernel = _sanitize_kernel(kernel_size=kernel_size, length=seq_len)
    if kernel == 1:
        return x.squeeze(0) if squeeze_batch else x

    pad = kernel // 2
    x_t = x.permute(0, 2, 1)  # [B, C, L]

    if seq_len > 1 and pad < seq_len:
        x_t = F.pad(x_t, (pad, pad), mode='reflect')
    else:
        x_t = F.pad(x_t, (pad, pad), mode='replicate')

    weight = torch.ones((channels, 1, kernel), device=x.device, dtype=x.dtype) / float(kernel)
    out = F.conv1d(x_t, weight, groups=channels)
    out = out.permute(0, 2, 1)  # [B, L, C]
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if squeeze_batch:
        out = out.squeeze(0)
    return out


def _sanitize_stl_smoother(window: int, length: int, minimum: int = 7):
    if length < 3:
        return None
    max_window = length if length % 2 == 1 else length - 1
    if max_window < minimum:
        return max(3, max_window)
    window = max(minimum, int(window))
    window = min(window, max_window)
    if window % 2 == 0:
        window -= 1
    return max(3, window)


def stl_decompose_same_length(
    x,
    trend_kernel=25,
    seasonal_kernel=8,
    stl_period=None,
    show_progress=False,
    progress_desc='STL',
):
    """
    STL decomposition for [B, L, C], [N, L, C], or [L, C].
    Uses statsmodels on CPU and restores the original tensor device/dtype.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    squeeze_batch = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_batch = True

    if x.dim() != 3:
        raise ValueError(f"stl_decompose_same_length expects 2D/3D input, got {tuple(x.shape)}")

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError as exc:
        raise ImportError(
            "STL decomposition requires statsmodels. Install it in the DRTS environment with: "
            "pip install statsmodels"
        ) from exc

    device = x.device
    dtype = x.dtype
    x_np = x.detach().cpu().numpy()
    bsz, seq_len, channels = x_np.shape

    period = int(stl_period) if stl_period is not None and int(stl_period) > 0 else int(seasonal_kernel)
    period = max(2, period)
    seasonal_window = _sanitize_stl_smoother(max(7, int(seasonal_kernel)), seq_len, minimum=7)
    trend_window = _sanitize_stl_smoother(max(int(trend_kernel), period + 1), seq_len, minimum=7)

    trend_np = np.zeros_like(x_np, dtype=np.float32)
    seasonal_np = np.zeros_like(x_np, dtype=np.float32)
    residual_np = np.zeros_like(x_np, dtype=np.float32)

    if seq_len < 2 * period + 1 or seasonal_window is None or trend_window is None:
        if show_progress:
            print(
                f"[STL] {progress_desc}: seq_len={seq_len} is too short for "
                f"period={period}; using moving_avg fallback."
            )
        trend = moving_average_same_length(x, kernel_size=trend_kernel)
        detrended = x - trend
        seasonal = moving_average_same_length(detrended, kernel_size=seasonal_kernel)
        residual = x - trend - seasonal
        return trend, seasonal, residual

    total = bsz * channels
    iterator = ((b, c) for b in range(bsz) for c in range(channels))
    iterator = tqdm(
        iterator,
        total=total,
        desc=progress_desc,
        unit='series',
        leave=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for b, c in iterator:
        series = np.nan_to_num(x_np[b, :, c], nan=0.0, posinf=0.0, neginf=0.0)
        try:
            res = STL(
                series,
                period=period,
                seasonal=seasonal_window,
                trend=trend_window,
                robust=True,
            ).fit()
            trend_np[b, :, c] = res.trend.astype(np.float32)
            seasonal_np[b, :, c] = res.seasonal.astype(np.float32)
            residual_np[b, :, c] = res.resid.astype(np.float32)
        except Exception:
            series_t = torch.as_tensor(series, dtype=dtype, device=device).view(1, seq_len, 1)
            trend_t = moving_average_same_length(series_t, kernel_size=trend_kernel)
            detrended_t = series_t - trend_t
            seasonal_t = moving_average_same_length(detrended_t, kernel_size=seasonal_kernel)
            residual_t = series_t - trend_t - seasonal_t
            trend_np[b, :, c] = trend_t.squeeze(0).squeeze(-1).detach().cpu().numpy()
            seasonal_np[b, :, c] = seasonal_t.squeeze(0).squeeze(-1).detach().cpu().numpy()
            residual_np[b, :, c] = residual_t.squeeze(0).squeeze(-1).detach().cpu().numpy()

    trend = torch.as_tensor(trend_np, dtype=dtype, device=device)
    seasonal = torch.as_tensor(seasonal_np, dtype=dtype, device=device)
    residual = torch.as_tensor(residual_np, dtype=dtype, device=device)

    if squeeze_batch:
        trend = trend.squeeze(0)
        seasonal = seasonal.squeeze(0)
        residual = residual.squeeze(0)

    return trend, seasonal, residual


def decompose_series(
    x,
    trend_kernel=25,
    seasonal_kernel=8,
    stl_period=None,
    show_progress=False,
    progress_desc='STL',
):
    """
    Decompose series into trend/seasonal/residual components with STL.
    Input and outputs keep exactly the same shape.
    """
    trend, seasonal, residual = stl_decompose_same_length(
        x,
        trend_kernel=trend_kernel,
        seasonal_kernel=seasonal_kernel,
        stl_period=stl_period,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    trend = torch.nan_to_num(trend, nan=0.0, posinf=0.0, neginf=0.0)
    seasonal = torch.nan_to_num(seasonal, nan=0.0, posinf=0.0, neginf=0.0)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    return trend, seasonal, residual
