# losses/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import torchvision.models as models


# ---------- KL helpers (keep your original semantics, but safer dtype/device)
def latent_kl(prior_mean: torch.Tensor, posterior_mean: torch.Tensor) -> torch.Tensor:
    """
    Simple 0.5 * (p - q)^2 per-element, sum over spatial/channel dims, mean over batch.
    Keeps original behavior but uses stable ops & explicit dims.
    """
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    # sum over channel+spatial dims (assumes shape [B, C, H, W])
    kl = torch.sum(kl, dim=[1, 2, 3])
    return torch.mean(kl)


def aggregate_kl_loss(prior_means: Dict[str, torch.Tensor], posterior_means: Dict[str, torch.Tensor]) -> torch.Tensor:
    kl_stages = []
    for p, q in zip(list(prior_means.values()), list(posterior_means.values())):
        kl_stages.append(latent_kl(p, q).unsqueeze(dim=-1))
    if len(kl_stages) == 0:
        return torch.tensor(0.0, device=next(iter(prior_means.values())).device if len(prior_means) else 'cpu')
    kl_stages = torch.cat(kl_stages, dim=-1)
    kl_loss = torch.sum(kl_stages, dim=-1)
    # kl_loss is (batch, ) so return mean
    return torch.mean(kl_loss)


# ---------- Intensity loss (L1 or L2)
class Intensity_Loss(nn.Module):
    def __init__(self, l_num: int = 1):
        super(Intensity_Loss, self).__init__()
        assert l_num in (1, 2), "l_num should be 1 (L1) or 2 (L2)"
        self.l_num = l_num

    def forward(self, gen_frames: torch.Tensor, gt_frames: torch.Tensor) -> torch.Tensor:
        if self.l_num == 1:
            return torch.mean(torch.abs(gen_frames - gt_frames))
        else:
            return torch.mean((gen_frames - gt_frames) ** 2)


# ---------- Gradient loss (keeps your implementation, minor fixes)
class Gradient_Loss(nn.Module):
    def __init__(self, alpha: float, channels: int, device: str):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.device = device
        # horizontal filter [-1, 1]
        filt = torch.FloatTensor([[-1., 1.]]).to(device)
        self.filter_x = filt.view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # conv2d expects (out_ch, in_ch, kh, kw)
        self.filter_y = filt.view(1, 1, 2, 1).repeat(channels, 1, 1, 1)

    def forward(self, gen_frames: torch.Tensor, gt_frames: torch.Tensor) -> torch.Tensor:
        # pad then conv: conv2d with groups=channels (depthwise)
        gen_x = F.pad(gen_frames, (1, 0, 0, 0))
        gen_y = F.pad(gen_frames, (0, 0, 1, 0))
        gt_x = F.pad(gt_frames, (1, 0, 0, 0))
        gt_y = F.pad(gt_frames, (0, 0, 1, 0))

        # depthwise conv: use groups=channels and filters shaped (channels, 1, kh, kw)
        gen_dx = F.conv2d(gen_x, self.filter_x, groups=gen_frames.shape[1])
        gen_dy = F.conv2d(gen_y, self.filter_y, groups=gen_frames.shape[1])
        gt_dx = F.conv2d(gt_x, self.filter_x, groups=gt_frames.shape[1])
        gt_dy = F.conv2d(gt_y, self.filter_y, groups=gt_frames.shape[1])

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)
        return torch.mean((grad_diff_x ** self.alpha) + (grad_diff_y ** self.alpha))


# ---------- Entropy loss (keeps original)
class Entropy_Loss(nn.Module):
    def __init__(self):
        super(Entropy_Loss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-20
        tmp = torch.sum((-x) * torch.log(x + eps), dim=-1)
        return torch.mean(tmp)


# ---------- SSIM (differentiable) implementation
def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def create_window(window_size: int, channel: int, device: torch.device):
    _1D_window = _gaussian(window_size, 1.5).to(device)
    _2D_window = _1D_window[:, None] @ _1D_window[None, :]
    window = _2D_window.unsqueeze(0).unsqueeze(0)  # 1,1,ws,ws
    window = window.repeat(channel, 1, 1, 1)  # channel,1,ws,ws
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True):
    """
    img1,img2: (B,C,H,W), values in [0,1] recommended
    returns mean SSIM over batch and channels
    """
    channel = img1.shape[1]
    device = img1.device
    window = create_window(window_size, channel, device)

    mu1 = F.conv2d(img1, window, groups=channel, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, groups=channel, padding=window_size // 2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel, padding=window_size // 2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM_Loss(nn.Module):
    def __init__(self, window_size: int = 11):
        super(SSIM_Loss, self).__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(x, y, window_size=self.window_size, size_average=True)


# ---------- Perceptual (VGG) loss
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        # Use blocks up to relu2_2 or relu3_3 depending on resource; here use first few layers
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(0, 9)])   # relu2_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(9, 16)])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.resize = resize
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # expects input in [0,1]; normalize to ImageNet
        # If single-channel, repeat to 3 channels
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        f1 = self.slice1(pred_n)
        f2 = self.slice1(target_n)
        loss = F.l1_loss(f1, f2)
        f1 = self.slice2(pred_n)
        f2 = self.slice2(target_n)
        loss = loss + F.l1_loss(f1, f2)
        return loss


# ---------- Combined wrapper loss (for training script)
class CombinedLoss(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str):
        super().__init__()
        # weights - allow config to use lam_* names or nested loss_weights
        lw = config.get("loss_weights", {})
        self.lam_kl = config.get("lam_kl", lw.get("lam_kl", 1.0))
        self.lam_frame = config.get("lam_frame", lw.get("lam_frame", 1.0))
        self.lam_grad = config.get("lam_grad", lw.get("lam_grad", 1.0))
        self.lam_recon = config.get("lam_recon", lw.get("lam_recon", 1.0))
        self.lam_sparsity = config.get("lam_sparsity", lw.get("lam_sparsity", 0.0))
        self.lam_percep = config.get("lam_percep", lw.get("lam_percep", 0.0))
        self.lam_ssim = config.get("lam_ssim", lw.get("lam_ssim", 0.0))

        # components
        self.intensity = Intensity_Loss(l_num=config.get("intensity_loss_norm", 1))
        self.grad = Gradient_Loss(config.get("alpha", 1.0),
                                  config["model_paras"]["img_channels"] * config["model_paras"]["clip_pred"],
                                  device)
        self.percep = PerceptualLoss(device=device) if self.lam_percep > 0 else None
        self.ssim = SSIM_Loss() if self.lam_ssim > 0 else None

    def forward(self, model_out: Dict[str, torch.Tensor], gt_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        model_out expected to contain:
          - 'q_means' (dict), 'p_means' (dict)
          - 'frame_pred' : predicted frames (B, C, H, W)
          - 'frame_target' : gt frames (B, C, H, W)
          - optionally 'loss_recon', 'loss_sparsity' computed in model (scalars)
        Returns dict with components + total 'loss_all'
        """
        device = gt_frames.device
        loss_kl = aggregate_kl_loss(model_out.get("q_means", {}), model_out.get("p_means", {}))
        loss_frame = self.intensity(model_out["frame_pred"], model_out["frame_target"])
        loss_grad = self.grad(model_out["frame_pred"], model_out["frame_target"])
        loss_recon = model_out.get("loss_recon", torch.tensor(0.0, device=device))
        loss_sparsity = model_out.get("loss_sparsity", torch.tensor(0.0, device=device))

        loss_percep = torch.tensor(0.0, device=device)
        if self.percep is not None:
            loss_percep = self.percep(model_out["frame_pred"], model_out["frame_target"])

        loss_ssim = torch.tensor(0.0, device=device)
        if self.ssim is not None:
            loss_ssim = self.ssim(model_out["frame_pred"], model_out["frame_target"])

        loss_all = (self.lam_kl * loss_kl +
                    self.lam_frame * loss_frame +
                    self.lam_grad * loss_grad +
                    self.lam_recon * loss_recon +
                    self.lam_sparsity * loss_sparsity +
                    self.lam_percep * loss_percep +
                    self.lam_ssim * loss_ssim)

        return dict(
            loss_all=loss_all,
            loss_kl=loss_kl,
            loss_frame=loss_frame,
            loss_grad=loss_grad,
            loss_recon=loss_recon,
            loss_sparsity=loss_sparsity,
            loss_percep=loss_percep,
            loss_ssim=loss_ssim,
        )
