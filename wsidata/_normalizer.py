import torch


def transform(tensor):
    # Convert to float32 and scale (assuming input is uint8 or another dtype)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    tensor = tensor.to(torch.float32)
    # Scale if needed (assuming input is originally in [0,1] or [0,255])
    if tensor.max() <= 1:  # If already normalized in [0,1], scale to [0,255]
        tensor = tensor * 255
    return tensor


class ColorNormalizer(torch.nn.Module):
    def __init__(self, method="macenko"):
        super().__init__()

        try:
            import torchstain.torch.normalizers as norm
            import torchstain.torch.augmentors as aug
        except (ImportError, ModuleNotFoundError):
            raise ImportError("To use color normalization, please install torchstain.")

        self.method = method
        if method == "macenko":
            normalizer = norm.TorchMacenkoNormalizer()
        elif method == "reinhard":
            normalizer = norm.TorchReinhardNormalizer()
            normalizer.target_means = torch.tensor([72.909996, 20.8268, -4.9465137])
            normalizer.target_stds = torch.tensor([18.560713, 14.889295, 5.6756697])
        elif method == "reinhard_modified":
            normalizer = norm.TorchReinhardNormalizer(method="modified")
            normalizer.target_means = torch.tensor([72.909996, 20.8268, -4.9465137])
            normalizer.target_stds = torch.tensor([18.560713, 14.889295, 5.6756697])
        elif method == "multi_macenko":
            normalizer = norm.TorchMultiMacenkoNormalizer()
        elif method == "macenko_aug":
            normalizer = aug.TorchMacenkoAugmentor()
        else:
            raise NotImplementedError(f"Requested method '{method}' not implemented")
        self.normalizer = normalizer

    def __repr__(self):
        return f"ColorNormalizer(method='{self.method}')"

    def forward(self, img):
        t_img = transform(img)
        if self.method == "macenko":
            norm, _, _ = self.normalizer.normalize(I=t_img)
        else:
            norm = self.normalizer.normalize(I=t_img)
        return norm
