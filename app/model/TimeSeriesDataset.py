import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets=None, sequence_length=6):
        self.features = torch.FloatTensor(features) if not isinstance(features, torch.Tensor) else features
        self.targets = torch.FloatTensor(targets) if targets is not None and not isinstance(targets,
                                                                                            torch.Tensor) else targets
        self.sequence_length = sequence_length
        self.has_targets = targets is not None

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length + 1)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        if self.targets is not None:
            y = self.targets[idx + self.sequence_length - 1]
            return x, y
        else:
            # For prediction, ensure tensor shape is [sequence_length, n_features]
            # and then transpose to [n_features, sequence_length] for TCN
            return x.transpose(0, 1)  # Change shape from [seq_len, features] to [features, seq_len]
