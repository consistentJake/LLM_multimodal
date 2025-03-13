import torch
from torch.utils.data import Dataset
from collections import defaultdict

class ContrastiveDataset(Dataset):
    def __init__(self, sparse, dense, emb, labels, business_ids, user_ids):
        self.sparse = sparse
        self.dense = dense
        self.emb = emb
        self.labels = labels
        self.business_ids = business_ids
        self.user_ids = user_ids

        # Precompute indices for each label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels.numpy()):
            self.label_to_indices[int(label)].append(idx)

        # Convert lists to tensors for faster indexing
        self.label_to_indices = {
            label: torch.tensor(indices)
            for label, indices in self.label_to_indices.items()
        }

        # Number of samples for each label
        self.label_to_num_samples = {
            label: len(indices)
            for label, indices in self.label_to_indices.items()
        }
        self.number_samples = sparse.shape[0]

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        # Get anchor sample
        emb = self.emb[idx]
        anchor_label = int(self.labels[idx])

        # Get positive sample (same label)
        positive_indices = self.label_to_indices[anchor_label]
        rand_idx = idx
        while rand_idx == idx:
            rand_idx = torch.randint(0, self.label_to_num_samples[anchor_label], (1,))

        positive_idx = positive_indices[rand_idx].item()
        positive_embed = self.emb[positive_idx]

        return (
            self.sparse[idx],
            self.dense[idx],
            self.emb[idx],
            positive_embed,
            anchor_label,
            self.business_ids[idx],
            self.user_ids[idx]
        )
