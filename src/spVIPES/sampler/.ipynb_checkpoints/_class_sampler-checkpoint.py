from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


class ClassSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_indices = self._get_class_indices()

    def _get_class_indices(self):
        class_indices = {}
        for i, d in enumerate(self.dataset):
            label = d["labels"][0]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        return class_indices

    def __iter__(self):
        indices = []
        for class_indices in self.class_indices.values():
            indices += class_indices
        return iter(indices)

    def __len__(self):
        return len(self.dataset)
