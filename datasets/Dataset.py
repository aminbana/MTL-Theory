import PIL
import torch
from torch.utils.data import Dataset


class ContinualDataset(Dataset):
    def __init__(self, data, active_classes = None, is_a_path_dataset = False):

        # dataset shape is (num_classes, num_samples_per_class, channels, height, width)
        # labels shape is (num_classes)

        self.is_a_path_dataset = is_a_path_dataset

        self.data = data
        self.transform = None

        self.set_active_classes(active_classes)

    def set_active_classes(self, active_classes):
        if active_classes is None:
            active_classes = list(range(len(self.data)))

        self.active_classes = active_classes
        self.num_samples_per_active_class = [len (self.data[active_classes[d]]) for d in range(len(active_classes))]



    def __len__(self):
        return sum(self.num_samples_per_active_class)


    def __getitem__(self, idx):
        class_idx = 0
        while idx >= self.num_samples_per_active_class[class_idx]:
            idx -= self.num_samples_per_active_class[class_idx]
            class_idx += 1

        sample_idx = idx

        class_ = self.active_classes[class_idx]

        sample = self.data[class_][sample_idx]
        label = class_

        if self.is_a_path_dataset:
            sample = PIL.Image.open(sample).convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def set_transform(self, transform):
        self.transform = transform


class ReplayBuffer(ContinualDataset):
    def __init__(self, mem_size_per_task, num_all_tasks, fixed_mem_size = False):
        self.mem_size_per_task = mem_size_per_task
        self.num_all_tasks = num_all_tasks
        self.fixed_mem_size = fixed_mem_size

        super().__init__(torch.empty([0,0]), active_classes=None)

        self.data_ = []

    def get_per_class_budget(self, num_previous_classes, num_new_classes):
        num_all_classes = num_previous_classes + num_new_classes
        if self.fixed_mem_size:
            return self.mem_size_per_task * self.num_all_tasks // num_all_classes
        else:
            return self.mem_size_per_task // num_new_classes

    def weights_to_num_samples(self, capacity, weights, min_samples_per_class):
        weights = weights / weights.sum()
        num_samples = (weights * capacity).long()
        num_samples[num_samples < min_samples_per_class] = min_samples_per_class

        return num_samples


    def sample_task(self, dataset:ContinualDataset, weights = None, min_samples_per_class = 1):

        self.set_transform(dataset.transform)
        num_samples_per_class_ = self.get_per_class_budget(len(self.data_), len(dataset.active_classes))

        if weights is None:
            weights = torch.ones(len(self.data_) + len(dataset.active_classes))

        memory_budget = num_samples_per_class_ * (len(self.data_) + len(dataset.active_classes))

        num_samples_per_class = self.weights_to_num_samples(memory_budget, weights, min_samples_per_class)

        i = 0

        for c in range(len(self.data_)):
            self.data_[c] = self.data_[c][:num_samples_per_class[i]]
            i += 1

        for c in range (len(dataset.active_classes)):
            class_idx = dataset.active_classes[c]

            size = len(dataset.data[class_idx])
            indices = torch.randperm(size)[:num_samples_per_class[i]]
            self.data_.append(dataset.data[class_idx][indices])

            i += 1

        self.data = self.data_

        self.set_active_classes(None)


