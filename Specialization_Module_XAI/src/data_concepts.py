import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Table 1: Concept definitions (digit class membership)
CONCEPT_MAP = {
    'loop':             {'pos': [0, 6, 8, 9],         'neg': [1, 2, 3, 4, 5, 7]},
    'vertical_stroke':  {'pos': [1, 4, 7, 9],         'neg': [0, 2, 3, 5, 6, 8]},
    'horizontal_stroke':{'pos': [2, 4, 5, 7],         'neg': [0, 1, 3, 6, 8, 9]},
    'curvature':        {'pos': [0, 2, 3, 5, 6, 8, 9],'neg': [1, 4, 7]},
    'intersection':     {'pos': [4, 8, 9],            'neg': [0, 1, 2, 3, 5, 6, 7]},
}

MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

def get_mnist_loaders(batch_size=128, data_dir='./data'):
    """Create standard MNIST train/test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=0))

def get_concept_subsets(dataset, concept_name, max_per_class=500):
    """Create stratified positive/negative subsets for a concept."""
    pos_labels = CONCEPT_MAP[concept_name]['pos']
    neg_labels = CONCEPT_MAP[concept_name]['neg']
    pos_idx, neg_idx = [], []
    class_counts_pos = {c: 0 for c in pos_labels}
    class_counts_neg = {c: 0 for c in neg_labels}
    for i, (_, y) in enumerate(dataset):
        y = int(y)
        if y in pos_labels and class_counts_pos[y] < max_per_class:
            pos_idx.append(i); class_counts_pos[y] += 1
        elif y in neg_labels and class_counts_neg[y] < max_per_class:
            neg_idx.append(i); class_counts_neg[y] += 1
    return Subset(dataset, pos_idx), Subset(dataset, neg_idx)
