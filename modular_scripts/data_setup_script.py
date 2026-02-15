"""
Consists of functionality for DataLoaders for classifying image
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS):
  """
    Creates training and test dataloaders by taking in a train and test directory, turns them to datasets using ImageFolder and then to PyTorch DataLoaders
  Args:
    train_dir and test_dir are paths for train and test directories
    transforms on train and test data
    batch_size means number of samples per batch in each of DataLoaders
    num_workers means num of workers per dataloader
  Returns:
     Returns a tuple of train_dataloader, test_dataloader, class_names (list of target classes)
  """

  # ImageFolder for creating datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(root=test_dir, transform=transform)

  # Class names
  class_names = train_data.classes

  # Images to dataloaders
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)

  test_dataloader = DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=False,
                               pin_memory=True)

  return train_dataloader, test_dataloader, class_names
