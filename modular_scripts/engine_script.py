""" Contains functions for training and testing a model """

from typing import Tuple, Dict, List
import torch
from torch import nn
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

## Train Mode Function
def train_mode(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device) -> Tuple[float, float]:

  model.train()

  # Setting up evaluation metrics
  train_loss, train_acc = 0,0

  # Loop through dataloader databatch
  for batch, (X,y) in enumerate(dataloader):

    # Sending data to target device
    X,y = X.to(device), y.to(device)

    ## 1. Performing forward pass
    y_pred = model(X)

    ## 2. Loss calculation and accumulate to train_loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    ## 3. Optimizer zero grad
    optimizer.zero_grad()

    ## 4. Backward loss propagarion
    loss.backward()

    ## 5. Optimization
    optimizer.step()

    ## Calculating accuracy
    y_pred_class = y_pred.argmax(dim=1)
    train_acc += (y_pred_class==y).sum().item() / len(y_pred)

  ## Adjusting metrics for avg loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


## Testing loop
def test_mode(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):

  # Eval mode
  model.eval()

  # Test loss and tess accuracy metrics values
  test_loss, test_acc = 0,0

  # Inference mode
  with torch.inference_mode():

    ## Loop through dl
    for batch, (X, y) in enumerate(dataloader):

      # Data to target device
      X, y = X.to(device), y.to(device)

      ## Forward Pass
      test_pred = model(X)

      ## Loss
      loss = loss_fn(test_pred, y)
      test_loss += loss.item()

      # Accuracy
      test_pred_labels = test_pred.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

  ## Adjusting metrics
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

## Combining both to train step
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module=nn.CrossEntropyLoss(),
          epochs: int=5,
          device=device):

  ## Creating an empty dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  # Looping through the train and tetsing steps
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_mode(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)

    test_loss, test_acc = test_mode(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    ## Printing out whats goin on
    print(f"Epoch: {epoch} | Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    ## Results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
