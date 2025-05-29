import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import numpy as np
import shap
from torch.utils.data import DataLoader, Subset

np.random.seed(0)
torch.manual_seed(0)

# Define device and model parameters
device = 'cuda'  # Options: 'cpu', 'cuda' (for GPU)
batch_size = 200
epochs = 5

# Define the neural network model, using a simple MLP for MNIST
class MLP_MNIST(torch.nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from typing_extensions import Self

# added softmax
# converts them into soft probablity curve


class MLP_MNIST_Prob(torch.nn.Module):
    def __init__(self, base_model):
        super(MLP_MNIST_Prob, self).__init__()
        self.base_model = base_model  # your original MLP_MNIST

    def forward(self, x):
        logits = self.base_model(x)
        probs = F.softmax(logits, dim=1)  # convert to probabilities
        return probs

import torch.nn.functional as F
model = MLP_MNIST().to(device)
model_probs = MLP_MNIST_Prob(model).to(device)  # Wrap original trained model

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model (simplified training loop with logs for demonstration)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # Summing loss for averaging later
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}")

# Select a subset of data for Shapley value computation
background_data = Subset(train_dataset, indices=np.random.choice(len(train_dataset), 1000, replace=False))
background_loader = DataLoader(background_data, batch_size=1, shuffle=True)

# Create a background dataset for Deep SHAP
background_samples = torch.cat([data[0].unsqueeze(0) for data, _ in background_loader], 0).to(device)




# Initialize SHAP Deep Explainer
e = shap.DeepExplainer(model_probs, background_samples)


data_point, _ = next(iter(test_loader))  # Get a batch from the test loader
data_point = data_point.to(device)       # move data to the appropriate device
data_point = data_point[0:1]             # Select the first example from the batch

# Option to choose between DeepExplainer and GradientExplainer
try:
    e = shap.DeepExplainer(model_probs, background_samples)
    shap_values = e.shap_values(data_point)
except AssertionError as error:
    print("Switching to GradientExplainer due to error:", error)
    e = shap.GradientExplainer(model, background_samples)
    shap_values = e.shap_values(data_point)


# Adjust the shape of the data point for visualization
data_point = data_point.squeeze()  #

# ensure data_point is reshaped correctly for grayscale visualization
if data_point.ndim == 3:
    data_point = data_point.permute(1, 2, 0)  # Convert from [C, H, W] to [H, W, C] if needed

# Check and adjust dimensions for shap_values if necessary
shap_values = [sv.squeeze() for sv in shap_values]  # Remove unnecessary batch dimension

shap.image_plot(shap_values, -data_point.cpu().numpy())

import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)

class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model, loss, optimizer
model = MLP_MNIST().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transform, download=True)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         transform=transform, download=True)




batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Just pick a small subset of validation data for the demonstration
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
val_data_iter = iter(val_loader)
# chosing the validation samples in the dataset
val_img, val_label = next(val_data_iter)
val_img, val_label = val_img.to(device), val_label.to(device)

data_shapley_scores = torch.zeros(len(train_dataset), device=device) # initiate an empty array


epochs = 1
global_step= 0

# ------------------------------------------
# 4. Training Loop with In-Run Data Shapley
# ------------------------------------------
model.train()
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)


        # We create a combined batch:
        #   1) the normal training batch
        #   2) plus the single validation sample
        # to do "ghost dot-products" in ONE pass.
        # creates a 1D array to combine the images and validation set hence making it ghost dot product

        combined_imgs = torch.cat([images, val_img], dim=0)
        combined_labels = torch.cat([labels, val_label], dim=0)

        # Compute the loss on the combined set
        logits = model(combined_imgs) #
        train_batch_size = images.size(0)

        # The first part of logits is training data, the last part is for val data
        loss_train = loss_fn(logits[:train_batch_size], labels)
        loss_val   = loss_fn(logits[train_batch_size:], val_label)

        # Combined loss (we sum them so that backprop will let us
        # access both training-sample gradients and val-sample gradient)
        combined_loss = loss_train + loss_val



        # Ghost dot product for backward pass
        optimizer.zero_grad()
        # retain graph since we need more of the backprop later on
        combined_loss.backward(retain_graph=True)

        saved_state = {} # save the current state


        # We store the current gradients (which are for train+val) in saved_state.
        for name, param in model.named_parameters():
            if param.grad is not None:
                # save param.grad clone for later use
                saved_state[name] = param.grad.detach().clone()



        # zero the model gradient
        # only use val for back/forward pass
        optimizer.zero_grad()
        logits_val_only = model(val_img)
        loss_val_only = loss_fn(logits_val_only, val_label)
        # single-sample gradient
        loss_val_only.backward(create_graph=False)
        grad_val = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_val[name] = param.grad.detach().clone()



        # Now let's compute the "dot product" for each sample in the train batch:
        for i in range(train_batch_size):
            # zero grad
            param_zero = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.zero_()

            # backward on just the i-th example in the train batch
            single_logit = model(images[i].unsqueeze(0))
            single_loss = loss_fn(single_logit, labels[i].unsqueeze(0))
            single_loss.backward()

            # read off dot-product with grad_val
            dot_val = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_val:
                    # flatten both param.grad and grad_val[name]
                    dot_val += (param.grad.view(-1) * grad_val[name].view(-1)).sum()

            # Add to data_shapley_scores
            idx_in_full_dataset = batch_idx*batch_size + i  # approximate global index
            data_shapley_scores[idx_in_full_dataset] += -0.01 * dot_val.item() # mul by lr to scale back things


            # We used lr=0.01 -> so multiply by -lr. Negative sign because the loss change is -( grad_val · grad_train_i ).



        # restore the combined grad from saved_state
        for name, param in model.named_parameters():
            if name in saved_state:
                param.grad = saved_state[name] # now when it is in param save it to saved_state

        # Final logging
        optimizer.step()

        global_step += 1
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, combined_loss = {combined_loss.item():.4f}")


print("In-Run Data Shapley (first-order approx) computed for each training sample.")
print("data_shapley_scores shape:", data_shapley_scores.shape)

# Maybe some sorting


import torch

sorted_shite = torch.argsort(data_shapley_scores)

print("HIghest shapley scores")
for rank in range(10):
    idx = sorted_shite[rank].item()        # index in the dataset
    val = data_shapley_scores[idx].item()    # Shapley score
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Score: {val:.4f}")



print("Lowest shapley scores")
for rank in range(10):
    idx = sorted_shite[-(rank+1)].item()   # index from the end
    val = data_shapley_scores[idx].item()    # Shapley score
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Score: {val:.4f}")




