# initialize lists to store loss and accuracy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import random

device = 'mps'

class LeakySurrogate(nn.Module):
    def __init__(self, beta, z=1, threshold=1.0):
        super(LeakySurrogate, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_op = self.SpikeOperator.apply
        self.z = z
        self.mem = None

    # the forward function is called each time we call Leaky
    def forward(self, input_):
        spk = self.spike_op(self.mem - self.threshold, self.z)  # call the Heaviside function
        reset = (spk * self.threshold).detach() # removes spike_op gradient from reset
        self.mem = self.beta * self.mem + input_ - reset
        return spk

    # forward pass: Heaviside function
    @staticmethod
    class SpikeOperator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, z):
            """
            In the forward pass we compute a step function of the input Tensor
            and return it. ctx is a context object that we use to stash information which
            we need to later backpropagate our error signals. To achieve this we use the
            ctx.save_for_backward method.
            """
            ctx.save_for_backward(input)
            ctx.z = z
            spk = torch.zeros_like(input)
            spk[input > 0] = 1.0
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor we need to compute the
            surrogate gradient of the loss with respect to the input.
            Here we use the fast Sigmoid function with z = 1.
            """
            input, = ctx.saved_tensors
            z = ctx.z
            grad_input = grad_output.clone()
            # TODO: add your implementation here.
            # TJP - grad = torch.sigmoid(z * input) * (1 - torch.sigmoid(z * input)) * grad_input # 89%acc
            grad = (
                grad_input * z * torch.exp(-z * input)
                / ((1 + torch.exp(-z * input)) ** 2)
            )
            return grad, None


class SNN(nn.Module):
    def __init__(self, T, beta=0.8, z=1, threshold=1.0):
        super(SNN, self).__init__()
        self.T = T
        self.flatten = nn.Flatten()
        # 1st fully-connected layer
        self.fc1 = nn.Linear(28 * 28, 10)
        self.lif1 = LeakySurrogate(beta=beta, z=z, threshold=threshold)
        # 2nd fully-connected layer
        self.fc2 = nn.Linear(10, 10)
        # output layer neurons, whose firing rate will be served as the final prediction
        self.lif2 = LeakySurrogate(beta=beta, z=z, threshold=threshold)

    def init_mem(self, batch_size, feature_num):
        return nn.init.kaiming_uniform_(torch.empty(batch_size, feature_num)).to(device)

    # define the forward pass
    def forward(self, input_):
        self.lif1.mem = self.init_mem(input_.shape[1], 10)
        self.lif2.mem = self.init_mem(input_.shape[1], 10)

        output_spikes = 0
        for t in range(self.T):
            x = input_[t]
            x = self.flatten(x)
            x = self.fc1(x)
            spk1 = self.lif1(x)
            x = self.fc2(spk1)
            spk2 = self.lif2(x)
            output_spikes = output_spikes + spk2

        return output_spikes / self.T
    

### DATA METHODS

# mode 1: repeatedly pass the same original training sample to the network
def gen_train_data_img(x, T=50):
    res = []
    for t in range(T):
        res.append(x)
    return torch.stack(res)

# mode 2: compare each pixel to a fixed threshold and repeatedly pass the same input to the network
def threshold_encoder(x, threshold=0.5):
    # insert your code here
    return (x > threshold).to(x).float() # make same type as x, binary, 0.0 or 1.0


def gen_spike_data_static(x, T=50):
    res = []
    for t in range(T):
        encoded_img = threshold_encoder(x)
        res.append(encoded_img)
    return torch.stack(res)

# mode 3: each input feature is used as the probability a spike occurs at any given time step
def gen_spike_data_bernoulli(x, T=50):
    res = []
    for t in range(T):
        # TODO: add your implementation here.
        encoded_img = torch.bernoulli(x) # binary encoded image
        res.append(encoded_img)
    return torch.stack(res)


def get_dataloaders():
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    # test dataset
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    return train_loader, test_loader


### TRAINING
 

def train_snn(model, train_loader, criterion, optimizer, epoch, T=50):
    # set the model to training mode
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", leave=False)
    for batch_idx, (data, target) in tqdm(pbar):
        # clear the gradients
        optimizer.zero_grad()
        # move data to GPU
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target, 10).float() # convert to one-hot to match requirement for MSE loss
        output_fr = model(gen_spike_data_bernoulli(data, T))
        # compute the loss
        loss = criterion(output_fr, target_onehot)
        loss.backward()
        # update the weights
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({
            'Batch': batch_idx,
            'Loss': loss.item(),
            'Avg Loss': running_loss / (batch_idx + 1)
        })
        # if batch_idx % 100 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # compute average loss in each epoch
    average_loss = running_loss / len(train_loader)
    return average_loss

    
def test_snn(model, test_loader, criterion, T=50):
    # set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0

    # disable gradient calculation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_onehot = F.one_hot(target, 10).float()
            output_fr = model(gen_spike_data_bernoulli(data, T))
            # sum up batch loss
            test_loss += criterion(output_fr, target_onehot).item()
            # get the index of the max log-probability as the predicted category
            pred = output_fr.argmax(dim=1, keepdim=True)
            # compute correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # average test loss
    test_loss /= len(test_loader.dataset)
    # accuracy percentage
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'-> Test: Average loss: {test_loss:.4f}, Accuracy: {correct} / {len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy, test_loss


def train_eval(model, train_loader, test_loader, epochs, T):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # reinit model, we can create optimizer and criterion here
    train_losses_snn = []
    test_losses_snn = []
    test_accuracies_snn = []

    # train and test the model
    for epoch in range(epochs):
        train_loss = train_snn(model, train_loader, criterion, optimizer, epoch, T)
        train_losses_snn.append(train_loss)
        accuracy, test_loss = test_snn(model, test_loader, criterion, T)
        test_losses_snn.append(test_loss)
        test_accuracies_snn.append(accuracy)
    return train_losses_snn, test_losses_snn, test_accuracies_snn


def set_seed(seed):
    random.seed(seed)  # Set seed for random module
    np.random.seed(seed)  # Set seed for numpy
    torch.manual_seed(seed)  # Set seed for torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for torch if using GPU
        torch.cuda.manual_seed_all(seed)


def create_model(exp_name, hyperparam):
    set_seed(0)
    T = 5
    if "z" in exp_name:
        model = SNN(T, z=hyperparam).to(device)
    elif "beta" in exp_name:
        model = SNN(T, beta=hyperparam).to(device)
    elif "threshold" in exp_name:
        model = SNN(T, threshold=hyperparam).to(device)
    elif "timesteps" in exp_name:
        model = SNN(hyperparam).to(device)
        T = hyperparam
    return model, T


def run_trial(exp_name, hyperparam_list, train_loader, test_loader):
    df_results = pd.DataFrame(columns=['hyperparam', 'train_losses', 'test_losses', 'test_accuracies'])
    epochs = 10
    for hyperparam in hyperparam_list:
        model, T = create_model(exp_name, hyperparam)
        train, test, acc = train_eval(model, train_loader, test_loader, epochs, T)
        df_results.loc[len(df_results)] = [hyperparam, 
                                        str(train),
                                        str(test),
                                        str(acc)]
    
    filename = f"results/{exp_name}_results.csv"
    df_results.to_csv(filename, index=False)


### HYPER PARAM SEARCH


def z_tuning():
    train_loader, test_loader = get_dataloaders()
    exp_name = "z"
    z_values = [0.5, 1, 2, 5, 10] # next time
    run_trial(exp_name, z_values, train_loader, test_loader)


def beta_tuning():
    beta_values = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    train_loader, test_loader = get_dataloaders()
    exp_name = "beta"
    run_trial(exp_name, beta_values, train_loader, test_loader)


def T_tuning():
    T_values = [5, 10, 20, 50, 100]
    train_loader, test_loader = get_dataloaders()
    exp_name = "timesteps"
    run_trial(exp_name, T_values, train_loader, test_loader)


def thresh_tuning():
    threshold_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    train_loader, test_loader = get_dataloaders()
    exp_name = "threshold"
    run_trial(exp_name, threshold_values, train_loader, test_loader)

if __name__ == "__main__":
    T_tuning()
    thresh_tuning()
    beta_tuning()
    z_tuning()