# Creates, trains, and tests a simple CNN on MNIST digits data set

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

'''
A class for a simple CNN
'''
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    # Executes a foward pass of the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # relu applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.relu(self.fc1(x))  # activation after linear layer
        x = self.fc2(x) # final linear layer to 10 output nodes
        return F.log_softmax(x)

'''
Trains a model/CNN
epoch - epoch number
optimizer - model's optimizer
train_loader - data loader for training set
network - the model
train_losses - list for loss values, updated at log intervals
train_counter - list to count log intervals
log_intervals - interval of when to log/ save values
save - whether to save the model and optimizer to a file or not
'''
def train(epoch, optimizer, train_loader, network, train_losses, train_counter, log_interval, save):
  network.train() # model is training
  for batch_idx, (data, target) in enumerate(train_loader): # iterate through training dataloader 

    optimizer.zero_grad()
    output = network(data) # run data through model
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    
    # at log interval
    if batch_idx % log_interval == 0:
      # print info about interval
      # format with epoch number, how far through data, and loss
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

      # append loss and counter values
      train_losses.append(loss.item()) 

      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

      if save == True: # save info if save is True
        torch.save(network.state_dict(), 'results/model2.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer2.pth')


'''
Tests a CNN model
test_loader - test data loader
network - the model 
test_losses - list for losses
'''
def test(test_loader, network, test_losses):
  network.eval() # need eval mode
  test_loss = 0 # keep track of how the network is doing
  correct = 0

  # iterate through and test the data without changing the network
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data) # run data through
      test_loss += F.nll_loss(output, target, size_average=False).item() # get the loss
      pred = output.data.max(1, keepdim=True)[1] # target with greatest likelihood
      correct += pred.eq(target.data.view_as(pred)).sum() # storing accuracy 
  test_loss /= len(test_loader.dataset) # avg loss
  test_losses.append(test_loss) # append the loss
  # print out loss info
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# Displays the first 9 digits of the test set then trains the network and tests it
def main(argv):

    network = NeuralNet() # init model

    
    
    # define params
    n_epochs = 15 # number of epochs to train with
    batch_size_train = 128 # batch size for training
    batch_size_test = 1000 # batch size for testing
    learning_rate = 0.001
    momentum = 0.5
    
    log_interval = 10

    random_seed = 1 # use the same random seed for each iteration for consistency
    torch.backends.cudnn.enabled = False # not running on GPU
    torch.manual_seed(random_seed)

    # init training data loader 
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.FashionMNIST('data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
      batch_size=batch_size_train, shuffle=True)

    # init testing data loader 
    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
      batch_size=batch_size_test, shuffle=True)


    # Display the first 6 test images

    test_dataset = torchvision.datasets.FashionMNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    figure_0 = plt.figure(figsize=(8,8))
    cols, rows = 3, 2
    for i in range(1, cols * rows + 1):
        sample_idx = i
        img, label = test_dataset[sample_idx]
        figure_0.add_subplot(rows, cols, i)
        plt.title(i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.show()

    # run network

    # init optimizer for network given our parameters 
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    # lists for loss info
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # run training
    test(test_loader=test_loader, network=network, test_losses=test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch=epoch, optimizer=optimizer, train_loader=train_loader, network=network, train_losses=train_losses, train_counter=train_counter, log_interval=log_interval, save=True)
        test(test_loader=test_loader, network=network, test_losses=test_losses)

    # create a plot of the results and show it
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='green')
    plt.scatter(test_counter, test_losses, color='purple')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    plt.show()

if __name__ == "__main__":
    main(sys.argv)
