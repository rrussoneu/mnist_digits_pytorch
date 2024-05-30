import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

'''
A base convolutional neural network for digit classification
'''
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50) # fully connected layer
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # relu applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

'''
A convolutional neural network for digit classification 
similar to the base model but including an extra dropout layer
'''
class NeuralNet_Extra_Dropout(nn.Module):
    def __init__(self):
        super(NeuralNet_Extra_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout1d() # added droupout layer after fc1
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # relu applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.relu(self.fc1_drop(self.fc1(x))) # apply added dropout layer after fully connected
        x = self.fc2(x)
        return F.log_softmax(x)

'''
A convolutional neural network for digit classification similar to the base model 
but using a leaky relu activation function 
'''
class NeuralNet_Leaky(nn.Module):
    def __init__(self):
        super(NeuralNet_Leaky, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # activation applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

'''
A convolutional neural network for digit classification similar to the base model 
but using a leaky relu activation function including an extra dropout layer
'''
class NeuralNet_Leaky_Extra_Dropout(nn.Module):
    def __init__(self):
        super(NeuralNet_Leaky_Extra_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout1d() # added dropout layer
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # activation applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.leaky_relu(self.fc1_drop(self.fc1(x))) # use added dropout layer after fully connected layer
        x = self.fc2(x)
        return F.log_softmax(x)

'''
A convolutional neural network for digit classification similar to the base model 
but using a log sigmoid activation function
'''
class NeuralNet_LS(nn.Module):
    def __init__(self):
        super(NeuralNet_LS, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50) # fully connected layer
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    def forward(self, x):
        x = F.logsigmoid(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.logsigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # relu applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.logsigmoid(self.fc1(x)) 
        x = self.fc2(x)
        return F.log_softmax(x)

'''
A convolutional neural network for digit classification similar to the base model 
but using a log sigmoid activation function including an extra dropout layer
'''
class NeuralNet_LS_Extra_Dropout(nn.Module):
    def __init__(self):
        super(NeuralNet_LS_Extra_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 5x5 filters
        self.conv2_drop = nn.Dropout2d() # dropout layer - default percent is 50%
        self.flatten = nn.Flatten() # flattening layer
        self.fc1 = nn.Linear(320, 50) # fully connected layer
        self.fc1_dropout = nn.Dropout1d() # added dropout layer
        self.fc2 = nn.Linear(50, 10) # 10 for 0-9

    def forward(self, x):
        x = F.logsigmoid(F.max_pool2d(self.conv1(x), 2)) # max pool for convolution layer 1
        x = F.logsigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # relu applied after 50% droupout
        x = x.view(-1, 320) # changes input dimensionlity 
        x = self.flatten(x) # flatten
        x = F.logsigmoid(self.fc1_dropout(self.fc1(x)))
        x = self.fc2(x)
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
device - device to run on
'''
def train(epoch, optimizer, train_loader, network, train_losses, train_counter, log_interval, save: bool, iteration: str, device):
    network.train()
    print('Train Epoch: {}'.format(epoch))

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        output = output.to(device)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            if save == True:
                save_path = "results/experiment_models/model_" + iteration + ".pth"
                save_path_optimizer = "results/experiment_optimizers/optimizer_" + iteration + ".pth"
                torch.save(network.state_dict(), save_path)
                torch.save(optimizer.state_dict(), save_path_optimizer)

'''
Tests a CNN model
test_loader - test data loader
network - the model 
test_losses - list for losses
device - device to run on
'''
def test(test_loader, network, test_losses, device):
    network.eval() # eval mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            output = output.to(device)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1] # target with greatest likelihood
            correct += pred.eq(target.data.view_as(pred)).sum() # storing accuracy 
    test_loss /= len(test_loader.dataset) # avg loss
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

'''
    Trains and tests model based on network type and parameter iteration
    netowrk_type - which version to use, types in dictionary in first line of function
    batch_size - the training batch size
    num_epochs - the number of epochs to run
    dropout - percent dropout for dropout layer after second convultional layer (1,2, or 3 are inputs 
                that are multiplied by .25 for either 25,50,75%)
    device - device to run on
'''
def run_model(network_type:int, batch_size: int, num_epochs: int, dropout: int, device):
  # one net for each i
    networks = {1: NeuralNet(), 2: NeuralNet_Extra_Dropout(), 
              3: NeuralNet_Leaky(), 4: NeuralNet_Leaky_Extra_Dropout(), 
              5: NeuralNet_LS(), 6: NeuralNet_LS_Extra_Dropout()}

    network = networks[network_type] # set network type
    
    network.conv2_drop = nn.Dropout2d(.25*dropout) # dropout layer - default percent is 50%

    network = network.to(device) # push network to device
    #network.cuda()

    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                    momentum=momentum)

    # loaders based on j for batch size
    batch_size_train = batch_size
    batch_size_test = 1000

    train_data = torchvision.datasets.MNIST('data', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))
                                  ]))
    train_data.train_data.to(device)
    train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=batch_size_train, shuffle=True)

    test_data = torchvision.datasets.MNIST('data', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ]))
    test_data.test_data.to(device)

    test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=batch_size_test, shuffle=True)
    # epochs based on kq
    train_losses = []
    train_counter = []
    test_losses = []
    print("l:")
    print(len(train_loader.dataset))
    test_counter = [i*len(train_loader.dataset) for i in range((num_epochs) + 1)]

    iter_name = "Network" + str(network_type) + "Dropout" + str(dropout) + "Batch_Size" + str(batch_size) + "Num_Epochs" + str(num_epochs)

    loss_percent = 0
    
    loss_percent = test(test_loader=test_loader, network=network, test_losses=test_losses, device=device)
    for epoch in range(1, (num_epochs) + 1):
        train(epoch=epoch, optimizer=optimizer, train_loader=train_loader, network=network, train_losses=train_losses, train_counter=train_counter, log_interval=log_interval, save=True, iteration=iter_name, device=device)
        loss_percent = test(test_loader=test_loader, network=network, test_losses=test_losses, device=device)
    
    # save the iteration and loss info after training to a csv for comparison
    with open('net_data.csv', 'a+') as csv_file: # create file if it doesn't exist
        fields = [str(network_type), str(dropout), str(batch_size), str(num_epochs), str(loss_percent)]
        w = csv.writer(csv_file)
        w.writerow(fields)
    
    print(test_counter, test_losses)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='green', zorder=1)
    plt.scatter(test_counter, test_losses, color='purple', zorder=2)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    save_path = "figures/" + iter_name + ".png" 
    plt.savefig(save_path)

# Runs a variety of networks and saves their final accuracy results to a csv file 
def main():
    random_seed = 1
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available
        else "cpu"
    )        
    
    device = torch.device(device=device)
    print(f"Using {device}")
    
    torch.cuda.manual_seed(random_seed)
    for i in range(1,7): # three different activations functions used and added droupout layer version of each 
        for p in range(1,4): # different dropout rates for convultion layer dropout
            for j in range(48, 73, 12): #  batch sizes  48, 60, 72
                for k in range(3, 8, 2): # 3, 5, 7 epochs
                    itr_out = str(i) + " " + str(p) + " " + str(j) + " " + str(k)
                    print("iteration_num: " + itr_out)
                    run_model(i,j,k,p, device)


if __name__ == "__main__":
    main()