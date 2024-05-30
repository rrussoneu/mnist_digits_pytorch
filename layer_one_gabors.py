# A script to test the model created in main.py with Gabor filters as the first layer

import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import NeuralNet, train, test
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from evaluate import DrawnImageTransform



# Train and test network with constant first layer using gabor filters
def main():
    network = NeuralNet() # init network
    
    # Set up gabor filters to be used
    # 10 5x5 filters to stay consistent with previous version of networks first convolutional layer
    gabor_kernel_size = 5
    gabor_sigma = 5
    theta_range = np.arange(0, np.pi, np.pi / 10)
    gabor_frequency = 0.3
    gabor_phase = 0

    kernels = [] # store the gabor kernels here 

    for theta in theta_range:
        kernel = cv2.getGaborKernel(ksize=(gabor_kernel_size, gabor_kernel_size), sigma=gabor_sigma, theta=theta, lambd=gabor_frequency, gamma=gabor_phase)
        kernels.append(kernel)
        # uncomment to visualize kernels, gray cmap may look a bit better
        #plt.imshow(kernel, cmap='magma')
        #plt.title(str(theta))
        #plt.show()
   
    
    #print(network.conv1.weight.shape)
    #print(network.conv1.weight)

    kernels = np.array(kernels) # convert kernels to numpy array
    kernel_tensor = torch.from_numpy(kernels.astype(np.float32)).unsqueeze(1) # convert to tensor, need to specify data type to avoid errors
    #print(kernel_tensor.type)
    kernel_tensor.to('mps') # my macbook uses mps so need to send the tensor to device
    #print(kernel_tensor.shape)
    kernel_tensor.requires_grad = False # layer won't change
    new_conv1 = torch.nn.Conv2d(1, 10, kernel_size=5) # define new convolution layer
    # assign the gabor filters to the layer
    with torch.no_grad():
        new_conv1.weight = torch.nn.Parameter(kernel_tensor)
        
    #print(kernel_tensor)
    network.conv1 = new_conv1 # assign as conv1 in network

    network.conv1.weight.requires_grad = False # layer won't change, this line may be redundant 
    #print(network.conv1.weight)
    #print(network.conv1.weight.shape)

    
    # params for network
    n_epochs = 5 # number of epochs to train with
    batch_size_train = 64 # batch size for training
    batch_size_test = 1000 # batch size for testing
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1 # keep random seed the same for consistency
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
        train(epoch=epoch, optimizer=optimizer, train_loader=train_loader, network=network, train_losses=train_losses, train_counter=train_counter, log_interval=log_interval, save=False)
        test(test_loader=test_loader, network=network, test_losses=test_losses)

    # create a plot of the results and show it
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='green')
    plt.scatter(test_counter, test_losses, color='purple')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    plt.show()

    #print(network.conv1.weight)


    # Testing the results on the hand drawn images from task 1

    network.eval() # set to eval mode


    # using batch size 1 and only testing 10 images
    batch_size_test = 1

    drawn_preds = []
    drawn_data_predicted = []

    # define the image transform using created class
    image_transform = torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  DrawnImageTransform(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

    # load the hand drawn data
    drawn_data = torchvision.datasets.ImageFolder('hand_drawn', transform=image_transform)
    drawn_data_loader = torch.utils.data.DataLoader(drawn_data, batch_size=1, shuffle=False)

    
    # predict numbers
    with torch.no_grad():
        for data, target in drawn_data_loader:
            output = network(data)
            drawn_data_predicted.append(data[0])
            pred = output.data.max(1, keepdim=True)[1] # target with greatest likelihood
            drawn_preds.append(pred.item())
            #print(pred.item())
        
    # visualie the numbers and their predictions
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = i
        img = drawn_data_predicted[i - 1]
        figure.add_subplot(rows, cols, i)
        plt.title(drawn_preds[i-1])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()

    return

if __name__ == "__main__":
    main()