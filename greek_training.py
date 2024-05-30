# Uses previously trained model for Greek letter transfer learning 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from main import NeuralNet, train, test
from torchvision import datasets
import cv2

'''
    A class for preparing Greek letter image data
'''
class GreekTransform():
    def __init__(self):
        pass
    
    # runs transformations on image data
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x) # convert to greyscale
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 ) # handles any orientation issues
        x = torchvision.transforms.functional.center_crop( x, (28, 28) ) # crops the center
        return torchvision.transforms.functional.invert( x ) # invert the colors and return for white on black images

# Executes transfer learning with greek letters 
def main():

    network = NeuralNet() # init network
    network.load_state_dict(torch.load("results/model.pth")) # load model 

    for param in network.parameters(): # freeze the parameters
        param.requires_grad = False
    
    print(network)
    network.fc2 = torch.nn.Linear(50, 3) # adjust the final layer to have three outputs
    print(network)

    
    # get the training data
    greek_train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder("greek_train/",
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                GreekTransform(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ])),
        batch_size = 5, shuffle=True,
    )

    random_seed = 1 # consistent random seec
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    n_epochs = 8 # uses 8 epochs
    batch_size_train = 1
    batch_size_test = 1
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    

    # run training like main.py
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(greek_train_loader.dataset) for i in range(n_epochs + 1)]

    test(test_loader=greek_train_loader, network=network, test_losses=test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch=epoch, optimizer=optimizer, train_loader=greek_train_loader, network=network, train_losses=train_losses, train_counter=train_counter, log_interval=log_interval, save=False)
        test(test_loader=greek_train_loader, network=network, test_losses=test_losses)

    # visualize the results
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='green')
    plt.scatter(test_counter, test_losses, color='purple')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    plt.show()

    # test on my drawn images

    drawn_preds = []
    drawn_data_predicted = []

    # loads the images I drew
    greek_test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder("drawn_greek/",
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                GreekTransform(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ])),
        batch_size = 1, shuffle=False,
    )

    # get predictions
    with torch.no_grad():
        for data, target in greek_test_loader:
            output = network(data)
            drawn_data_predicted.append(data[0])
            pred = output.data.max(1, keepdim=True)[1] # target with greatest likelihood
            drawn_preds.append(pred.item())
            print(pred.item())
            print(f'Outputs: {output.data}, Prediction: {pred.item()}')
        

    # visualize results       
    figure_drawn = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img = drawn_data_predicted[i - 1]
        figure_drawn.add_subplot(rows, cols, i)
        plt.title(drawn_preds[i-1])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()



if __name__ == "__main__":
    main()