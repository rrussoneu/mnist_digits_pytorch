# Evaluates model results on hand drawn digits

import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from main import test, NeuralNet
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

'''
    A class for a pytorch transformation on images of hand drawn digits
    The digits are sized to 28x28 and have colors inverted to match MNIST data set already
'''
class DrawnImageTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        return x

# Tests the model created in main.py with the MNIST test data and hand drawn images
def main():
    network = NeuralNet() # init network
    network.load_state_dict(torch.load("results/model.pth")) # load model
    network.eval() # set to eval mode


    # using batch size 1 and only testing 10 images
    batch_size_test = 1

    # load the test data
    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
      batch_size=batch_size_test, shuffle=False)

    # track accuracy
    test_loss = 0
    correct = 0

    preds = []
    data_predicted = []

    # test the trained model
    with torch.no_grad():
        iterations = 0
        for data, target in test_loader:
            output = network(data)
            data_predicted.append(data[0])
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1] # target with greatest likelihood
            correct += pred.eq(target.data.view_as(pred)).sum() # storing accuracy 
            preds.append(pred.item())
            print(f'Test Iteration: {iterations}, Outputs: {output.data}, Prediction: {pred.item()}')
            iterations += 1
            if iterations >= 10: # only do the first 10 images
                break

   
    # visualize results on test data 
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img = data_predicted[i - 1]
        figure.add_subplot(rows, cols, i)
        plt.title(preds[i-1])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    
    # Drawn images
    drawn_preds = [] # predictions
    drawn_data_predicted = [] # image data

    # transformation for the images
    image_transform = torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  DrawnImageTransform(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
    
    drawn_data = torchvision.datasets.ImageFolder('hand_drawn', transform=image_transform) # image data
    drawn_data_loader = torch.utils.data.DataLoader(drawn_data, batch_size=1, shuffle=False) # image data loader using the hand drawn data

    # predict the hand drawn digits
    with torch.no_grad():
        for data, target in drawn_data_loader:
            output = network(data)
            drawn_data_predicted.append(data[0])
            pred = output.data.max(1, keepdim=True)[1] # target with greatest likelihood
            drawn_preds.append(pred.item())
            print(pred.item())
            
        
    # visualize the results
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img = drawn_data_predicted[i - 1]
        figure.add_subplot(rows, cols, i)
        plt.title(drawn_preds[i-1])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()
        
        


if __name__ == "__main__":
    main()
