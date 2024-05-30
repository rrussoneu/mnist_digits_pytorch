# examines the model created in main.py 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from main import NeuralNet
from torchvision import datasets
import cv2

from torchvision.transforms import ToTensor

# Examines the filters generated 
def main():
    network = NeuralNet() # init network
    network.load_state_dict(torch.load("results/model.pth")) # load previously trained model

    # get training data loader 
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data', train=True, download=False,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
      batch_size=1, shuffle=False)

    transformed_images = [] # array for filtered images
    filters = [] # array for filters

    # get the filters from first layer of CNN and apply them to the first training image
    with torch.no_grad():
        iterations = 0 # going to stop after the first image
        for data, target in train_loader:
            data_arr = data[0].numpy() # get data as numpy array
            for i in range(0, 10): # get the 10 filters and apply them

                # get filter from network and convert to numpy array
                filter = network.conv1.weight[i,0]
                kernel = filter.numpy()
                filters.append(kernel) # append filter to array of filters for display
                img = cv2.filter2D(data_arr[0], ddepth=-1, kernel=kernel) # use opencv to apply filter to image
                transformed_images.append(img) # store image in array for display
            
            # stop after first image
            iterations += 1
            if iterations > 0: 
                break

    # uncomment to see just the filters
    '''
    figure = plt.figure(figsize=(8,8)) # figure 1 for visualizing the filters by themselves
    cols, rows = 4, 3
    for i in range(1, cols*rows + 1 - 2): # add each filter to figure
        label = "Filter: " + str(i - 1)
        filter_img = filters[i - 1]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(filter_img.squeeze(), cmap="magma")
    '''
    
    
    figure_2 = plt.figure(figsize=(8,8)) # figure for filters and the transformed image
    cols_2, rows_2 = 4, 5
    for i in range(1, 21, 2): # get filter and transformed image, stepping by 2 to index arrays
        figure_2.add_subplot(rows_2, cols_2, i)
        curr_filter_image = filters[i // 2] # add current filter visualization to the figure
        plt.axis("off")
        plt.imshow(curr_filter_image.squeeze(), cmap="gray")
        curr_tfn_image = transformed_images[i//2]
        figure_2.add_subplot(rows_2, cols_2, i + 1) # put the transformed image next to the filter
        plt.axis("off")
        plt.imshow(curr_tfn_image.squeeze(), cmap="gray")

    # cleaner plots
    plt.xticks([])
    plt.yticks([])


    plt.show() # show figures

    
if __name__ == "__main__":
    main()