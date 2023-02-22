#######################################################################
#   CNN with CIFAR10 Datasets
#######################################################################
#
#   @Class Name(s): NET2 Process
#
#   @Description:   Image Classification
#
#
#   @Note:  Image classification on dataset
#
#   Version 0.0.1:  NET2 Class
#                   06 Feb 2023 Monday, 17:30 PM - Hasan Berkant Ödevci
#
#
#
#   @Author(s): Hasan Berkant Ödevci
#
#   @Mail(s):   berkanttodevci@gmail.com
#
#   Created on 06 Feb 2023 Monday, 17:30 PM.
#
#
########################################################################
try:
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data.sampler import SubsetRandomSampler
except ImportError:
    print("Please check the library...!")

# Define Devices about GPU or CPU
if(torch.cuda.is_available()):
    device = "cuda"  
    print("Cuda is Available")
    
else:
    device = "cpu"
    print("Device is CPU")



# Integrate mean and std with using transform normalize to datasets
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                     (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root = "D:/Projects/Python/Deep_Learning/Pytorch/CNN_with_CIFAR10/Dataset/", download=True, train=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root = "D:/Projects/Python/Deep_Learning/Pytorch/CNN_with_CIFAR10/Dataset/", download=True, train=False, transform=transform)

# obtain training indices that will be used for validation
valid_size = 0.2

# Number of train_dataset
num_train = len(train_dataset)

# Indices
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

valid_loader = DataLoader(train_dataset, batch_size=128, sampler=valid_sampler)
train_dataLoader = DataLoader(train_dataset , batch_size= 128, sampler=train_sampler)
test_dataLoader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class NET2(nn.Module):
    def __init__(self):
        super(NET2,self).__init__()
        # Set Convolutional Layers
        self.CNN_layers = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            # Convolutional Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # Convolutional Layer 3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*4*4, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )
    # Defining Forward Pass
    def forward(self,x):
        x = self.CNN_layers(x)
        # Flatten
        x = x.view(x.size(0),-1)
        # Then pass it through the linear layer
        x = self.linear_layers(x)
        # Softmax Activation Function
        x = nn.functional.log_softmax(x, dim=1)
        return x
    
# Assign class into object
model = NET2()
model.to(device)

# Define Optimizer with learning rate
optimizer = optim.Adam(model.parameters(), lr = 0.001)
#This criterion computes the cross entropy loss between input logits and target.
criterion = nn.CrossEntropyLoss()

# Number of epoch
epochs = 20

# Train and Valid List
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

# Training Part
def train(model, epochs,optimizer,criterion):
    for epoch in range(epochs):
        # Prep model for training
        model.train()
        train_loss = 0
        train_correct = 0
        n_samples = 0

        for index, (image,label) in enumerate(train_dataLoader):
            images = image.to(device)
            labels = label.to(device)

            # Training Pass
            optimizer.zero_grad()

            # Forward Pass
            output = model(images)

            # Calculation entropy loss between prediction and labels
            loss = criterion(output,labels)

            # Backward pass
            loss.backward()

            # Update Parameters
            optimizer.step()

            # Calculation train loss
            train_loss += loss.item()

            # Find maximum value and index on dataset
            _,predicted = torch.max(output.data,1)

            # Sum n_sampler
            n_samples += labels.size(0)

            # If predicted values are equalt to labels, the number of correct values is gathered
            train_correct += (predicted == labels).sum().item()

            # Calculate train accuracy
            train_acc = train_correct/n_samples

        model.eval()
        valid_loss = 0
        n_samples = 0
        valid_correct = 0

        for image,label in valid_loader:
            images = image.to(device)
            labels = label.to(device)

            # Calculate output
            output = model(images)

            # Loss
            loss = criterion(output,labels)

            # valid loss
            valid_loss += loss.item()

            _, predicted = torch.max(output.data,1)

            # Sum n_sampler
            n_samples += labels.size(0)

            # If predicted values are equalt to labels, the number of correct values is gathered
            valid_correct += (predicted == labels).sum().item()

            # Valid Accuracy
            valid_accuracy = valid_correct/n_samples

        # Append valid and training accuracy into list
        validation_accuracy.append(valid_accuracy)
        validation_loss.append(valid_loss/len(valid_loader))
        training_accuracy.append(train_acc)
        training_loss.append(train_loss/len(train_dataLoader))

        print("Epoch: {}/{}  ".format(epoch+1, epochs),  
                "Training loss: {:.4f}  ".format(train_loss/len(train_dataLoader)),
                "Training Accuracy : {:.4f}".format(train_acc),
                "Validation loss: {:.4f}".format(valid_loss/len(valid_loader)),
                "Validation Accuracy : {:.4f}".format(valid_accuracy))
    
    print("Training Finished")
    return validation_accuracy,validation_loss,training_accuracy,training_loss

def torch_no_grad(model,criterion):
    model.eval()
    test_loss = 0
    n_samples = 0
    test_correct = 0
    with torch.no_grad():
        for image,label in test_dataLoader:
            images = image.to(device)
            labels = label.to(device)
            # Calculate model output
            output = model(images)
            # Calculate test loss
            loss = criterion(output,labels)
            # Sum test loss
            test_loss += loss.item()

            _,predicted = torch.max(output.data,1)
            # Sum n samples
            n_samples += labels.size(0)
            # Calculate accuracy if predictions are equal to labels.
            test_correct += (predicted == labels).sum().item()
            # Calculate test accuracy
            test_accuracy = test_correct/n_samples
        
        print("Test Accuracy: {:.4f}".format(test_accuracy),
              ("Test Loss: {:.4f}".format(test_loss/len(test_dataLoader))))

def plot(train_acc, train_loss,valid_acc,valid_loss):
    # Convert list to array
    training_loss = np.array(train_loss)
    validation_loss = np.array(valid_loss)
    training_accuracy = np.array(train_acc)
    validation_accuracy = np.array(valid_acc)

    # Define graph
    fig,axes = plt.subplots(nrows=1, ncols=2, figsize = (10,5))

    # Axes[0]
    axes[0].plot(validation_loss, label = "Validation Loss")
    axes[0].plot(training_loss, label = "Training Loss")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Axes[1]
    axes[1].plot(validation_accuracy,label = "Validation Accuracy")
    axes[1].plot(training_accuracy, label = "Training Accuracy")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    train(model=model,epochs = epochs,optimizer = optimizer,criterion = criterion)
    torch_no_grad(model=model,criterion = criterion)
    plot(training_accuracy,training_loss,validation_accuracy,validation_loss) 

