import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from skimage import io

#preparing the data set 
class dataset(Dataset):
    def __init__(self, csv_file:str, root_dir:str, transform=None) :
        super().__init__()
        self.annotation=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotation)


    def __getitem__(self, index):
        img_path =os.path.join(self.root_dir, self.annotation.iloc[index,0])
        image=io.imread(img_path)
        y_lable1 = torch.tensor(int(self.annotation.iloc[index,3]))
        y_lable2 = torch.tensor(int(self.annotation.iloc[index,4]))
        y_lable3 = torch.tensor(int(self.annotation.iloc[index,6]))
        
        if self.transform:
            image = self.transform(image)

        return image,y_lable1,y_lable2,y_lable3




#hyper parameters 
num_epochs = 2
learning_rate = 0.001
batch_size = 64 

#load data 
train_set=dataset(csv_file= r'E:\colleage\ARL\persiption\data_set\driving_log_train.csv',root_dir= r'E:\colleage\ARL\persiption\data_set\IMG', transform=transforms.ToTensor())

test_set=dataset(csv_file= r'E:\colleage\ARL\persiption\test\driving_log_test.csv',root_dir= r'E:\colleage\ARL\persiption\test\IMG', transform=transforms.ToTensor())

train_loader= DataLoader(train_set, batch_size, shuffle=True)
test_loader=DataLoader(test_set, batch_size, shuffle=False)









class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 128*2, 5)
        self.fc1 = nn.Linear(89088, 64)
        self.h1  = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # N, 3, 32, 32
        x = F.relu(self.conv1(x))   # -> N, 64, 320, 160
        x = self.pool(x)            # -> N, 64, 158, 78
        x = F.relu(self.conv2(x))   # -> N, 128, 158, 78
        x = self.pool(x)            # -> N, 128, 77,37
        x = F.relu(self.conv3(x))   # -> N, 256, 77,37
        x = torch.flatten(x, 1)     # -> N, 1024
        x = F.relu(self.fc1(x))     # -> N, 64
        x = F.relu(self.h1(x)) 
        x = self.fc2(x)             # -> N, 1
        return x

model = ConvNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # You can choose an appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss1 = 0.0

    for images, label1,label2, label3 in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, label1.float())  # Compute the loss for label1 (adjust as needed)
        #loss1 = criterion(outputs, label2.float())
        #loss2 = criterion(outputs, label3.float())
        #total_loss= loss + loss1 + loss2
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
      

        running_loss1 += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss1: {running_loss1 / len(train_loader)}")

    running_loss2 = 0.0
    for images, label1,label2, label3 in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        #loss = criterion(outputs, label1.float())  # Compute the loss for label1 (adjust as needed)
        loss1 = criterion(outputs, label2.float())
        #loss2 = criterion(outputs, label3.float())
        #total_loss= loss + loss1 + loss2
        loss1.backward()  # Backpropagation
        optimizer.step()  # Update the weights
      

        running_loss2 += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss2: {running_loss2 / len(train_loader)}")

    running_loss3 = 0.0
    for images, label1,label2, label3 in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        #loss = criterion(outputs, label1.float())  # Compute the loss for label1 (adjust as needed)
        #loss1 = criterion(outputs, label2.float())
        loss2 = criterion(outputs, label3.float())
        #total_loss= loss + loss1 + loss2
        loss2.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        
      

        running_loss3 += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss3: {running_loss3 / len(train_loader)}")


# Save the model
torch.save(model.state_dict(), 'my_model.pth')

# Now you can perform testing or save other information as needed

# To load the model back for testing or further use:
# Define the model architecture first
model = ConvNet()

# Load the saved model state dictionary
model.load_state_dict(torch.load('my_model.pth'))



#testing loop

# Put the model in evaluation mode
model.eval()
total_loss1 = 0.0
total_loss2 = 0.0
total_loss3 = 0.0
predictions1 = []
predictions2 = []
predictions3 = []
with torch.no_grad():
    for images, label1, label2, label3 in test_loader:
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, label1.float())  # Compute the loss for label1 (adjust as needed)
        total_loss1 += loss.item()
        predictions1.extend(outputs.cpu().numpy())  # Store the predictions
    
    for images, label1, label2, label3 in test_loader:
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, label2.float())  # Compute the loss for label1 (adjust as needed)
        total_loss2 += loss.item()
        predictions2.extend(outputs.cpu().numpy())  # Store the predictions

    for images, label1, label2, label3 in test_loader:
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, label3.float())  # Compute the loss for label1 (adjust as needed)
        total_loss3 += loss.item()
        predictions3.extend(outputs.cpu().numpy())  # Store the predictions

''''
# Calculate the average test loss
average_loss = total_loss1 / len(test_loader)
print(f"Average Test Loss1: {average_loss}")

average_loss = total_loss2 / len(test_loader)
print(f"Average Test Loss2: {average_loss}")

average_loss = total_loss3 / len(test_loader)
print(f"Average Test Loss3: {average_loss}")
'''










