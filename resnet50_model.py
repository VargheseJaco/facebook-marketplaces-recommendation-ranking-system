#%%
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision import transforms as T
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

mps_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ImageDataset(Dataset):
    
    def __init__(self, to_image = False):
        super().__init__()
        self.labels = pd.read_csv('training_data.csv',lineterminator='\n')
        self.to_image = to_image

    def __getitem__(self, index):
        img_path = f'cleaned_images/{self.labels['id'].iloc[index]}.jpg'
        image = read_image(img_path).float()
        label = self.labels['label'].iloc[index]
        if self.to_image == True:    
            transform = T.ToPILImage()
            image = transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)
    
data = ImageDataset(to_image=False)

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = len(data) - (train_size + val_size)

train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class ImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)

    def forward(self, X):
        return self.resnet50(X)
    
def train(model, dataloader, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter()
    batch_idx = 0
    
    for epoch in range(epochs):
        for batch in dataloader:
            feature, label = batch
            feature, label = feature.to(mps_device), label.to(mps_device)
            # print(feature)
            prediction = model(feature)
            loss = F.cross_entropy(prediction, label)
            loss.backward()
            # print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('Loss', loss.item(), batch_idx)
            batch_idx += 1
        print(f'epoch {epoch} completed')
        torch.save(model.state_dict(), f'model_evaluation/epoch_{time.time}.pt')
        sd = model.state_dict()
        torch.save(sd['resnet50.fc.weight'], f'model_evaluation/weights/epoch_{epoch}.pt')

imgmodel = ImageClassifier()
imgmodel.to(device=mps_device)

train(imgmodel, train_loader)
# %%

# %%
