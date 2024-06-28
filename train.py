import cProfile
import pstats
import time
    

def train_ghostnet():
    import os
    import torch
    from torchvision.datasets import ImageFolder
    from torchvision.datasets.folder import default_loader
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader, random_split, Subset
    import timm
    from torch import nn, optim
    from tqdm import tqdm
    import numpy as np


    class RobustImageFolder(ImageFolder):
        def __init__(self, root, transform=None):
            super(RobustImageFolder, self).__init__(root, transform=transform, loader=self.robust_loader)
            self.transform = transform

        def robust_loader(self, path):
            try:
                return default_loader(path)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                return None

        def __getitem__(self, index):
            while True:
                #print(index)
                path, target = self.samples[index]
                sample = self.loader(path)
                if sample is None:
                    print(f"Skipping corrupted image: {path}")
                    index = (index + 1) % len(self.samples)
                else:
                    break
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target


    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(torch.cuda.is_available())  # Should return True if CUDA is properly installed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the custom dataset
    dataset_path = 'C:\\Users\\paperspace\\Documents\\adelos\\dataset\\train_faces'
    full_dataset = RobustImageFolder(dataset_path, transform=transform)
    num_images = len(full_dataset)
    #num_images = 1280
    all_indices = np.arange(len(full_dataset))
    random_indices = np.random.permutation(all_indices)[:num_images]
    subset = Subset(full_dataset, random_indices)



    train_size = int(0.8 * len(subset))
    batch_size = 128
    val_size = len(subset) - train_size
    train_dataset, val_dataset = random_split(subset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Load a pre-trained model from timm
    model = timm.create_model('ghostnetv2_160', pretrained=True)

    model.classifier = nn.Linear(model.classifier.in_features, 1)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    counter = 0
    best_model_path = 'models/best_model.pth'  # Path to save the best model
    max_epoches = 100
    # Training loop
    model.train()

    for epoch in range(max_epoches):  # Maximum of 10 epochs
        running_loss = 0.0
        #print(type(train_loader))
        startTime = time.perf_counter()
        for images, labels in tqdm(train_loader):
            #print(images.shape)
            #print(f'Image Extracting:{time.perf_counter()-startTime}')
            startTime = time.perf_counter()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            #print(f'Image Loading:{time.perf_counter()-startTime}')
            optimizer.zero_grad()
            startTime = time.perf_counter()
            outputs = model(images)
            #print(f'Output Calculation:{time.perf_counter()-startTime}')
            labels = labels.float().view(-1, 1)
            startTime = time.perf_counter()
            loss = criterion(outputs, labels)
            #print(f'Loss Calculation:{time.perf_counter()-startTime}')
            startTime = time.perf_counter()
            loss.backward()
            #print(f'Backward Calculation:{time.perf_counter()-startTime}')
            startTime = time.perf_counter()
            optimizer.step()
            #print(f'Optimizing Calculation:{time.perf_counter()-startTime}')
            running_loss += loss.item()

        # Validation loop
        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                labels = labels.float().view(-1, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()  # For binary classification
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}, Accuracy: {accuracy:.2f}%")
        torch.save(model.state_dict(), f'models/epoch_{epoch+1}_valloss_{val_loss}.pth')
        # Check if this is the best model and save
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Validation Loss: {best_loss}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")

    # Load and use the best model later if needed
    model.load_state_dict(torch.load(best_model_path))


if __name__ == "__main__":
    profile = cProfile.Profile()
    profile.enable()
    train_ghostnet()
    profile.disable()

    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profile, stream=f)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()
