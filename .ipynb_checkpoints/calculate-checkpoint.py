import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std



def check_class_distribution(loader):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    label_count = Counter(all_labels)
    for label, count in label_count.items():
        print(f'Class {label}: {count} instances')


    
   

def main():
    # Transform the data by converting images to tensors
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    # Assuming you have a dataset loader ready
    dataset = datasets.ImageFolder('new_train', transform=transform)
    dataset1 = datasets.ImageFolder('new_test', transform=transform)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)  # More manageable batch size
    test_loader = DataLoader(dataset1, batch_size=100, shuffle=True, num_workers=1)

    # Use this function with your train_loader and test_loader
    #check_class_distribution(train_loader)
    #check_class_distribution(test_loader)


    mean, std = get_mean_std(train_loader)
    print("Mean:", mean)
    print("Standard Deviation:", std)



if __name__ == '__main__':
    main()

