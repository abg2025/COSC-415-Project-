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



def check_class_distribution(loader, class_names):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())
    label_count = Counter(all_labels)
    for label, count in label_count.items():
        print(f'Class {class_names[label]}: {count} instances')


    
def main():   

# Transform the data by converting images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load your datasets
    train_dataset = datasets.ImageFolder('hand2_train', transform=transform)
    test_dataset = datasets.ImageFolder('hand_not_removed_test', transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=100, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=1)

    # Get class names
    class_names = train_dataset.classes  # This assumes train and test have the same classes

    # Use this function with your train_loader and test_loader
    print("Training Set Class Distribution:")
    check_class_distribution(train_loader, class_names)
    
    print("\nTesting Set Class Distribution:")
    check_class_distribution(test_loader, class_names)



if __name__ == '__main__':
    main()

