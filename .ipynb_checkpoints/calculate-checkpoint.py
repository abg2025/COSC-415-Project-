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
    transform = transforms.Compose([transforms.ToTensor()])

    # Assuming you have a dataset loader ready
    dataset = datasets.ImageFolder('new_train', transform=transform)
    dataset1 = datasets.ImageFolder('new_test', transform=transform)
    train_loader = DataLoader(dataset, batch_size=len(dataset), num_workers=1)
    test_loader = DataLoader(dataset1, batch_size=len(dataset), num_workers=1)
    # Use this function with your train_loader and test_loader
    check_class_distribution(train_loader)
    check_class_distribution(test_loader)
    # mean, std = get_mean_std(loader)
    # print("Mean:", mean)
    # print("Standard Deviation:", std)



if __name__ == '__main__':
    main()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))