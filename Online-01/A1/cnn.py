import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# transforms for CIFAR-10 (defined at module-level — safe)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Network-in-Network (NiN) implementation for CIFAR-10
class NiNBlock(nn.Module):
    """NiN block: Conv(kxk) -> ReLU -> Conv(1x1) -> ReLU -> Conv(1x1) -> ReLU
    Preserves spatial resolution (padding chosen for odd kernel sizes).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(NiNBlock, self).__init__()

        padding = (kernel_size - 1) // 2
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class NiN3(nn.Module):
    """Full NiN model as specified:
    NiN Block 1 (out=32)
    -> MaxPool(2x2)
    -> NiN Block 2 (out=64)
    -> MaxPool(2x2)
    -> NiN Block 3 (out=64)
    -> 1x1 Conv (64 -> 10)
    -> Global Average Pooling
    -> logits
    """
    def __init__(self, num_classes=10):
        super(NiN3, self).__init__()
        self.features = nn.Sequential(
            NiNBlock(3, 32),
            nn.MaxPool2d(2),
            NiNBlock(32, 64),
            nn.MaxPool2d(2),
            NiNBlock(64, 64),
        )
        # classifier: 1x1 conv to class channels followed by GAP
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)          # (N, num_classes, H, W)
        x = self.gap(x)                 # (N, num_classes, 1, 1)
        x = x.view(x.size(0), -1)       # (N, num_classes)
        return x  # raw logits


# training / evaluation helpers

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    for i, (images, labels) in enumerate(loader, 1):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        if i % 100 == 0:
            avg = running_loss / total
            print(f'Epoch [{epoch}] Step [{i}/{len(loader)}]  loss={avg:.4f}')

    # return running_loss / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def main():
    # move dataset/dataloaders and training under main() to be Windows-safe
    train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=False,
                                                transform=transform_test,
                                                download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)

    model = NiN3(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

    # final evaluation
    test_acc = evaluate(model, test_loader, device)
    print(f'Test Accuracy of the model on the 10000 test images: {test_acc:.2f} %')

    # save checkpoint
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    # required on Windows to safely spawn dataloader worker processes
    import multiprocessing
    multiprocessing.freeze_support()
    main()