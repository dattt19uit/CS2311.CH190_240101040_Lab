import torch
from snntorch import spikegen
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=128):
    """
    Tải dữ liệu MNIST và tạo DataLoader.
    Args:
        batch_size (int): Kích thước batch.
    Returns:
        tuple: train_loader, test_loader.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def encode_data(data, num_steps):
    """
    Mã hóa dữ liệu thành chuỗi xung sử dụng mã hóa tỷ lệ.
    Args:
        data (torch.Tensor): Dữ liệu đầu vào (batch, channels, height, width).
        num_steps (int): Số bước thời gian.
    Returns:
        torch.Tensor: Chuỗi xung (num_steps, batch, channels, height, width).
    """
    return spikegen.rate(data, num_steps=num_steps)