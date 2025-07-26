from ex2.model import SpikingCNN
from ex2.data import get_mnist_loaders
from ex2.train import train_and_evaluate

# Khởi tạo tham số
num_steps = 50
batch_size = 128
num_epochs = 1
lr = 5e-3

# Tải dữ liệu
train_loader, test_loader = get_mnist_loaders(batch_size)

# Khởi tạo mô hình
model = SpikingCNN(num_steps=num_steps, beta=0.9)

# Huấn luyện và đánh giá
train_and_evaluate(model, train_loader, test_loader, num_steps, num_epochs, lr)