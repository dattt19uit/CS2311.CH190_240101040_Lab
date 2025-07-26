import torch
from snntorch import functional as SF
import matplotlib.pyplot as plt

def train_and_evaluate(model, train_loader, test_loader, num_steps, num_epochs=1, lr=5e-3):
    """
    Huấn luyện và đánh giá mạng CSNN.
    Args:
        model: Mô hình SpikingCNN.
        train_loader: DataLoader cho tập huấn luyện.
        test_loader: DataLoader cho tập kiểm tra.
        num_steps (int): Số bước thời gian.
        num_epochs (int): Số epoch huấn luyện.
        lr (float): Learning rate.
    Returns:
        list: Lịch sử độ chính xác huấn luyện.
    """
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accuracy_hist = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for data, targets in train_loader:
            data = encode_data(data, num_steps)
            spk_rec = model(data)
            loss_val = loss_fn(spk_rec, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            _, predicted = torch.max(spk_rec.sum(dim=0), 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        train_accuracy = 100 * correct / total
        train_accuracy_hist.append(train_accuracy)
        print(f"Epoch {epoch+1}, Độ chính xác huấn luyện: {train_accuracy:.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = encode_data(data, num_steps)
            spk_rec = model(data)
            _, predicted = torch.max(spk_rec.sum(dim=0), 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    test_accuracy = 100 * correct / total
    print(f"Độ chính xác trên tập kiểm tra: {test_accuracy:.2f}%")

    plt.plot(train_accuracy_hist)
    plt.title("Độ chính xác huấn luyện qua các epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Độ chính xác (%)")
    plt.savefig("outputs/csnn_accuracy_plot.png")
    plt.close()

    return train_accuracy_hist

from ex2.data import encode_data