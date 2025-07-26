import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class SpikingCNN(nn.Module):
    """
    Mạng Spiking CNN cho nhận dạng chữ số MNIST.
    """
    def __init__(self, num_steps=50, beta=0.9):
        """
        Khởi tạo mạng CSNN.
        Args:
            num_steps (int): Số bước thời gian mô phỏng.
            beta (float): Hệ số giảm tiềm năng màng.
        """
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        """
        Xử lý dữ liệu đầu vào qua mạng CSNN.
        Args:
            x (torch.Tensor): Dữ liệu đầu vào dạng chuỗi xung (num_steps, batch, 1, 28, 28).
        Returns:
            torch.Tensor: Chuỗi xung đầu ra.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_rec = []
        for step in range(self.num_steps):
            cur1 = self.conv1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.max_pool2d(spk1, 2)
            cur3 = self.conv2(cur2)
            spk2, mem2 = self.lif2(cur3, mem2)
            cur4 = F.max_pool2d(spk2, 2)
            cur5 = cur4.view(cur4.size(0), -1)
            cur6 = self.fc1(cur5)
            spk3, mem3 = self.lif3(cur6, mem3)
            spk_rec.append(spk3)
        return torch.stack(spk_rec)