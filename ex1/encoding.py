import torch
import matplotlib.pyplot as plt
from snntorch import spikegen


def encode_to_spike_train(signal, num_steps, gain=1.0, offset=0.0, plot=False):
    """
    Chuyển đổi tín hiệu liên tục thành chuỗi xung sử dụng mã hóa tỷ lệ.
    Args:
        signal (torch.Tensor): Tín hiệu liên tục đầu vào (1D tensor).
        num_steps (int): Số bước thời gian mô phỏng.
        gain (float): Hệ số khuếch đại tín hiệu.
        offset (float): Độ lệch tín hiệu.
        plot (bool): Nếu True, vẽ biểu đồ tín hiệu và chuỗi xung.
    Returns:
        torch.Tensor: Chuỗi xung (spike train).
    """
    signal = signal.view(-1)
    spike_train = spikegen.rate(signal, num_steps=num_steps, gain=gain, offset=offset)

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(signal.numpy())
        plt.title("Tín hiệu liên tục")
        plt.xlabel("Thời gian")
        plt.ylabel("Biên độ")

        plt.subplot(1, 2, 2)
        plt.plot(spike_train.numpy())
        plt.title("Chuỗi xung (Rate Coding)")
        plt.xlabel("Bước thời gian")
        plt.ylabel("Xung (0 hoặc 1)")
        plt.tight_layout()
        # plt.savefig("outputs/encoding_plot.png")
        plt.close()

    return spike_train