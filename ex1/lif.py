import snntorch as snn
import torch
import matplotlib.pyplot as plt

class LIFNeuron:
    """
    Class mô phỏng neuron Leaky Integrate-and-Fire (LIF).
    """

    def __init__(self, beta=0.9, threshold=1.0):
        """
        Khởi tạo neuron LIF.
        Args:
            beta (float): Hệ số giảm tiềm năng màng.
            threshold (float): Ngưỡng phát xung.
        """
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.mem = self.lif.init_leaky()

    def simulate(self, input_current, num_steps, plot=False):
        """
        Mô phỏng phản ứng của neuron LIF với dòng điện đầu vào.
        Args:
            input_current (torch.Tensor): Dòng điện đầu vào (1D tensor).
            num_steps (int): Số bước thời gian mô phỏng.
            plot (bool): Nếu True, vẽ biểu đồ tiềm năng màng và các xung.
        Returns:
            tuple: Danh sách các xung và tiềm năng màng.
        """
        spk_rec = []
        mem_rec = []
        for step in range(num_steps):
            spk, self.mem = self.lif(input_current[step], self.mem)
            spk_rec.append(spk.item())
            mem_rec.append(self.mem.item())

        if plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(mem_rec)
            plt.title("Tiềm năng màng theo thời gian")
            plt.xlabel("Bước thời gian")
            plt.ylabel("Tiềm năng màng")

            plt.subplot(1, 2, 2)
            plt.plot(spk_rec)
            plt.title("Các xung theo thời gian")
            plt.xlabel("Bước thời gian")
            plt.ylabel("Xung (0 hoặc 1)")
            plt.tight_layout()
            # plt.savefig("outputs/lif_plot.png")
            plt.close()

        return spk_rec, mem_rec