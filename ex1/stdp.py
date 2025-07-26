import torch
import matplotlib.pyplot as plt

def update_stdp_weights(pre_spikes, post_spikes, w_init, A_plus=0.1, A_minus=-0.1, tau_plus=20, tau_minus=20,
                        num_iterations=10, plot=False):
    """
    Cập nhật trọng số synapse sử dụng quy tắc STDP.
    Args:
        pre_spikes (list): Thời điểm các xung pre-synaptic.
        post_spikes (list): Thời điểm các xung post-synaptic.
        w_init (float): Trọng số ban đầu.
        A_plus (float): Hệ số tăng cường synapse (LTP).
        A_minus (float): Hệ số suy yếu synapse (LTD).
        tau_plus (float): Hằng số thời gian cho LTP.
        tau_minus (float): Hằng số thời gian cho LTD.
        num_iterations (int): Số lần lặp cập nhật trọng số.
        plot (bool): Nếu True, vẽ biểu đồ thay đổi trọng số.
    Returns:
        list: Danh sách các trọng số qua các lần lặp.
    """
    w = torch.tensor(w_init, dtype=torch.float)
    weight_changes = [w.item()]

    for _ in range(num_iterations):
        w_temp = w.clone()
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if dt > 0:
                    dw = A_plus * torch.tensor(-dt / tau_plus)  # LTP
                elif dt < 0:
                    dw = A_minus * torch.tensor(abs(dt) / tau_minus)  # LTD
                else:
                    dw = 0
                w_temp += dw
        w = w_temp
        weight_changes.append(w.item())

    if plot:
        plt.plot(weight_changes)
        plt.title("Thay đổi trọng số synapse qua các lần lặp STDP")
        plt.xlabel("Lần lặp")
        plt.ylabel("Trọng số")
        # plt.savefig("outputs/stdp_plot.png")
        # plt.close()

    return weight_changes