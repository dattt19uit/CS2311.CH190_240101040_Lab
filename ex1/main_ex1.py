import torch
from ex1.encoding import encode_to_spike_train
from ex1.lif import LIFNeuron
from ex1.stdp import update_stdp_weights

# Thử nghiệm Encoding
time_steps = 100
t = torch.linspace(0, 1, time_steps)
signal = torch.sin(2 * torch.pi * 5 * t)
spike_train = encode_to_spike_train(signal, num_steps=time_steps, gain=1.0, offset=0.0, plot=True)

# Thử nghiệm LIF
lif_neuron = LIFNeuron(beta=0.9, threshold=1.0)
num_steps = 50
input_current = torch.zeros(num_steps)
input_current[10:30] = 0.5
spk_rec, mem_rec = lif_neuron.simulate(input_current, num_steps, plot=True)

# Thử nghiệm STDP
pre_spikes = [10, 20, 30]
post_spikes = [15, 25]
weight_changes = update_stdp_weights(pre_spikes, post_spikes, w_init=0.5, plot=True)
print(f"Trọng số cuối cùng: {weight_changes[-1]:.4f}")