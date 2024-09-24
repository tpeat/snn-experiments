# Spiking Neural Network Experiments

Spiking Neural Networks (SNNs) are a class of neural networks designed to
closer represent biology that uses discrete events called spikes to process when to
propagate information. Using the MNSIT digit classification dataset, we explore
the impact of number timesteps (T ), which determines temporal resolution;
the surrogate gradient scale (z), which influences gradient approximation; the
firing threshold (Vth) which determines when a neuron spikes; and the leakage
factor (β), which represents the decay rate of the membrane potential. For each
hyperparameter experiment, we trained the SNN for 10 epochs, initializing each
model with the same random seed to ensure reproducibility.

For full scope of the experiments, read more [here](report_tristan_peat.pdf)