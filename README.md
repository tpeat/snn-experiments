# Spiking Neural Network Experiments

Spiking Neural Networks (SNNs) are a class of neural networks that more closely emulate biological processes by using discrete events, known as spikes, to control when information is propagated. In this study, we use the MNIST digit classification dataset to investigate how key hyperparameters affect SNN performance. Specifically, we examine the number of timesteps ($T$), which determines the temporal resolution; the surrogate gradient scale ($z$), which influences the approximation of gradients for backpropagation; the firing threshold ($V_{th}$), which dictates when a neuron fires; and the leakage factor ($\beta$), which controls the decay rate of the neuron's membrane potential. Each model was trained for 10 epochs, with identical random seeds to ensure reproducibility across experiments.