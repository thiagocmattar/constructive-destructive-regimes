# Edge of Stability: practical experimental setups

Primary sources: [Cohen et al., ICLR 2021, Appendix I and L/M](https://arxiv.org/pdf/2103.00065),
[official implementation](https://github.com/locuslab/edge-of-stability/tree/github),
[historical PyTorch language-model example](https://github.com/pytorch/examples/tree/13acec6d7c78dacd5e1fe9b0b4a325e1d39abc15/word_language_model).

| Case in the paper | Training objective / effective update batch | Architecture and loss | Hessian coverage |
| --- | --- | --- | --- |
| CIFAR-10 5k | First 5,000 training images; every GD update averages all 5,000 | 200-200 MLP, tanh/ReLU/ELU/softplus/hardtanh; or two 32-channel convolutions and pooling; half squared L2 error to one-hot labels or cross entropy | All 5,000 images |
| Full CIFAR-10 | All 50,000 images; 50 fixed ghost batches of 1,000 for BN | ResNet-32 with BN; VGG-11 with/without BN, dropout removed; cross entropy | First 5,000 images / first five ghost groups |
| WikiText-2 | Full word-level training corpus, gradients accumulated before the GD update | Transformer, embedding 200, two heads, FFN 200, two layers, dropout zero, causal chunks of length 35; negative log likelihood | Subset described as 2,500 training examples |
| Chebyshev regression | All 20 evenly spaced points on [-1,1], polynomial degrees 3,4,5 | 100-unit one-hidden-layer tanh, Xavier initialization; MSE | Toy objective |
| Gaussian synthetic regression | Full synthetic data | Deep linear networks | Synthetic objective |

ResNet and ViT name architectures, not datasets. ResNet-32 is tested; ViT is not.
The Transformer is a small word-level language model, not a vision transformer.
The paper's main spectral quantity is the largest training-objective Hessian
eigenvalue, with the GD quadratic reference lambda_max=2/eta. We plot eta times
that quantity and a reference at 2. This is not a global neural-network
convergence theorem.

The released GD script defaults to physical batches of 1,000, seed-dependent
initialization, an optional 99% training-accuracy stopping rule and an abridged
5,000-example Hessian objective. Physical accumulation does not introduce
stochasticity. For BN it defines the ghost-normalization objective.

The article describes CIFAR channel normalization; the released 5k MLP code
normalizes each pixel coordinate. We preserve our existing released-code MLP
baseline and use channel normalization for the full-CIFAR ResNet case. The
Transformer appendix does not fully specify corpus ordering, physical sequence
batch size, the meaning of its Hessian 'example', or all preprocessing revision
details. Those choices must be explicitly declared here rather than claimed
as bitwise replication. Our historical tutorial reference predates submission.

For a publication-level assessment, three selected cases demonstrate architecture
and task variation over two datasets. They do not establish a universal effect,
or isolate architecture from data/loss changes. Stochastic tests need replicated
seeds and actual vector-gradient variance estimates, not loss variance or an
estimate inferred from a single batch mean.
