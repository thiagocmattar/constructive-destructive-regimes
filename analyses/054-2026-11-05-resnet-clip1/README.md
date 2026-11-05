# ResNet-32 global clipping at 1

This matched experiment tests whether global clipping before Adam prevents repeated training-loss spikes at a constant learning rate of 0.003. Run231 starts both Spark trajectories from the same saved RTX 4090 checkpoint at state 256 and observes every boundary through 576. The full objective contains 50,000 CIFAR-10 images, with physical batches of 1,000 and seed 0.

The figure compares log training loss, raw gradient agreement, gradient norms before and after clipping, the directional curvature diagnostic, and the LR ratio. Shading separates the RTX 4090 prefix from the Spark continuation. The green curve records the actual input to Adam at states 256 through 575. State 576 has an observed raw gradient and loss. Excursions use the existing Analysis036 trough and recovery detector with a 10% threshold.

After terminal collection, run `01_report.py` to verify source artifacts, repeat the checkpoint equality check and generate the vector PDF, exact plotted values and numerical summary. Use `results.json` for the post-fork window, last 32 states, endpoint, activation count and right-censored excursions. This is a single-seed development window.

Completed. [Paired figure](figures/resnet_clip1.pdf) · [Findings](FINDINGS.md) · [Verified summary](results.json).
