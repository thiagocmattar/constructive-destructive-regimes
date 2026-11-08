The general idea for instability trancing is this (below). I'll first be informal and describe in plain words my line of thought and how we developed this idea, before we go on to the formality, experiments and results (see analyses\063-2026-11-07-architecture-instability-tracing), ok?

1) Stability analysis maps into intervention, e.g.
- etakappa > 2 -> reduce learning rate
- adam spike decoupling analysis (ref: Adaptive Preconditioners Trigger Loss Spikes in Adam) -> reduce beta to couple again
- rho <= -gradient norms ratio -> gradient clipping

But what if we go deeper into the representations? We're observing the loss, but how to map it directly into the representation? Gradients maps into weights, but let's see the representations.. so we started:
- measuring the change in loss output for a fixed probe (data sample) accross 2 steps. The cosine between those responses maps into network output F oscilation. From the destructive regime idea, I saw: network output oscilates a lot as well. So the destructive regime is mapping in the function response;
- then I tried to measure how this oscilation correlates with gradient oscilation. And the answer is: greatly (correlation >0.7). So, I thought: there's something there.. we can map the oscilation backwards into the network activations, and therefore its hidden / latent representation;
- so I kept going backwards accross all layers. Then I did a SVD analysis of this representation oscilation, and the finding is strikingly:

```
The network exhibits highly structured, low-dimensional functional oscillations, with one dominant mode reversing direction repeatedly. 
```

The analysis analyses\063-2026-11-07-architecture-instability-tracing have the scripts, ALGORITHM (I expect the algorithm to be a contribution), analysis, results, and even a report: analyses\063-2026-11-07-architecture-instability-tracing\report\instability-tracing-report.pdf

There is an AMAZING figure at the 2nd page of analyses\065-2026-11-07-mlp-frozen-lr-double\figures\07-distributed-displacement-and-controls.pdf which is a heatmap showing how the oscilation moves backward through the network, how its activation spikes. Maybe we could do the figures analyses\063-2026-11-07-architecture-instability-tracing\figures more paper ready following this kind of aggregation between layers, showing how they propagate to SVD, etc.

PS: I think we have a VERY STRONG RICH CONTRIBUTION HERE:
- algorithm
- findings
- interpretation
- motivation for interventions, etc.

PS: I did not do an extensive analysis of the literature on this topic. I think we're adjacent to mechanistic interpretability, but we could be doing something new like mechanicist stability analysis. I think this would be very very strong. Maybe even field-shaping.
