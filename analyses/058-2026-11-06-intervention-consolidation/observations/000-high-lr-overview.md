# 00: High-rate Adam overview

Source: Analysis055 `figures/adam_full_trajectories_high.pdf` and its
`plot_manifest.json`; underlying Run215 evidence remains owned there.
Generator: `../01_figures.py` copies the authenticated PDF byte for byte.
There is no new reduction, rescaling or event selection.

Caption: High-rate full-batch Adam trajectories on CIFAR-10/MLP (eta=0.001),
CIFAR-10/ResNet-32 (0.003) and WikiText-2/Transformer (0.003). Rows show training
loss, projected gradient change, consecutive raw-gradient cosine and gradient
norm relative to its initial value. Filled gray intervals mark rise phases of
recovered excursions greater than 10%; outlined intervals have unobserved
recovery. Loss and relative norm use logarithmic axes.

Coverage: 1,024 updates per arm; losses at states 0--1024, raw norms at 0--1023,
and paired diagnostics at 1--1023. This is descriptive context, not a matched
intervention figure. One initialization per model, dependent events and finite
follow-up prevent independent-seed uncertainty or a prediction claim. Event
shading here has a different meaning from the pre-fork shading in Figures 01--06.
The observation and uncertainty contract in Analysis055 is preserved.
