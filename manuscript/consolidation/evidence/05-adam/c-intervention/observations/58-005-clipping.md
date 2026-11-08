> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [005-clipping.md](../../../../../../analyses/058-2026-11-06-intervention-consolidation/observations/005-clipping.md).

# 05: MLP and ResNet gradient clipping

Sources: Analyses052/054 findings and plotted tables; Runs223/228/229/231 raw
boundary/metric records and source inventories. Generator `../01_figures.py`
combines the two MLP thresholds into one column and uses the fresh Run231 Spark
control for ResNet, not the distinct Run223 control shown in Figure 01.

Caption: Global gradient clipping before Adam moment updates on CIFAR-10.
Left: MLP caps 1.0 (blue) and 0.5 (orange), against unchanged Adam (gray).
Right: ResNet-32 cap 1.0 (blue), against its matched unclipped control (gray).
Rows show log training loss, projected raw-gradient change, raw cosine, gradient
norm and applied/base LR. Solid norm curves are raw gradients; dashed norm
curves are actual post-clipping optimizer inputs. Gray bands precede the fork.
All curves are unsmoothed; Y scales are independent.

Coverage: MLP states 448--800, fork 512, 288 updates, base LR 0.001, 5,000
images. ResNet states 192--576, fork 256, 320 updates, base LR 0.003, 50,000
images. For ResNet the shaded prefix is historical RTX4090 context; treatment
and fresh control both run on Spark after 256. Cross-GPU fork gradient error
was 0.0133%; exact within-Spark replay passed. Dashed input norms cover outgoing
updates only (MLP 512--799; ResNet 256--575).

Result: MLP caps 1.0/0.5 increase mean loss by 48.01%/2.08% versus unchanged;
0.5 avoids the largest bursts seen under 1.0 but still worsens tail loss.
ResNet clipping reduces mean loss 28.3% and maximum loss 52.2%, while increasing
tail and endpoint loss. A raw-gradient input cap is not an actual-update norm
cap. These are separate single-seed comparisons, not a pooled clipping effect;
event counts alone and isolated endpoints can misrepresent the tradeoff.
