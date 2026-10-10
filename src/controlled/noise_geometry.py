"""Exact signal/noise decompositions along one adapted parameter trajectory."""
import numpy as np


def noise_terms(signal, noise):
    """Pathwise identities; zero-mean noise does not remove the cross terms."""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if signal.shape != noise.shape or signal.ndim != 2:
        raise ValueError("Signal and noise must have equal (updates, coordinates) shapes")
    signal_sum = np.cumsum(signal, axis=0)
    noise_sum = np.cumsum(noise, axis=0)
    signal_b2 = np.einsum("ij,ij->i", signal_sum, signal_sum)
    noise_b2 = np.einsum("ij,ij->i", noise_sum, noise_sum)
    prefix_cross = 2 * np.einsum("ij,ij->i", signal_sum, noise_sum)
    signal_energy = np.cumsum(np.einsum("ij,ij->i", signal, signal))
    noise_energy = np.cumsum(np.einsum("ij,ij->i", noise, noise))
    same_step_cross = 2 * np.cumsum(np.einsum("ij,ij->i", signal, noise))
    return dict(signal_B_squared=signal_b2, noise_B_squared=noise_b2,
                prefix_cross=prefix_cross, signal_energy=signal_energy,
                noise_energy=noise_energy, same_step_cross=same_step_cross,
                signal_all_pair=signal_b2 - signal_energy,
                noise_all_pair=noise_b2 - noise_energy,
                temporal_cross=prefix_cross - same_step_cross)
