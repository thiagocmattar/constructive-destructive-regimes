"""Retained-Gram adversarial controls: step magnitude versus directional structure.

No new captures, trajectory changes, window selection or statistical null test.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum(gram):
    gram = (gram + gram.T) / 2
    values = np.linalg.eigvalsh(gram)[::-1]
    assert values[-1] >= -1e-9 * max(values[0], 1)
    values = np.maximum(values, 0)
    energy = values.sum()
    np.testing.assert_allclose(energy, np.trace(gram), rtol=1e-10)
    # Independent SVD of the retained Gram gives the same nonnegative spectrum.
    np.testing.assert_allclose(np.linalg.svd(gram, compute_uv=False), values,
                               rtol=1e-8, atol=1e-10 * max(values[0], 1))
    fractions = values / energy
    return dict(top1=float(fractions[0]),
                rank95=int(np.searchsorted(np.cumsum(fractions), .95) + 1),
                participation_ratio=float(1 / np.sum(fractions ** 2)),
                energy=float(energy), fractions=fractions.tolist())


def main():
    sources, results = {}, []
    for case in ("mlp", "resnet", "transformer"):
        reference = HERE / ("mlp_450_500_results.json" if case == "mlp" else f"{case}_results.json")
        prior = json.loads(reference.read_text())
        if case == "mlp":
            raw = ROOT / "analyses/062-2026-11-07-mlp-gradient-amplification/artifacts/all-layer-svd/grams.npz"
            key, selection = "output", slice(3, 54)  # Original rows 447..800; fit 450..500.
        else:
            raw = ROOT / f"runs/243-2026-11-07-architecture-instability-tracing/artifacts/cloud/{case}/grams.npz"
            key, selection = "output.log_probabilities", slice(1, None)
        relative = raw.relative_to(ROOT).as_posix()
        assert sha(raw) == prior["sources"][relative], relative
        sources[relative] = sha(raw)
        sources[reference.relative_to(ROOT).as_posix()] = sha(reference)
        with np.load(raw) as archive:
            gram = archive[key][selection, selection]
        assert len(gram) == len(prior["states"])
        observed = spectrum(gram)
        old = next(s for s in prior["stages"] if s["name"] == key)
        np.testing.assert_allclose(observed["top1"], old["top1"], rtol=1e-10)
        assert observed["rank95"] == old["rank95"]

        # Counterfactual orthogonal steps preserve every row's squared norm.
        diagonal = np.diag(gram)
        reference_spectrum = spectrum(np.diag(diagonal))
        np.testing.assert_allclose(reference_spectrum["fractions"],
                                   np.sort(diagonal)[::-1] / diagonal.sum())
        keep = diagonal > 0
        roots = np.sqrt(diagonal[keep])
        normalized = gram[np.ix_(keep, keep)] / np.outer(roots, roots)
        np.testing.assert_allclose(np.diag(normalized), 1)
        directional = spectrum(normalized)
        # Row-order permutation cannot change the spectrum; SVD alone is not an oscillation test.
        np.testing.assert_allclose(spectrum(gram[::-1, ::-1])["fractions"], observed["fractions"],
                                   atol=1e-12)
        results.append(dict(case=case, output_space=key,
                            outgoing_start=prior["states"][0], outgoing_end=prior["states"][-1],
                            rows=len(gram), zero_rows_excluded_from_normalization=int((~keep).sum()),
                            observed=observed, same_energy_orthogonal=reference_spectrum,
                            unit_row_directional=directional,
                            original_output_leading_sign_flips=old["leading_sign_flips"]))
    output = dict(
        date="2026-11-08", purpose="Adversarial check of spectral concentration for the consolidation plan",
        controls={"same_energy_orthogonal": "diag(K): preserve every step norm, remove cross-step directional overlap",
                  "unit_row_directional": "diag(K)^(-1/2) K diag(K)^(-1/2), excluding zero-energy rows"},
        scope="Original selected windows/probes; diagnostic references, not a randomized null or quiet-window comparison",
        interpretation="Off-diagonal structure increases concentration beyond magnitudes alone. Temporal order needs cosine/coefficient histories; spectra are invariant to row permutations.",
        checks=dict(source_hashes=True, original_top1_and_rank95=True, PSD_trace=True,
                    independent_Gram_SVD=True, diagonal_reference=True,
                    unit_diagonal=True, row_permutation_invariance=True),
        sources=sources, generator_sha256=sha(Path(__file__)), cases=results)
    (HERE / "spectral_controls.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8", newline="\n")
    for result in results:
        print(result["case"], "T", result["rows"],
              {key: (round(result[key]["top1"] * 100, 3), result[key]["rank95"])
               for key in ("observed", "same_energy_orthogonal", "unit_row_directional")})


if __name__ == "__main__":
    main()
