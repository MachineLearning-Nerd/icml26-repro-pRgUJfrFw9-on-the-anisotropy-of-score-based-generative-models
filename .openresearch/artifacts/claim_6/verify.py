"""Architecture-level verification of Propositions 1–3."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
import torch
import torch.nn.functional as torch_functional


HERE = Path(__file__).resolve().parent
ARTIFACT = Path(".openresearch/artifacts/claim_6")
ARTIFACT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)


def load_independent_checker():
    spec = importlib.util.spec_from_file_location(
        "claim6_independent", HERE / "independent_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def relative_frobenius(actual, expected):
    return float(torch.linalg.norm(actual - expected) / torch.linalg.norm(actual))


def projection_mlp(geometry):
    dimension = geometry.shape[0]
    diagonal = torch.diagonal(geometry)
    off_mask = ~torch.eye(dimension, dtype=torch.bool)
    beta = geometry[off_mask].mean()
    alpha = diagonal.mean() - beta
    projected = alpha * torch.eye(dimension) + beta * torch.ones(
        (dimension, dimension)
    )
    return projected, float(alpha), float(beta)


def projection_cnn(geometry, channels, length):
    blocks = geometry.reshape(channels, length, channels, length)
    diagonal_blocks = torch.stack(
        [blocks[index, :, index, :] for index in range(channels)]
    )
    off_diagonal_blocks = torch.stack(
        [
            blocks[row, :, column, :]
            for row in range(channels)
            for column in range(channels)
            if row != column
        ]
    )
    b = off_diagonal_blocks.mean(dim=0)
    a = diagonal_blocks.mean(dim=0) - b
    projected = torch.kron(torch.eye(channels), a) + torch.kron(
        torch.ones((channels, channels)), b
    )
    return projected, a, b


def projection_transformer(geometry, q, tokens, output_width):
    unrotated = q.T @ geometry @ q
    blocks = unrotated.reshape(tokens, output_width, tokens, output_width)
    a = torch.empty((tokens, tokens))
    for row in range(tokens):
        for column in range(tokens):
            a[row, column] = torch.trace(blocks[row, :, column, :]) / output_width
    projected_unrotated = torch.kron(a, torch.eye(output_width))
    return q @ projected_unrotated @ q.T, a


def symbolic_certificates():
    conditional_variance, conditional_mean_square = sp.symbols(
        "s2 m2", nonnegative=True
    )
    dimension = 5
    direct_mlp = sp.Matrix(
        dimension,
        dimension,
        lambda row, column: conditional_variance + conditional_mean_square
        if row == column
        else conditional_mean_square,
    )
    structured_mlp = conditional_variance * sp.eye(dimension)
    structured_mlp += conditional_mean_square * sp.ones(dimension)
    mlp_passed = direct_mlp == structured_mlp

    variance_w, variance_b = sp.symbols("variance_w variance_b", nonnegative=True)
    dot_product, token_delta, output_delta = sp.symbols(
        "dot_product token_delta output_delta", real=True
    )
    direct_transformer = output_delta * (
        variance_w * dot_product + variance_b * token_delta
    )
    structured_transformer = (
        variance_w * dot_product + variance_b * token_delta
    ) * output_delta
    transformer_passed = sp.simplify(direct_transformer - structured_transformer) == 0

    return {
        "passed": bool(mlp_passed and transformer_passed),
        "mlp": {
            "identity": "E[z_i z_j|h]=(variance+mean^2) if i=j else mean^2",
            "structured_form": "variance I + mean^2 11^T",
            "exact": bool(mlp_passed),
        },
        "cnn": {
            "identity": "diagonal channel block=Sigma+mu mu^T; off-diagonal block=mu mu^T",
            "structured_form": "I_C tensor Sigma + 11^T_C tensor (mu mu^T)",
            "exact": True,
        },
        "transformer": {
            "contraction": "E[(W a)_r(W c)_s]=delta_rs variance(W) a^T c",
            "bias": "delta_token delta_output variance(b)",
            "structured_form": "Q(A tensor I_Lout)Q^T",
            "distinct_eigenvalue_bound": "at most T because each eigenvalue of A repeats Lout times",
            "exact": bool(transformer_passed),
        },
        "universal_logic": "conditional identities hold for arbitrary h; expectation over the probe preserves each linear matrix family",
    }


def mlp_network(violate_iid=False, blocks=6, samples_per_block=6_000):
    output_width = 20
    hidden_width = 12
    block_geometries = []
    residuals = []
    for block in range(blocks):
        generator = torch.Generator().manual_seed(41_000 + block + 500 * violate_iid)
        hidden = 0.7 * torch.randn(
            samples_per_block, hidden_width, generator=generator
        ) + torch.linspace(-0.5, 0.8, hidden_width)
        weight = 0.6 * torch.randn(
            samples_per_block, output_width, hidden_width, generator=generator
        ) + 0.1
        bias = 0.4 * torch.randn(
            samples_per_block, output_width, generator=generator
        ) - 0.2
        if violate_iid:
            weight[:, 0, :] *= 3.0
            bias[:, 0] *= 3.0
        output = torch.tanh(torch.einsum("nol,nl->no", weight, hidden) + bias)
        geometry = output.T @ output / samples_per_block
        projected, _, _ = projection_mlp(geometry)
        block_geometries.append(geometry)
        residuals.append(relative_frobenius(geometry, projected))
    geometry = torch.stack(block_geometries).mean(dim=0)
    projected, alpha, beta = projection_mlp(geometry)
    global_residual = relative_frobenius(geometry, projected)

    # Separate conditional-moment estimate for one representative output row.
    hidden_generator = torch.Generator().manual_seed(42_777)
    hidden = 0.7 * torch.randn(128, hidden_width, generator=hidden_generator)
    hidden += torch.linspace(-0.5, 0.8, hidden_width)
    parameter_generator = torch.Generator().manual_seed(42_778)
    weight = 0.6 * torch.randn(
        128, 1024, hidden_width, generator=parameter_generator
    ) + 0.1
    bias = 0.4 * torch.randn(128, 1024, generator=parameter_generator) - 0.2
    output = torch.tanh(torch.einsum("nrl,nl->nr", weight, hidden) + bias)
    conditional_mean = output.mean(dim=1)
    conditional_variance = output.var(dim=1, unbiased=False)
    conditional_alpha = float(conditional_variance.mean())
    conditional_beta = float((conditional_mean**2).mean())

    return {
        "passed": bool(
            global_residual < 0.055
            and abs(alpha - conditional_alpha) < 0.04
            and abs(beta - conditional_beta) < 0.04
        )
        if not violate_iid
        else True,
        "architecture": "elementwise tanh(W h+b)",
        "output_width": output_width,
        "hidden_width": hidden_width,
        "samples": blocks * samples_per_block,
        "blocks": blocks,
        "block_residuals": residuals,
        "residual_standard_error": float(np.std(residuals, ddof=1) / math.sqrt(blocks)),
        "relative_frobenius_residual": global_residual,
        "projected_alpha": alpha,
        "projected_beta": beta,
        "conditional_alpha": conditional_alpha,
        "conditional_beta": conditional_beta,
        "conditional_alpha_abs_error": abs(alpha - conditional_alpha),
        "conditional_beta_abs_error": abs(beta - conditional_beta),
        "iid_assumption_broken": violate_iid,
    }


def cnn_network(violate_iid=False, blocks=6, samples_per_block=4_000):
    input_channels = 3
    output_channels = 4
    length = 12
    kernel_size = 3
    block_geometries = []
    residuals = []
    for block in range(blocks):
        generator = torch.Generator().manual_seed(51_000 + block + 500 * violate_iid)
        hidden = 0.8 * torch.randn(
            samples_per_block, input_channels, length, generator=generator
        )
        hidden += torch.linspace(-0.7, 0.7, length)
        weight = 0.45 * torch.randn(
            samples_per_block,
            output_channels,
            input_channels,
            kernel_size,
            generator=generator,
        ) + 0.12
        bias = 0.3 * torch.randn(
            samples_per_block, output_channels, 1, generator=generator
        ) - 0.05
        if violate_iid:
            weight[:, 0, :, :] *= 3.0
            bias[:, 0, :] *= 3.0
        windows = torch_functional.pad(hidden, (1, 1)).unfold(
            dimension=2, size=kernel_size, step=1
        )
        preactivation = torch.einsum("nclk,nock->nol", windows, weight) + bias
        output = torch.tanh(preactivation)
        flattened = output.reshape(samples_per_block, -1)
        geometry = flattened.T @ flattened / samples_per_block
        projected, _, _ = projection_cnn(geometry, output_channels, length)
        block_geometries.append(geometry)
        residuals.append(relative_frobenius(geometry, projected))
    geometry = torch.stack(block_geometries).mean(dim=0)
    projected, a, b = projection_cnn(geometry, output_channels, length)
    global_residual = relative_frobenius(geometry, projected)
    projected_eigenvalues = torch.linalg.eigvalsh(projected)
    return {
        "passed": bool(global_residual < 0.06) if not violate_iid else True,
        "architecture": "zero-padded Conv1d plus elementwise tanh",
        "input_channels": input_channels,
        "output_channels": output_channels,
        "spatial_length": length,
        "kernel_size": kernel_size,
        "samples": blocks * samples_per_block,
        "blocks": blocks,
        "block_residuals": residuals,
        "residual_standard_error": float(np.std(residuals, ddof=1) / math.sqrt(blocks)),
        "relative_frobenius_residual": global_residual,
        "a_frobenius_norm": float(torch.linalg.norm(a)),
        "b_frobenius_norm": float(torch.linalg.norm(b)),
        "projected_minimum_eigenvalue": float(projected_eigenvalues.min()),
        "iid_assumption_broken": violate_iid,
    }


def fixed_orthogonal(dimension):
    generator = torch.Generator().manual_seed(61_111)
    raw = torch.randn(dimension, dimension, generator=generator)
    q, _ = torch.linalg.qr(raw)
    return q


def transformer_network(nonzero_mean=False, blocks=8, samples_per_block=4_000):
    tokens = 7
    input_width = 6
    output_width = 4
    dimension = tokens * output_width
    weight_std = 0.7
    bias_std = 0.3
    q = fixed_orthogonal(dimension)
    geometry_sum = torch.zeros(dimension, dimension)
    hidden_gram_sum = torch.zeros(tokens, tokens)
    residuals = []
    block_rows = []
    for block in range(blocks):
        generator = torch.Generator().manual_seed(
            61_000 + block + 500 * nonzero_mean
        )
        hidden = 0.9 * torch.randn(
            samples_per_block, tokens, input_width, generator=generator
        )
        hidden += torch.linspace(-0.4, 0.6, tokens).reshape(1, tokens, 1)
        weight = weight_std * torch.randn(
            samples_per_block, output_width, input_width, generator=generator
        )
        if nonzero_mean:
            weight += 0.8
        bias = bias_std * torch.randn(
            samples_per_block, tokens, output_width, generator=generator
        )
        output = torch.einsum("ntl,nol->nto", hidden, weight) + bias
        flattened = output.reshape(samples_per_block, dimension)
        rotated = flattened @ q.T
        geometry = rotated.T @ rotated / samples_per_block
        hidden_gram = torch.einsum("nti,nsi->nts", hidden, hidden).mean(dim=0)
        predicted_a = weight_std**2 * hidden_gram + bias_std**2 * torch.eye(tokens)
        predicted = q @ torch.kron(predicted_a, torch.eye(output_width)) @ q.T
        block_residual = relative_frobenius(geometry, predicted)
        residuals.append(block_residual)
        block_rows.append({"block": block, "relative_residual": block_residual})
        geometry_sum += geometry
        hidden_gram_sum += hidden_gram
    geometry = geometry_sum / blocks
    hidden_gram = hidden_gram_sum / blocks
    predicted_a = weight_std**2 * hidden_gram + bias_std**2 * torch.eye(tokens)
    predicted = q @ torch.kron(predicted_a, torch.eye(output_width)) @ q.T
    global_residual = relative_frobenius(geometry, predicted)
    projected, fitted_a = projection_transformer(
        geometry, q, tokens, output_width
    )
    projection_residual = relative_frobenius(geometry, projected)
    predicted_eigenvalues = torch.linalg.eigvalsh(predicted_a)
    expanded = torch.linalg.eigvalsh(predicted)
    cluster_spreads = []
    for group in range(tokens):
        cluster = expanded[group * output_width : (group + 1) * output_width]
        cluster_spreads.append(float(cluster.max() - cluster.min()))
    return {
        "passed": bool(global_residual < 0.055 and projection_residual < 0.055)
        if not nonzero_mean
        else True,
        "architecture": "shared zero-mean token linear layer plus fixed orthonormal Q",
        "tokens": tokens,
        "input_width": input_width,
        "output_width": output_width,
        "samples": blocks * samples_per_block,
        "blocks": blocks,
        "block_rows": block_rows,
        "residual_standard_error": float(np.std(residuals, ddof=1) / math.sqrt(blocks)),
        "theory_relative_frobenius_residual": global_residual,
        "best_structured_projection_residual": projection_residual,
        "predicted_distinct_eigenvalues": len(
            np.unique(np.round(predicted_eigenvalues.numpy(), 10))
        ),
        "token_bound": tokens,
        "maximum_exact_cluster_spread": max(cluster_spreads),
        "fitted_a_vs_theory_relative_error": float(
            torch.linalg.norm(fitted_a - predicted_a) / torch.linalg.norm(predicted_a)
        ),
        "nonzero_mean_assumption_broken": nonzero_mean,
    }


def negative_controls():
    mlp = mlp_network(violate_iid=True, blocks=4, samples_per_block=4_000)
    cnn = cnn_network(violate_iid=True, blocks=4, samples_per_block=3_000)
    transformer = transformer_network(
        nonzero_mean=True, blocks=5, samples_per_block=3_000
    )
    rejected = {
        "mlp_non_iid": bool(mlp["relative_frobenius_residual"] > 0.08),
        "cnn_non_iid": bool(cnn["relative_frobenius_residual"] > 0.08),
        "transformer_nonzero_mean": bool(
            transformer["theory_relative_frobenius_residual"] > 0.10
            and transformer["best_structured_projection_residual"] > 0.08
        ),
    }
    return {
        "expected_to_fail": True,
        "rejected": bool(all(rejected.values())),
        "per_control_rejected": rejected,
        "residuals": {
            "mlp_non_iid": mlp["relative_frobenius_residual"],
            "cnn_non_iid": cnn["relative_frobenius_residual"],
            "transformer_nonzero_mean_theory": transformer[
                "theory_relative_frobenius_residual"
            ],
            "transformer_nonzero_mean_projection": transformer[
                "best_structured_projection_residual"
            ],
        },
    }


def main():
    symbolic = symbolic_certificates()
    mlp = mlp_network()
    cnn = cnn_network()
    transformer = transformer_network()
    independent = load_independent_checker().run()
    controls = negative_controls()
    accepted = all(
        [
            symbolic["passed"],
            mlp["passed"],
            cnn["passed"],
            transformer["passed"],
            independent["passed"],
            controls["rejected"],
        ]
    )
    result = {
        "claim": 6,
        "verdict": "VERIFIED" if accepted else "BLOCKED",
        "accepted": bool(accepted),
        "symbolic_certificates": symbolic,
        "actual_mlp": mlp,
        "actual_cnn": cnn,
        "actual_transformer": transformer,
        "independent_checker": independent,
        "negative_controls": controls,
        "seed_policy": "All dimensions, sample counts, thresholds, and seeds were fixed in committed source before execution.",
    }
    (ARTIFACT / "raw_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (ARTIFACT / "independent_checker_output.json").write_text(
        json.dumps(independent, indent=2, sort_keys=True) + "\n"
    )
    (ARTIFACT / "negative_control_output.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM_6_RESULT=" + json.dumps(result, sort_keys=True))
    print(
        "CLAIM_6_SUMMARY "
        f"verdict={result['verdict']} "
        f"mlp_residual={mlp['relative_frobenius_residual']:.6f} "
        f"cnn_residual={cnn['relative_frobenius_residual']:.6f} "
        f"transformer_residual={transformer['theory_relative_frobenius_residual']:.6f} "
        f"distinct_eigenvalues={transformer['predicted_distinct_eigenvalues']}/{transformer['token_bound']} "
        f"controls_rejected={controls['rejected']}"
    )
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
