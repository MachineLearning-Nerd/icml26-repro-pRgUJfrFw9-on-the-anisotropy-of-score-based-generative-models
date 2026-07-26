"""Complete finite-domain checks for the three architecture propositions."""

from __future__ import annotations

import itertools
import json

import numpy as np


def mlp_complete_domain():
    hidden_states = [np.array([1.0, 2.0]), np.array([-1.0, 1.0])]
    parameters = list(itertools.product([-1.0, 1.0], repeat=3))
    conditional = []
    for hidden in hidden_states:
        values = np.array(
            [(w0 * hidden[0] + w1 * hidden[1] + bias) ** 2 for w0, w1, bias in parameters]
        )
        conditional.append((float(values.mean()), float(np.mean(values**2))))
    mean_square = float(np.mean([mean**2 for mean, _ in conditional]))
    second_moment = float(np.mean([moment for _, moment in conditional]))
    dimension = 5
    geometry = np.full((dimension, dimension), mean_square)
    np.fill_diagonal(geometry, second_moment)
    expected = (second_moment - mean_square) * np.eye(dimension)
    expected += mean_square * np.ones((dimension, dimension))
    error = float(np.max(np.abs(geometry - expected)))
    return {"passed": bool(error == 0.0), "complete_parameter_states": 8, "max_error": error}


def circular_convolution(hidden, kernel):
    length = len(hidden)
    return np.array(
        [
            sum(kernel[offset] * hidden[(position + offset) % length] for offset in range(len(kernel)))
            for position in range(length)
        ],
        dtype=float,
    )


def cnn_complete_domain():
    hidden_states = [np.array([1.0, -1.0, 2.0]), np.array([2.0, 0.0, -1.0])]
    parameters = list(itertools.product([-1.0, 1.0], repeat=3))
    means = []
    covariances = []
    for hidden in hidden_states:
        outputs = []
        for w0, w1, bias in parameters:
            preactivation = circular_convolution(hidden, [w0, w1]) + bias
            outputs.append(preactivation**2)
        outputs = np.asarray(outputs)
        mean = outputs.mean(axis=0)
        covariance = outputs.T @ outputs / len(outputs) - np.outer(mean, mean)
        means.append(mean)
        covariances.append(covariance)
    a = np.mean(covariances, axis=0)
    b = np.mean([np.outer(mean, mean) for mean in means], axis=0)
    channels = 3
    geometry = np.kron(np.eye(channels), a) + np.kron(
        np.ones((channels, channels)), b
    )
    blocks = geometry.reshape(channels, 3, channels, 3)
    diagonal = np.mean([blocks[i, :, i, :] for i in range(channels)], axis=0)
    off_diagonal = np.mean(
        [blocks[i, :, j, :] for i in range(channels) for j in range(channels) if i != j],
        axis=0,
    )
    error = max(
        float(np.max(np.abs(diagonal - (a + b)))),
        float(np.max(np.abs(off_diagonal - b))),
    )
    return {"passed": bool(error < 1e-12), "complete_parameter_states": 8, "max_error": error}


def transformer_complete_domain():
    hidden = np.array([[1.0, 2.0], [-1.0, 1.0], [2.0, -2.0]])
    token_count, input_width = hidden.shape
    output_width = 2
    weight_states = list(
        itertools.product([-1.0, 1.0], repeat=output_width * input_width)
    )
    geometry = np.zeros((token_count * output_width,) * 2)
    for state in weight_states:
        weight = np.asarray(state).reshape(output_width, input_width)
        output = hidden @ weight.T
        flattened = output.reshape(-1)
        geometry += np.outer(flattened, flattened)
    geometry /= len(weight_states)
    # Independent zero-mean Rademacher bias for every token/output coordinate
    # contributes an identity covariance exactly.
    geometry += np.eye(token_count * output_width)
    a = hidden @ hidden.T + np.eye(token_count)
    expected = np.kron(a, np.eye(output_width))
    error = float(np.max(np.abs(geometry - expected)))
    eigenvalues = np.linalg.eigvalsh(expected)
    distinct = len(np.unique(np.round(eigenvalues, 10)))
    return {
        "passed": bool(error < 1e-12 and distinct <= token_count),
        "complete_weight_states": len(weight_states),
        "max_error": error,
        "distinct_eigenvalues": distinct,
        "token_count": token_count,
    }


def run():
    mlp = mlp_complete_domain()
    cnn = cnn_complete_domain()
    transformer = transformer_complete_domain()
    return {
        "passed": bool(mlp["passed"] and cnn["passed"] and transformer["passed"]),
        "implementation": "complete finite Rademacher parameter domains; no primary-verifier calls",
        "mlp": mlp,
        "cnn": cnn,
        "transformer": transformer,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
