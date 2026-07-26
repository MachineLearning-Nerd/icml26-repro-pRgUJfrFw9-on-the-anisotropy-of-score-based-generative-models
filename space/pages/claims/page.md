# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6d51f69ddfa9", "created_at": "2026-07-22T22:55:51+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Score Anisotropy Directions (SADs) are defined as the ordered orthonormal eigenvectors of the average geometry matrix G_F = E[(F_θ(x_σ,σ))²], with ordering by eigenvalue magnitude conjectured to determine the network's generation preference (Definition 2).
2. Theorem 1 shows that under denoising score matching with linear architectures and rank-one data, convergence rate decays with the eigenvalues of the conditioning matrix, and this anisotropic effect is amplified further under SGD (Theorem 1).
3. Theorem 2 characterizes the orthogonal transformation that minimizes versus maximizes alignment α with the SADs, showing minimization corresponds to reversing the eigenvalue ordering (Theorem 2).
4. For an iDDPM U-Net, generation performance degrades monotonically as the eigenvalue associated with the aligned data direction increases, with the model performing best on data aligned with small-eigenvalue SAD eigenvectors (Figure 6).
5. On MNIST, CelebA-HQ, and CIFAR-10, minimizing alignment α with the SADs substantially improves Sliced Wasserstein-2 distances relative to the default natural alignment, whereas maximizing alignment induces mode collapse (Figure 8, Figure 9).
6. Closed-form average geometry expressions are derived per architecture: MLPs yield G_F = αI + β·11ᵀ, CNNs yield a block-diagonal Kronecker structure reflecting spatial organization, and Transformers yield at most T distinct eigenvalues where T is the number of tokens (Section on architecture-specific results).
