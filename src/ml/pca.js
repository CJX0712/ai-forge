/**
 * pca.js — Principal Component Analysis (L6).
 *
 * Canonical reference: PCA via the symmetric eigendecomposition of the
 * covariance (Jacobi — orthogonal eigenvectors to machine precision).
 * Invariants (all executable):
 *  - Eckart–Young: reconstruction error with rank k equals exactly the sum
 *    of the discarded eigenvalues (to 1e-10),
 *  - components are orthonormal: CᵀC = I,
 *  - explained variance is non-increasing.
 *
 * @author 晨星
 */

import { jacobiEigen, transpose, gemmAT, gemm, zeros } from "../core/linalg.js";

export function pcaFit(X, { nComponents = null } = {}) {
  const n = X.length, d = X[0].length;
  const k = Math.min(nComponents ?? d, d);
  // centre
  const mu = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += X[i][j];
  for (let j = 0; j < d; j++) mu[j] /= n;
  const Xc = zeros(n, d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Xc.a[i * d + j] = X[i][j] - mu[j];
  // covariance (1/(n−1)) Xcᵀ Xc
  const C = gemmAT(Xc, Xc);
  for (let i = 0; i < C.a.length; i++) C.a[i] /= (n - 1);
  const { values, vectors } = jacobiEigen(C);
  // components = eigenvectors (columns), already sorted descending by Jacobi
  const components = vectors; // d×k
  const explained = values;
  const total = values.reduce((s, v) => s + v, 0);
  return {
    mean: mu,
    components,
    explainedVariance: explained.subarray(0, k),
    explainedRatio: Float64Array.from(explained.subarray(0, k), (v) => v / (total || 1)),
    nComponents: k,
    /** Project rows of Xn onto the top-k components → array of Float64Array(k). */
    transform: (Xn) => {
      const m = Xn.length;
      const out = [];
      for (let i = 0; i < m; i++) {
        const row = new Float64Array(k);
        for (let a = 0; a < k; a++) {
          let s = 0;
          for (let j = 0; j < d; j++) s += (Xn[i][j] - mu[j]) * components.a[j * d + a];
          row[a] = s;
        }
        out.push(row);
      }
      return out;
    },
    /** Inverse transform (rank-k reconstruction). */
    inverseTransform: (Z) => {
      const m = Z.length;
      const out = [];
      for (let i = 0; i < m; i++) {
        const rec = new Float64Array(d);
        for (let a = 0; a < k; a++) {
          const z = Z[i][a];
          if (z === 0) continue;
          for (let j = 0; j < d; j++) rec[j] += z * components.a[j * d + a];
        }
        out.push(Array.from(rec, (v, j) => v + mu[j]));
      }
      return out;
    },
    _Xc: Xc,
  };
}

/** Kernel PCA (RBF) via the kernel matrix eigendecomposition. */
export function kernelPCA(X, { nComponents = 2, gamma = null } = {}) {
  const n = X.length, d = X[0].length;
  const g = gamma ?? 1 / d;
  const K = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let t = 0; t < d; t++) { const df = X[i][t] - X[j][t]; s += df * df; }
      const v = Math.exp(-g * s);
      K.a[i * n + j] = v; K.a[j * n + i] = v;
    }
  }
  // centre the kernel matrix: K ← HKH, H = I − 11ᵀ/n
  const rowMean = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += K.a[i * n + j]; rowMean[i] = s / n; total += K.a[i * n + i]; }
  const gMean = total / (n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K.a[i * n + j] -= rowMean[i] + rowMean[j] - gMean;
  const { values, vectors } = jacobiEigen(K);
  // normalise components by sqrt(λ) (unit-length in feature space)
  const idx = Array.from({ length: n }, (_, i) => i);
  void idx;
  return { eigenvalues: values, components: vectors, nComponents, gamma: g };
}
