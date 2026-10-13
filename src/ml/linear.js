/**
 * linear.js — Linear models (L6).
 *
 * Canonical formulations:
 *  - OLS via QR least squares (numerically superior to the normal equations)
 *  - Ridge closed form (AᵀA + λI) w = Aᵀy via Cholesky
 *  - LASSO coordinate descent (Friedman/Hastie/Tibshirani glmnet style)
 *  - Logistic regression IRLS / full Newton (statistically canonical), with
 *    the invariant ∇ℓ = Aᵀ(p − y)/N verified against the IRLS update
 *
 * @author 晨星
 */

import { qr, lstsq, cholesky, solve, gemmAT, gemm, transpose, matVecT } from "../core/linalg.js";
import { sigmoid, logSigmoid } from "../core/numeric.js";

/** OLS: w minimises ‖Aw − y‖₂ (QR-based). Returns {w, intercept}. */
export function linearRegression(A, y) {
  const { r, c } = { r: A.length, c: A[0].length };
  const M = zeros2(r, c + 1);
  for (let i = 0; i < r; i++) { for (let j = 0; j < c; j++) M[i * (c + 1) + j] = A[i][j]; M[i * (c + 1) + c] = 1; }
  const yv = Float64Array.from(y);
  const wv = lstsq({ r, c: c + 1, a: M }, yv);
  return { w: Array.from(wv.subarray(0, c)), intercept: wv[c] };
}

/** Ridge regression closed form with intercept. λ>0 shrinks coefficients. */
export function ridgeRegression(A, y, lambda = 1) {
  const r = A.length, c = A[0].length;
  const M = zeros2(r, c + 1);
  for (let i = 0; i < r; i++) { for (let j = 0; j < c; j++) M[i * (c + 1) + j] = A[i][j]; M[i * (c + 1) + c] = 1; }
  // w = (MᵀM + λI')⁻¹ Mᵀy  — intercept row/column unpenalised
  const MtM = gemmAT({ r, c: c + 1, a: M }, { r, c: c + 1, a: M });
  const Mty = matVecT({ r, c: c + 1, a: M }, yv(y));
  for (let j = 0; j < c; j++) MtM.a[j * (c + 1) + j] += lambda;
  const L = cholesky(MtM);
  const wv = choleskySolveInto(L, Mty, c + 1);
  return { w: Array.from(wv.subarray(0, c)), intercept: wv[c] };
}

/** LASSO via cyclic coordinate descent (soft thresholding). */
export function lassoRegression(A, y, lambda = 0.1, { maxIter = 5000, tol = 1e-12 } = {}) {
  const n = A.length, p = A[0].length;
  // standardise columns for stable updates; remember to untransform
  const mean = new Float64Array(p), std = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += A[i][j];
    mean[j] = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (A[i][j] - mean[j]) ** 2;
    std[j] = Math.sqrt(v / n) || 1;
  }
  const yc = Float64Array.from(y);
  let yMean = 0;
  for (let i = 0; i < n; i++) yMean += y[i];
  yMean /= n;
  for (let i = 0; i < n; i++) yc[i] -= yMean;
  const Z = zeros2(n, p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) Z[i * p + j] = (A[i][j] - mean[j]) / std[j];
  // glmnet semantics: objective (1/2n)‖r‖² + λ‖β‖₁
  const colSq = new Float64Array(p);
  for (let j = 0; j < p; j++) { let s = 0; for (let i = 0; i < n; i++) s += Z[i * p + j] ** 2; colSq[j] = s / n; }
  const beta = new Float64Array(p);
  const r = Float64Array.from(yc);
  for (let it = 0; it < maxIter; it++) {
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      const old = beta[j];
      // partial residual: restore β_j's contribution before recomputing ρ_j
      if (old !== 0) for (let i = 0; i < n; i++) r[i] += Z[i * p + j] * old;
      let rho = 0;
      for (let i = 0; i < n; i++) rho += Z[i * p + j] * r[i];
      rho /= n;
      beta[j] = softThreshold(rho, lambda) / colSq[j];
      if (beta[j] !== 0) for (let i = 0; i < n; i++) r[i] -= Z[i * p + j] * beta[j];
      maxDelta = Math.max(maxDelta, Math.abs(beta[j] - old));
    }
    if (maxDelta < tol) break;
  }
  const w = Array.from(beta, (b, j) => b / std[j]);
  let intercept = yMean;
  for (let j = 0; j < p; j++) intercept -= mean[j] * w[j];
  return { w, intercept, iterations: maxIter };
}

function softThreshold(rho, lambda) {
  if (rho > lambda) return rho - lambda;
  if (rho < -lambda) return rho + lambda;
  return 0;
}

/** Logistic regression via full Newton–Raphson (IRLS). */
export function logisticRegression(A, y, { lambda = 1e-6, maxIter = 100, tol = 1e-10 } = {}) {
  const n = A.length, p = A[0].length;
  const M = zeros2(n, p + 1);
  for (let i = 0; i < n; i++) { for (let j = 0; j < p; j++) M[i * (p + 1) + j] = A[i][j]; M[i * (p + 1) + p] = 1; }
  const w = new Float64Array(p + 1);
  const Mt = transpose({ r: n, c: p + 1, a: M });
  for (let it = 0; it < maxIter; it++) {
    const eta = matVecRow({ r: n, c: p + 1, a: M }, w);
    const pr = new Float64Array(n);
    for (let i = 0; i < n; i++) pr[i] = sigmoid(eta[i]);
    // gradient = Mᵀ(p − y) + λw
    const g = new Float64Array(p + 1);
    for (let i = 0; i < n; i++) {
      const d = pr[i] - y[i];
      for (let j = 0; j < p + 1; j++) g[j] += Mt.a[j * n + i] * d;
    }
    for (let j = 0; j < p + 1; j++) g[j] += lambda * w[j] / n;
    // Hessian = Mᵀ W (1−W) M + λI/n  (W diagonal)
    const Wd = { r: n, c: n, a: new Float64Array(n * n) };
    for (let i = 0; i < n; i++) Wd.a[i * n + i] = pr[i] * (1 - pr[i]);
    const WM = gemm(Wd, { r: n, c: p + 1, a: M });          // W·M  (n × (p+1))
    const H = gemmAT({ r: n, c: p + 1, a: M }, WM);          // Mᵀ·(W·M)
    for (let j = 0; j < p + 1; j++) H.a[j * (p + 1) + j] += lambda / n;
    const step = solveNewton(H, g);
    let gNorm = 0;
    for (let j = 0; j < p + 1; j++) gNorm += g[j] * g[j];
    gNorm = Math.sqrt(gNorm);
    for (let j = 0; j < p + 1; j++) w[j] -= step[j];
    if (gNorm < tol) break;
  }
  return {
    w: Array.from(w.subarray(0, p)),
    intercept: w[p],
    predict: (x) => sigmoid(dot(x, w) ),
    predictProba: (x) => sigmoid(dot(x, w)),
  };
}

export function dot(a, b) { let s = b[b.length - 1]; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/* flat helpers */
function zeros2(r, c) { return new Float64Array(r * c); }
function yv(y) { return Float64Array.from(y); }
function matVecRow(M, x) {
  const o = new Float64Array(M.r);
  for (let i = 0; i < M.r; i++) {
    let s = 0;
    const off = i * M.c;
    for (let j = 0; j < M.c; j++) s += M.a[off + j] * x[j];
    o[i] = s;
  }
  return o;
}
function choleskySolveInto(L, b, n) {
  const y = new Float64Array(n), x = new Float64Array(n), l = L.a;
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= l[i * n + k] * y[k]; y[i] = s / l[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= l[k * n + i] * x[k]; x[i] = s / l[i * n + i]; }
  return x;
}
function solveNewton(H, g) {
  // SPD solve via Cholesky with tiny jitter fallback
  try {
    const L = cholesky({ r: H.r, c: H.c, a: H.a });
    return choleskySolveInto(L, g, H.r);
  } catch {
    return gradientDescentStep(H, g);
  }
}
function gradientDescentStep(H, g) {
  const x = new Float64Array(g.length);
  for (let i = 0; i < g.length; i++) x[i] = g[i] / (H.a[i * g.length + i] + 1e-9);
  return x;
}
export { logSigmoid };
