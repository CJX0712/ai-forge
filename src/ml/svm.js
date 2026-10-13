/**
 * svm.js — Soft-margin SVM trained with SMO (L6).
 *
 * Canonical reference: Platt 1998 (Sequential Minimal Optimization) with the
 * simplified heuristic set: 1st-choice violator + 2nd-choice maximiser.
 * Invariants enforced by the test suite:
 *   Σ αᵢyᵢ = 0 (to 1e-10),
 *   KKT complementary slackness: αᵢ=0 ⇒ yᵢf(xᵢ) ≥ 1−tol,
 *   0<αᵢ<C ⇒ yᵢf(xᵢ) = 1±tol.
 *
 * @author 晨星
 */

export function svmTrain(X, y, { C = 1.0, tol = 1e-3, maxPasses = 8, kernel = null, gamma = null } = {}) {
  const n = X.length, d = X[0].length;
  const K = kernel || linearKernel;
  const g = gamma ?? 1 / d;
  const kern = (i, j) => kernel ? kernel(X[i], X[j], g) : dotP(X[i], X[j]);
  const alpha = new Float64Array(n);
  const b = { v: 0 };
  const E = new Float64Array(n);
  const precomputed = kernel ? buildGram(X, g, kernel) : null;
  const kfun = precomputed ? (i, j) => precomputed[i * n + j] : kern;

  const f = (i) => {
    let s = b.v;
    for (let j = 0; j < n; j++) if (alpha[j] !== 0) s += alpha[j] * y[j] * kfun(i, j);
    return s;
  };
  const updateE = (i) => { E[i] = f(i) - y[i]; };
  for (let i = 0; i < n; i++) updateE(i);

  let passes = 0;
  let iter = 0;
  /** Attempt one α_i/α_j joint update (Platt §2.2). Returns true if changed. */
  const tryPair = (i, j) => {
    if (i === j) return false;
    const Ei = E[i], Ej = E[j];
    const aiOld = alpha[i], ajOld = alpha[j];
    let L, H;
    if (y[i] !== y[j]) { L = Math.max(0, alpha[j] - alpha[i]); H = Math.min(C, C + alpha[j] - alpha[i]); }
    else { L = Math.max(0, alpha[i] + alpha[j] - C); H = Math.min(C, alpha[i] + alpha[j]); }
    if (L === H) return false;
    const eta = 2 * kfun(i, j) - kfun(i, i) - kfun(j, j);
    if (eta >= 0) return false;
    let aj = ajOld - (y[j] * (Ei - Ej)) / eta;
    aj = Math.min(H, Math.max(L, aj));
    if (Math.abs(aj - ajOld) < 1e-12) return false;
    const ai = aiOld + y[i] * y[j] * (ajOld - aj);
    // b update (Platt eq. 12/13)
    const b1 = b.v - Ei - y[i] * (ai - aiOld) * kfun(i, i) - y[j] * (aj - ajOld) * kfun(i, j);
    const b2 = b.v - Ej - y[i] * (ai - aiOld) * kfun(i, j) - y[j] * (aj - ajOld) * kfun(j, j);
    if (0 < ai && ai < C) b.v = b1;
    else if (0 < aj && aj < C) b.v = b2;
    else b.v = (b1 + b2) / 2;
    alpha[i] = ai; alpha[j] = aj;
    for (let t = 0; t < n; t++) updateE(t);
    return true;
  };

  while (passes < maxPasses && iter < 30000) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = E[i];
      const rI = y[i] * Ei;
      const violates = (rI < -tol && alpha[i] < C) || (rI > tol && alpha[i] > 0);
      if (!violates) continue;
      // Platt's 2nd-choice heuristic, with full fallback sweep (§2.2):
      // if the best |Ei−Ej| candidate fails, sweep every other index.
      let jBest = -1, best = -1;
      for (let t = 0; t < n; t++) {
        if (t === i) continue;
        const diff = Math.abs(Ei - E[t]);
        if (diff > best) { best = diff; jBest = t; }
      }
      let done = tryPair(i, jBest);
      if (!done) {
        // start from a random point and sweep (Platt's non-bound fallback)
        const start = Math.floor(Math.random() * n);
        for (let s = 0; s < n && !done; s++) {
          const t = (start + s) % n;
          if (t === i) continue;
          done = tryPair(i, t);
        }
      }
      if (done) { changed++; iter++; }
    }
    passes = changed === 0 ? passes + 1 : 0;
  }
  const sv = [];
  for (let i = 0; i < n; i++) if (alpha[i] > 1e-8) sv.push({ index: i, alpha: alpha[i], x: X[i], y: y[i] });
  return {
    alpha, b: b.v, supportVectors: sv, C, kernel: K, gamma: g,
    decision(x) {
      let s = b.v;
      for (const v of sv) s += v.alpha * v.y * (kernel ? kernel(v.x, x, g) : dotP(v.x, x));
      return s;
    },
    predict(x) { return this.decision(x) >= 0 ? 1 : -1; },
  };
}

/** RBF kernel — the default non-linear workhorse. */
export const rbfKernel = (a, b, gamma) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.exp(-gamma * s);
};
export const linearKernel = (a, b) => dotP(a, b);
export const polynomialKernel = (a, b, { gamma = 1, r = 0, degree = 3 } = {}) => {
  let s = r;
  for (let i = 0; i < a.length; i++) s += gamma * a[i] * b[i];
  let p = 1;
  for (let k = 0; k < degree; k++) p *= s;
  return p;
};
export function buildGram(X, gamma, kernel) {
  const n = X.length;
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const v = kernel(X[i], X[j], gamma); G[i * n + j] = v; G[j * n + i] = v; }
  return G;
}
function dotP(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
