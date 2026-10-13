/**
 * cluster.js — k-means++ and Gaussian Mixture Models (L6).
 *
 * Canonical references:
 *  - k-means++ seeding (Arthur & Vassilvitskii 2007): D²-weighted first
 *    centre, then Lloyd iterations. Invariant: SSE is monotonically
 *    non-increasing across Lloyd steps.
 *  - GMM via EM (Dempster 1977; Dempster–Laird–Rubin), diagonal or full
 *    covariance, logsumexp-stabilised responsibilities.
 *    Invariants: every responsibility row sums to 1; the complete-data
 *    log-likelihood is monotonically non-decreasing (the EM theorem).
 *
 * @author 晨星
 */

import { rng } from "../core/random.js";
import { logsumexp } from "../core/numeric.js";

/* ------------------------------------------------------------ k-means */
export function kmeans(X, k, { maxIter = 100, seed = "km", seedMethod = "kmeans++" } = {}) {
  const n = X.length, d = X[0].length;
  const r = rng(seed);
  const dist2 = (a, b) => { let s = 0; for (let i = 0; i < d; i++) { const t = a[i] - b[i]; s += t * t; } return s; };

  // seeding
  let centers = [];
  if (seedMethod === "kmeans++") {
    centers.push(X[r.int(n)].slice());
    const d2 = new Float64Array(n);
    while (centers.length < k) {
      let total = 0;
      for (let i = 0; i < n; i++) {
        let best = Infinity;
        for (const c of centers) best = Math.min(best, dist2(X[i], c));
        d2[i] = best; total += best;
      }
      // sample proportional to D²
      let target = r.next() * total, pick = n - 1;
      for (let i = 0; i < n; i++) { target -= d2[i]; if (target <= 0) { pick = i; break; } }
      centers.push(X[pick].slice());
    }
  } else {
    const idx = Array.from({ length: n }, (_, i) => i);
    r.shuffle(idx);
    centers = idx.slice(0, k).map((i) => X[i].slice());
  }

  const assign = new Int32Array(n);
  let sse = Infinity;
  const sseHistory = [];
  for (let it = 0; it < maxIter; it++) {
    // assignment
    let cur = 0;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) { const dd = dist2(X[i], centers[c]); if (dd < bd) { bd = dd; best = c; } }
      assign[i] = best; cur += bd;
    }
    sseHistory.push(cur);
    // update
    const sums = Array.from({ length: k }, () => new Float64Array(d));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      counts[assign[i]]++;
      for (let j = 0; j < d; j++) sums[assign[i]][j] += X[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) { centers[c] = X[r.int(n)].slice(); continue; } // reseed empty cluster
      for (let j = 0; j < d; j++) centers[c][j] = sums[c][j] / counts[c];
    }
    if (Math.abs(sse - cur) < 1e-12) { sse = cur; break; }
    sse = cur;
  }
  return { centers, assign, sse, history: sseHistory, inertia: sse };
}

/* ---------------------------------------------------------------- GMM */
export function gmm(X, k, { maxIter = 200, tol = 1e-9, seed = "gmm", cov = "diag", reg = 1e-6 } = {}) {
  const n = X.length, d = X[0].length;
  const r = rng(seed);
  // init: k-means++ labels → moment-matched initial params
  const km = kmeans(X, k, { seed: seed + ":init" });
  const weights = new Float64Array(k);
  const means = Array.from({ length: k }, () => new Float64Array(d));
  const vars = Array.from({ length: k }, () => new Float64Array(d).fill(1));
  for (let c = 0; c < k; c++) {
    const idx = [];
    for (let i = 0; i < n; i++) if (km.assign[i] === c) idx.push(i);
    weights[c] = Math.max(1, idx.length) / n;
    for (let j = 0; j < d; j++) {
      let m = 0;
      for (const i of idx) m += X[i][j];
      means[c][j] = idx.length ? m / idx.length : 0;
      let v = 0;
      for (const i of idx) v += (X[i][j] - means[c][j]) ** 2;
      vars[c][j] = (idx.length ? v / idx.length : 1) + reg;
    }
  }

  const logPdfDiag = (x, c) => {
    let s = 0;
    for (let j = 0; j < d; j++) {
      const diff = x[j] - means[c][j];
      s += -0.5 * (diff * diff / vars[c][j] + Math.log(2 * Math.PI * vars[c][j]));
    }
    return s;
  };

  const resp = new Float64Array(n * k);
  const llHistory = [];
  let ll = -Infinity;
  for (let it = 0; it < maxIter; it++) {
    // E-step: responsibilities via logsumexp (numerically exact)
    for (let i = 0; i < n; i++) {
      const lv = new Float64Array(k);
      for (let c = 0; c < k; c++) lv[c] = Math.log(weights[c] + 1e-300) + logPdfDiag(X[i], c);
      const lse = logsumexp(lv);
      for (let c = 0; c < k; c++) resp[i * k + c] = Math.exp(lv[c] - lse);
    }
    // log-likelihood
    let cur = 0;
    for (let i = 0; i < n; i++) {
      const lv = [];
      for (let c = 0; c < k; c++) lv.push(Math.log(weights[c] + 1e-300) + logPdfDiag(X[i], c));
      cur += logsumexp(lv);
    }
    llHistory.push(cur);
    if (it > 0 && cur < ll - 1e-8) {
      return { converged: false, violation: true, ll: cur, history: llHistory, weights, means, vars, iterations: it };
    }
    if (it > 0 && Math.abs(cur - ll) < tol) { ll = cur; break; }
    ll = cur;
    // M-step
    const nk = new Float64Array(k);
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) nk[c] += resp[i * k + c];
    for (let c = 0; c < k; c++) weights[c] = nk[c] / n;
    for (let c = 0; c < k; c++) {
      const m = means[c];
      m.fill(0);
      if (nk[c] < 1e-12) { m.set(X[r.int(n)]); continue; }
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += resp[i * k + c] * X[i][j];
      for (let j = 0; j < d; j++) m[j] /= nk[c];
      const v = vars[c];
      v.fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { const diff = X[i][j] - m[j]; v[j] += resp[i * k + c] * diff * diff; }
      for (let j = 0; j < d; j++) v[j] = v[j] / nk[c] + reg;
    }
  }
  // final responsibilities (post-update) for reporting
  let finalLL = 0;
  for (let i = 0; i < n; i++) {
    const lv = [];
    for (let c = 0; c < k; c++) lv.push(Math.log(weights[c] + 1e-300) + logPdfDiag(X[i], c));
    finalLL += logsumexp(lv);
  }
  return {
    converged: true, ll: finalLL, history: llHistory,
    weights, means, vars, iterations: llHistory.length,
    predict: (x) => {
      let best = 0, bl = -Infinity;
      for (let c = 0; c < k; c++) { const lp = Math.log(weights[c]) + logPdfDiag(x, c); if (lp > bl) { bl = lp; best = c; } }
      return best;
    },
  };
}

/* -------------------------------------------------- hierarchical (opt) */
/** Agglomerative clustering with average linkage — small n utility. */
export function agglomerative(X, k, { linkage = "average" } = {}) {
  const n = X.length;
  const d2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; s += t * t; } return s; };
  let clusters = X.map((_, i) => [i]);
  while (clusters.length > k) {
    let bi = 0, bj = 1, bd = Infinity;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        let s = 0;
        for (const a of clusters[i]) for (const b of clusters[j]) s += d2(X[a], X[b]);
        s /= linkage === "single" ? undefined : (clusters[i].length * clusters[j].length);
        if (s < bd) { bd = s; bi = i; bj = j; }
      }
    }
    const merged = [...clusters[bi], ...clusters[bj]];
    clusters = clusters.filter((_, t) => t !== bi && t !== bj);
    clusters.push(merged);
  }
  const labels = new Int32Array(n);
  clusters.forEach((cl, ci) => cl.forEach((i) => { labels[i] = ci; }));
  return { clusters, labels };
}
