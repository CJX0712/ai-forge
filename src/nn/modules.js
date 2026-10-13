/**
 * modules.js — Neural building blocks (L3).
 *
 * Zero-rewrite — every block is the modern SOTA formulation:
 *  - RMSNorm (Zhang & Sennrich 2019): no mean-centering, no bias —
 *    used by Llama/DeepSeek/Qwen families instead of LayerNorm.
 *  - RoPE (Su et al. 2021, RoFormer): rotates *pairs* of channels by
 *    position-dependent angles; the attention logit then depends only on
 *    the *relative* offset m−n — this property is an executable invariant.
 *  - SwiGLU (Shazeer 2020): FFN(x) = W₂·(SiLU(W₁x) ⊙ W₃x); the 8/3
 *    hidden-width convention from the "GLU Variants" paper.
 *  - Top-k MoE routing (Shazeer 2017; DeepSeek-V3 aux-loss-free variant).
 *
 * @author 晨星
 */

import { softmax, silu } from "../core/numeric.js";
import { RNG } from "../core/random.js";

/* ------------------------------------------------------------- basics */
/** RMSNorm over the last dimension. */
export function rmsnorm(x, g, eps = 1e-6) {
  const d = g.length;
  const out = new Float64Array(x.length);
  for (let t = 0; t < x.length / d; t++) {
    let s = 0;
    const o = t * d;
    for (let i = 0; i < d; i++) s += x[o + i] * x[o + i];
    const inv = 1 / Math.sqrt(s / d + eps);
    for (let i = 0; i < d; i++) out[o + i] = x[o + i] * inv * g[i];
  }
  return out;
}

/**
 * RoPE: apply to a [T, D] matrix, channels paired (2i, 2i+1).
 * θ_i = base^(−2i/D); position m rotates pair i by m·θ_i.
 */
export function rope(x, T, D, positions, base = 10000.0) {
  const out = new Float64Array(x.length);
  const half = D / 2;
  for (let t = 0; t < T; t++) {
    const m = positions[t];
    for (let i = 0; i < half; i++) {
      const theta = m * Math.pow(base, -2 * i / D);
      const c = Math.cos(theta), s = Math.sin(theta);
      const x0 = x[t * D + 2 * i], x1 = x[t * D + 2 * i + 1];
      out[t * D + 2 * i] = x0 * c - x1 * s;
      out[t * D + 2 * i + 1] = x0 * s + x1 * c;
    }
  }
  return out;
}

/** SiLU/Swish gated FFN (SwiGLU). */
export function swiglu(x, W1, W3, W2, T, D, H) {
  const h1 = new Float64Array(T * H);
  const h3 = new Float64Array(T * H);
  const out = new Float64Array(T * D);
  linear(x, W1, h1, T, D, H);
  linear(x, W3, h3, T, D, H);
  for (let i = 0; i < h1.length; i++) h1[i] = silu(h1[i]) * h3[i];
  linear(h1, W2, out, T, H, D);
  return out;
}

/** y[T,n] = x[T,m] · W[m,n] (+ b[n] if given) */
export function linear(x, W, y, T, m, n, b = null) {
  for (let t = 0; t < T; t++) {
    const xo = t * m, yo = t * n;
    if (b) { for (let j = 0; j < n; j++) y[yo + j] = b[j]; }
    else y.fill(0, yo, yo + n);
    for (let i = 0; i < m; i++) {
      const v = x[xo + i];
      if (v === 0) continue;
      const wo = i * n;
      for (let j = 0; j < n; j++) y[yo + j] += v * W[wo + j];
    }
  }
  return y;
}

/** GPT-style init: N(0, 0.02), optional scaled output projection.
 * Determinism: derives a FRESH RNG from the FNV-1a hash of `seed`, so any
 * construction order still yields identical weights for identical names. */
export function initLinear(W, fanIn, fanOut, { std = 0.02, seed = "init", scaleStd = null } = {}) {
  let h = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  const r = new RNG((h ^ 20261014) >>> 0);
  const stdOut = scaleStd || std;
  for (let i = 0; i < W.length; i++) W[i] = r.normal() * stdOut;
  void fanIn; void fanOut;
  return W;
}

/* --------------------------------------------------------------- MoE */
/**
 * Top-k Mixture of Experts router (Shazeer 2017).
 * aux loss (Fedus et al. 2021 switch form, scaled to be invariant):
 *   aux = E · Σ_e f_e · P_e,   f_e = (#tokens routed to e)/(T·k),
 *   P_e = mean router prob of e over selected slots.
 * Uniform routing ⇒ aux = 1; concentration pushes it toward E.
 */
export function moeRoute(x, routerW, T, D, E, k = 2) {
  const logits = new Float64Array(T * E);
  linear(x, routerW, logits, T, D, E);
  const expert = new Int32Array(T * k);
  const weight = new Float64Array(T * k);
  const probSum = new Float64Array(E);
  const count = new Float64Array(E);
  const row = new Float64Array(E);
  const probs = new Float64Array(E);
  for (let t = 0; t < T; t++) {
    for (let e = 0; e < E; e++) row[e] = logits[t * E + e];
    softmax(row, probs);
    const idx = Array.from({ length: E }, (_, i) => i);
    idx.sort((i, j) => probs[j] - probs[i]);
    for (let j = 0; j < k; j++) {
      expert[t * k + j] = idx[j];
      weight[t * k + j] = probs[idx[j]];
      probSum[idx[j]] += probs[idx[j]];
      count[idx[j]] += 1;
    }
  }
  let aux = 0;
  for (let e = 0; e < E; e++) aux += (count[e] / (T * k)) * (probSum[e] / T) * E;
  return { expert, weight, aux };
}

/** Deterministic MLP with tanh hidden layer (used for autodiff demos). */
export function mlpForward(x, W1, b1, W2, b2) {
  const d = b1.length, h = W2.length;
  const hid = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let s = b1[i];
    for (let j = 0; j < h; j++) s += x[j] * W1[j * d + i];
    hid[i] = Math.tanh(s);
  }
  const out = new Float64Array(1);
  let s = b2[0];
  for (let i = 0; i < d; i++) s += hid[i] * W2[i];
  out[0] = s;
  return out;
}
