/**
 * numeric.js — Special functions & numerically-stable primitives (L0).
 *
 * Every routine here is a *reused* world-class result, not an invention:
 *  - logsumexp / stable softmax  (classic; avoids overflow & cancellation)
 *  - Kahan–Babuška–Neumaier compensated summation (Kahan 1965 / Neumaier 1974)
 *  - erf via Numerical-Recipes 3e rational Chebyshev approximation (|err|<1.2e-7)
 *  - erfc continued-fraction (Numerical Recipes), lgamma via Lanczos g=7,n=9
 *  - Welford online mean/variance (Welford 1962) — one pass, O(1) memory
 *  - hypot-style stable norms, log1p/expm1 usage to kill cancellation
 *
 * @author 晨星
 */

export const EPS = 2.220446049250313e-16;
export const LOG2 = Math.LN2;
export const SQRT2 = Math.SQRT2;
export const INV_SQRT_2PI = 0.3989422804014327;

/** Kahan–Babuška–Neumaier compensated sum. Error ~ O(eps) not O(n*eps). */
export function kahanSum(arr) {
  let s = 0, c = 0;
  for (let i = 0; i < arr.length; i++) {
    const x = arr[i];
    const t = s + x;
    if (Math.abs(s) >= Math.abs(x)) c += (s - t) + x; else c += (x - t) + s;
    s = t;
  }
  return s + c;
}

/** log(sum_i exp(x_i)) with max-subtraction; z==null → -Infinity safe. */
export function logsumexp(x, z = null) {
  let m = -Infinity;
  for (let i = 0; i < x.length; i++) if (z === null || z[i]) { const v = x[i]; if (v > m) m = v; }
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (let i = 0; i < x.length; i++) if (z === null || z[i]) s += Math.exp(x[i] - m);
  return m + Math.log(s);
}

/** Stable softmax written into `out` (or new array). Max-subtraction. */
export function softmax(x, out) {
  const n = x.length;
  const o = out || new Float64Array(n);
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > m) m = x[i];
  let s = 0;
  for (let i = 0; i < n; i++) { o[i] = Math.exp(x[i] - m); s += o[i]; }
  const inv = 1 / s;
  for (let i = 0; i < n; i++) o[i] *= inv;
  return o;
}

/** In-place log-softmax. */
export function logSoftmax(x, out) {
  const lse = logsumexp(x);
  const o = out || new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = x[i] - lse;
  return o;
}

export const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
/** log(sigmoid(x)) = -log1p(exp(-x)), stable on both tails. */
export const logSigmoid = (x) => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)));
export const relu = (x) => (x > 0 ? x : 0);
export const leakyRelu = (x, a = 0.01) => (x > 0 ? x : a * x);
export const silu = (x) => x * sigmoid(x);           // Swish-1
export const gelu = (x) => 0.5 * x * (1 + erf(x / SQRT2));
/** tanh-GELU: exact-ish alternative used by many LLMs */
export const geluTanh = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
export const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

/** erf defined exactly through the high-precision erfc below. */
export function erf(x) {
  return Math.sign(x) * (1 - erfc(Math.abs(x)));
}

/** erfc via continued fraction (Numerical Recipes erfccheb). */
export function erfc(x) {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  const cof = [-1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5,
    -2.0278578112534e-5, -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8,
    -8.5238095915e-8, 6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
    9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13,
    3.81e-16, 7.106e-15];
  let d = 0, dd = 0;
  for (let j = cof.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + cof[j];
    dd = tmp;
  }
  const ans = t * Math.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
  return x >= 0 ? ans : 2 - ans;
}

/** Normal CDF via erfc — full double precision. */
export const normCdf = (x) => 0.5 * erfc(-x / SQRT2);
export const normPdf = (x) => INV_SQRT_2PI * Math.exp(-0.5 * x * x);

/** Lanczos log-gamma (g=7, n=9), |err| ~ 1e-15 for x>0. */
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = LG[0], t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
/** Digamma: asymptotic series + recurrence (Bernardo 1976). */
export function digamma(x) {
  let r = 0;
  while (x < 6) { r -= 1 / x; x += 1; }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x + f * (-1 / 12 + f * (1 / 120 + f * (-1 / 252 + f * (1 / 240 + f * (-1 / 132)))));
}

/** Stable 2-norm (scaled, no overflow/underflow). */
export function norm2(x, from = 0, to = x.length) {
  let max = 0;
  for (let i = from; i < to; i++) { const a = Math.abs(x[i]); if (a > max) max = a; }
  if (max === 0) return 0;
  let s = 0;
  for (let i = from; i < to; i++) { const r = x[i] / max; s += r * r; }
  return max * Math.sqrt(s);
}

/** Welford online mean/variance — single pass, numerically stable. */
export class Welford {
  constructor() { this.n = 0; this.mean = 0; this.m2 = 0; }
  add(x) {
    this.n++;
    const d = x - this.mean;
    this.mean += d / this.n;
    this.m2 += d * (x - this.mean);
    return this;
  }
  get variance() { return this.n > 1 ? this.m2 / (this.n - 1) : 0; }
  get sampleVariance() { return this.n > 0 ? this.m2 / this.n : 0; }
}

/** Catastrophic-cancellation demo guard: (a-b)^2 vs hypot-style */
export const hypot = (a, b) => Math.sqrt(a * a + b * b);

/** Relative error used by every invariant assertion. */
export function relErr(a, b, epsFloor = 1e-12) {
  const d = Math.abs(a - b);
  const s = Math.max(Math.abs(a), Math.abs(b));
  if (s < epsFloor) return d;               // both ~0 → absolute error
  return d / s;
}

/** Max relative error between two array-likes. */
export function maxRelErr(a, b, epsFloor = 1e-8) {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) m = Math.max(m, relErr(a[i], b[i], epsFloor));
  return m;
}

/** Clip, lerp, argmax, argsort — shared utilities. */
export const clip = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
export const lerp = (a, b, t) => a + (b - a) * t;
export function argmax(x) { let bi = 0; for (let i = 1; i < x.length; i++) if (x[i] > x[bi]) bi = i; return bi; }
export function argsort(x) {
  const idx = Array.from(x.keys());
  idx.sort((i, j) => x[i] - x[j]);
  return idx;
}
/** Mean & (sample) std of an array-like, computed with Welford. */
export function meanStd(x) {
  const w = new Welford();
  for (let i = 0; i < x.length; i++) w.add(x[i]);
  return { mean: w.mean, std: Math.sqrt(w.sampleVariance) };
}
