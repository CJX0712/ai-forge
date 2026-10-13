/**
 * random.js — Deterministic PRNG layer (L0).
 *
 * Why not Math.random(): reproducibility is a hard requirement of every
 * invariant in this project. Every stochastic algorithm must be bit-identical
 * across Node, browsers and runs.
 *
 * Reused world-class results (zero-rewrite principle):
 *  - sfc32 (Doty-Humphrey, PractRand-passing, 2^32 minimum period) — pure
 *    32-bit ops, no BigInt, identical results in every JS engine.
 *  - Marsaglia polar Box–Muller: no trig, no log singularity at 0, one
 *    cached second deviate.
 *  - Marsaglia–Tsang gamma, Vose alias method (O(1) categorical sampling).
 *
 * @author 晨星
 */

/** sfc32 — 128-bit state, passes PractRand, ~2^32 minimum period. */
export class RNG {
  constructor(seed = 0x9e3779b9) {
    // splitmix32 to expand one seed into 4 well-mixed words
    let s = seed >>> 0 || 1;
    const sm = () => {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
      return (z ^ (z >>> 15)) >>> 0;
    };
    this.a = sm(); this.b = sm(); this.c = sm(); this.d = sm();
    for (let i = 0; i < 12; i++) this.next(); // warm up
  }

  /** @returns {number} uniform in [0, 1) */
  next() {
    let a = this.a | 0, b = this.b | 0, c = this.c | 0, d = this.d | 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    this.a = a; this.b = b; this.c = c; this.d = d;
    return (t >>> 0) / 4294967296;
  }

  uniform(x = 0, y = 1) { return x + (y - x) * this.next(); }
  int(n) { return Math.floor(this.next() * n); }
  bernoulli(p) { return this.next() < p; }
  exponential(rate = 1) { return -Math.log(1 - this.next()) / rate; }

  /** standard normal, Marsaglia polar (cached second deviate) */
  normal() {
    if (this._spare !== undefined) { const s = this._spare; this._spare = undefined; return s; }
    let u, v, s;
    do { u = this.next() * 2 - 1; v = this.next() * 2 - 1; s = u * u + v * v; }
    while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    this._spare = v * f;
    return u * f;
  }

  /** Gamma(k, 1) — Marsaglia–Tsang boost */
  gamma(k) {
    if (k < 1) return this.gamma(k + 1) * Math.pow(this.next(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = this.normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = this.next();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  beta(a, b) { const x = this.gamma(a); return x / (x + this.gamma(b)); }

  /** Dirichlet via normalised gammas */
  dirichlet(alpha) {
    const out = new Float64Array(alpha.length);
    let s = 0;
    for (let i = 0; i < alpha.length; i++) { out[i] = this.gamma(alpha[i]); s += out[i]; }
    for (let i = 0; i < out.length; i++) out[i] /= s;
    return out;
  }

  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  fillNormal(a) { for (let i = 0; i < a.length; i++) a[i] = this.normal(); return a; }
}

/** Vose alias table: O(n) build, O(1) sample. */
export class AliasTable {
  constructor(prob) {
    const n = prob.length;
    const sum = prob.reduce((s, x) => s + x, 0);
    const p = prob.map((x) => (x * n) / sum);
    this.n = n;
    this.prob = new Float64Array(n);
    this.alias = new Int32Array(n);
    const small = [], large = [];
    for (let i = 0; i < n; i++) (p[i] < 1 ? small : large).push(i);
    while (small.length && large.length) {
      const l = small.pop(), g = large.pop();
      this.prob[l] = p[l];
      this.alias[l] = g;
      p[g] = p[g] + p[l] - 1;
      (p[g] < 1 ? small : large).push(g);
    }
    while (large.length) this.prob[large.pop()] = 1;
    while (small.length) this.prob[small.pop()] = 1;
  }
  sample(rng) {
    const i = rng.int(this.n);
    return rng.next() < this.prob[i] ? i : this.alias[i];
  }
}

const _registry = new Map();
/** Named deterministic streams: rng("gmm") is bit-identical everywhere. */
export function rng(name = "default", seed = 20261014) {
  if (!_registry.has(name)) {
    let h = 0x811c9dc5;
    for (let i = 0; i < name.length; i++) { h ^= name.charCodeAt(i); h = Math.imul(h, 0x01000193); }
    _registry.set(name, new RNG((h ^ seed) >>> 0));
  }
  return _registry.get(name);
}
