/**
 * autodiff.js — Reverse-mode automatic differentiation (L1).
 *
 * Design decision (and why): we deliberately ship a *tape-based* reverse-mode
 * engine rather than a graph-compiled one, because the project's hard
 * requirement is *verifiability*: a tape makes gradient checking against
 * central differences trivial and exact, which is the gold standard used by
 * every serious framework (PyTorch `gradcheck`, JAX `check_grads`).
 *
 * Reused results:
 *  - Reverse-mode AD (Linnainmaa 1970 / Rumelhart–Hinton–Williams 1986)
 *  - Topological ordering via DFS + reverse traversal
 *  - Central-difference gradient check: (f(x+h) − f(x−h)) / 2h, O(h²) error,
 *    optimal h ≈ eps^(1/3)·max(1,|x|)  (Numerical Recipes / PyTorch convention)
 *
 * @author 晨星
 */

import { softmax, logsumexp } from "./numeric.js";

let TAPE = null;               // active recording tape
let NODE_ID = 0;

export function noGrad(fn) { const p = TAPE; TAPE = null; try { return fn(); } finally { TAPE = p; } }
export const isGradEnabled = () => TAPE !== null;

export class Tensor {
  constructor(data, shape, requiresGrad = false) {
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
    this.shape = shape || [this.data.length];
    this.requiresGrad = requiresGrad && TAPE !== null;
    this.grad = this.requiresGrad ? new Float64Array(this.data.length) : null;
    this._inputs = [];
    this._backward = null;
    this.id = NODE_ID++;
    this.name = "";
  }
  get size() { return this.data.length; }
  /** row count for 2-D tensors */
  get r() { return this.shape.length >= 2 ? this.shape[this.shape.length - 2] : 1; }
  get c() { return this.shape[this.shape.length - 1]; }

  zeroGrad() { if (this.grad) this.grad.fill(0); }
  detach() { return new Tensor(this.data, this.shape, false); }
  clone() { return new Tensor(Float64Array.from(this.data), this.shape, this.requiresGrad); }
  toString() {
    return `Tensor(${this.shape.join("x")}${this.requiresGrad ? ", grad" : ""})`;
  }
}

export function tensor(data, shape, requiresGrad = true) {
  return new Tensor(data, shape, requiresGrad);
}
/**
 * Leaf parameter — *unconditionally* collects gradients, tape or not.
 * (A parameter created outside a tape must still receive gradients when a
 * later tape pass touches it; this is the PyTorch `requires_grad=True`
 * semantics and the bug class the L2 invariant suite exists to catch.)
 */
export function param(data, shape) {
  const t = new Tensor(data, shape, false);
  t.requiresGrad = true;
  t.grad = new Float64Array(t.size);
  return t;
}

function record(out, inputs, backward) {
  if (TAPE === null) return out;
  out.requiresGrad = inputs.some((t) => t.requiresGrad && t.grad);
  if (out.requiresGrad) {
    out.grad = new Float64Array(out.data.length);
    out._inputs = inputs.filter((t) => t.requiresGrad && t.grad);
    out._backward = backward;
    TAPE.push(out);
  }
  return out;
}

function broadcastIndex(shape, idx, out) {
  // map flat index of the *result* back to flat index of an input
  const nd = shape.length;
  let rem = idx;
  for (let d = nd - 1; d >= 0; d--) out[d] = rem % shape[d], rem = Math.floor(rem / shape[d]);
}

/* ---------------------------------------------------------------- ops */
export function add(a, b) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  const bs = b.shape;
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] + b.data[i % b.size];
  return record(out, [a, b], () => {
    if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i];
    if (b.grad) for (let i = 0; i < a.size; i++) b.grad[i % b.size] += out.grad[i];
  });
}
export function sub(a, b) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] - b.data[i % b.size];
  return record(out, [a, b], () => {
    if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i];
    if (b.grad) for (let i = 0; i < b.size; i++) b.grad[i] += -out.grad[i % b.size];
  });
}
/** elementwise multiply (with broadcasting of b over a) */
export function mul(a, b) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * b.data[i % b.size];
  return record(out, [a, b], () => {
    if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * b.data[i % b.size];
    if (b.grad) for (let i = 0; i < a.size; i++) b.grad[i % b.size] += out.grad[i] * a.data[i];
  });
}
export function scale(a, s) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * s;
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * s; });
}
export function sum(a) {
  const out = new Tensor(new Float64Array([0]), [1]);
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  out.data[0] = s;
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[0]; });
}
export function mean(a) {
  const out = new Tensor(new Float64Array([0]), [1]);
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  out.data[0] = s / a.size;
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[0] / a.size; });
}
export function exp(a) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = Math.exp(a.data[i]);
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * out.data[i]; });
}
export function log(a) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = Math.log(a.data[i]);
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] / a.data[i]; });
}
export function tanh(a) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = Math.tanh(a.data[i]);
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * (1 - out.data[i] * out.data[i]); });
}
export function relu(a) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] > 0 ? a.data[i] : 0;
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += a.data[i] > 0 ? out.grad[i] : 0; });
}
export function square(a) {
  const out = new Tensor(new Float64Array(a.size), a.shape.slice());
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * a.data[i];
  return record(out, [a], () => { if (a.grad) for (let i = 0; i < a.size; i++) a.grad[i] += out.grad[i] * 2 * a.data[i]; });
}

/** Matrix multiply: [m,k] · [k,n] → [m,n] */
export function matmul(a, b) {
  const m = a.r, k = a.c, n = b.c;
  const out = new Tensor(new Float64Array(m * n), [m, n]);
  const A = a.data, B = b.data, C = out.data;
  for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) {
    const av = A[i * k + p]; if (av === 0) continue;
    for (let j = 0; j < n; j++) C[i * n + j] += av * B[p * n + j];
  }
  return record(out, [a, b], () => {
    if (a.grad) {
      for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += out.grad[i * n + j] * B[p * n + j];
        a.grad[i * k + p] += s;
      }
    }
    if (b.grad) {
      for (let p = 0; p < k; p++) for (let j = 0; j < n; j++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += A[i * k + p] * out.grad[i * n + j];
        b.grad[p * n + j] += s;
      }
    }
  });
}

/** Row-wise softmax over a [T, V] tensor. */
export function softmaxRows(a) {
  const [T, V] = [a.r, a.c];
  const out = new Tensor(new Float64Array(T * V), [T, V]);
  const tmp = new Float64Array(V);
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) tmp[v] = a.data[t * V + v];
    softmax(tmp, tmp);
    for (let v = 0; v < V; v++) out.data[t * V + v] = tmp[v];
  }
  return record(out, [a], () => {
    if (!a.grad) return;
    for (let t = 0; t < T; t++) {
      let dot = 0;
      for (let v = 0; v < V; v++) dot += out.grad[t * V + v] * out.data[t * V + v];
      for (let v = 0; v < V; v++) a.grad[t * V + v] += out.data[t * V + v] * (out.grad[t * V + v] - dot);
    }
  });
}

/** Mean cross-entropy: logits [T,V], targets Int array of length T. */
export function crossEntropy(logits, targets) {
  const [T, V] = [logits.r, logits.c];
  const out = new Tensor(new Float64Array([0]), [1]);
  const row = new Float64Array(V);
  let loss = 0;
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) row[v] = logits.data[t * V + v];
    loss += logsumexp(row) - row[targets[t]];
  }
  out.data[0] = loss / T;
  return record(out, [logits], () => {
    if (!logits.grad) return;
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < V; v++) row[v] = logits.data[t * V + v];
      const lse = logsumexp(row);
      for (let v = 0; v < V; v++) {
        const p = Math.exp(row[v] - lse);
        logits.grad[t * V + v] += (p - (v === targets[t] ? 1 : 0)) / T;
      }
    }
  });
}

/* ------------------------------------------------------------ engine */
/** Run reverse-mode AD from `root`. Seeds root.grad = 1. */
export function backward(root) {
  if (!root.grad) root.grad = new Float64Array(root.size);
  root.grad[0] = 1;
  const topo = [];
  const seen = new Set();
  (function dfs(t) {
    if (seen.has(t.id)) return;
    seen.add(t.id);
    for (const p of t._inputs) dfs(p);
    topo.push(t);
  })(root);
  for (let i = topo.length - 1; i >= 0; i--) {
    const t = topo[i];
    if (t._backward) t._backward();
  }
}

/** Fresh tape; run fn() inside it. */
export function withTape(fn) {
  const prev = TAPE;
  TAPE = [];
  try { return fn(); } finally { TAPE = prev; }
}

/**
 * Central-difference gradient check — the project's #1 invariant.
 * Returns max relative error between analytic and numeric gradients.
 * Optimal step: h = eps^(1/3) · max(1, |x|).
 */
export function gradCheck(f, params, opts = {}) {
  const { h = 1e-5, tol = 1e-6, verbose = false } = opts;
  let maxRel = 0;
  for (const p of params) {
    if (!p.grad) continue;
    const analytic = Float64Array.from(p.grad);
    for (let i = 0; i < p.size; i++) {
      const orig = p.data[i];
      const step = h * Math.max(1, Math.abs(orig));
      p.data[i] = orig + step; const fp = f();
      p.data[i] = orig - step; const fm = f();
      p.data[i] = orig;
      const numeric = (fp - fm) / (2 * step);
      const denom = Math.max(1e-8, Math.abs(numeric), Math.abs(analytic[i]));
      const rel = Math.abs(numeric - analytic[i]) / denom;
      if (verbose && rel > tol) console.warn(`  grad mismatch @${i}: ana=${analytic[i]} num=${numeric}`);
      maxRel = Math.max(maxRel, rel);
    }
  }
  return maxRel;
}
