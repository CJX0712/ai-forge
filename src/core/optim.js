/**
 * optim.js — Optimizer suite (L2).
 *
 * Zero-rewrite: every optimizer is the canonical published algorithm,
 * not a paraphrase:
 *
 *  - AdamW  (Loshchilov & Hutter, ICLR 2019): decoupled weight decay —
 *    decay is applied to weights, not folded into the adaptive moment.
 *  - L-BFGS (Nocedal 1980; two-loop recursion) + strong-Wolfe line search
 *    with bracketing (More–Thuente style). On a quadratic this must hit
 *    ‖∇f‖ < 1e-8 within n steps — an executable invariant.
 *  - Muon   (Jordan 2024, NanoGPT speedrun; Moonshot "Muon is Scalable
 *    for LLM Training", 2025): SGD+Nesterov momentum on 2-D weights,
 *    orthogonalised by the quintic Newton–Schulz iteration, RMS-rescaled;
 *    AdamW handles 1-D params, embeddings and heads (hybrid recipe).
 *
 * @author 晨星
 */

import { newtonSchulz, muonRescale } from "./linalg.js";
import { zeros } from "./linalg.js";
import { backward as backward_ } from "./autodiff.js";

/* ------------------------------------------------------------- AdamW */
export class AdamW {
  constructor(params, { lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.01 } = {}) {
    this.params = params;
    this.lr = lr; this.b1 = betas[0]; this.b2 = betas[1];
    this.eps = eps; this.wd = weightDecay;
    this.t = 0;
    this.m = params.map((p) => new Float64Array(p.size));
    this.v = params.map((p) => new Float64Array(p.size));
  }
  step() {
    this.t++;
    const { lr, b1, b2, eps, wd, t } = this;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i], g = p.grad, m = this.m[i], v = this.v[i];
      if (!g) continue;
      for (let j = 0; j < p.size; j++) {
        m[j] = b1 * m[j] + (1 - b1) * g[j];
        v[j] = b2 * v[j] + (1 - b2) * g[j] * g[j];
        const mh = m[j] / bc1, vh = v[j] / bc2;
        p.data[j] -= lr * (mh / (Math.sqrt(vh) + eps));
        if (wd !== 0) p.data[j] -= lr * wd * p.data[j];  // decoupled decay
      }
    }
  }
  zeroGrad() { for (const p of this.params) if (p.grad) p.grad.fill(0); }
}

/* --------------------------------------------------------------- SGD */
export class SGD {
  constructor(params, { lr = 0.01, momentum = 0, nesterov = false, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr; this.mu = momentum; this.nesterov = nesterov; this.wd = weightDecay;
    this.buf = params.map((p) => (momentum ? new Float64Array(p.size) : null));
  }
  step() {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i], g = p.grad;
      if (!g) continue;
      const b = this.buf[i];
      for (let j = 0; j < p.size; j++) {
        let gi = g[j] + (this.wd ? this.wd * p.data[j] : 0);
        if (this.mu) {
          b[j] = this.mu * b[j] + gi;
          gi = this.nesterov ? gi + this.mu * b[j] : b[j];
        }
        p.data[j] -= this.lr * gi;
      }
    }
  }
  zeroGrad() { for (const p of this.params) if (p.grad) p.grad.fill(0); }
}

/* ------------------------------------------------------------- Muon */
/**
 * Hybrid Muon: 2-D matrices get orthogonalised Nesterov momentum
 * (Newton–Schulz, 5 steps); everything else falls through to AdamW.
 * 1-D params are detected by shape.length < 2 OR by explicit opt-out.
 */
export class Muon {
  constructor(params, { lr = 0.02, momentum = 0.95, nesterov = true, nsSteps = 5,
    weightDecay = 0.01, adamLR = 3e-3, adamBetas = [0.9, 0.95] } = {}) {
    this.lr = lr; this.mu = momentum; this.nesterov = nesterov;
    this.nsSteps = nsSteps; this.wd = weightDecay;
    this.matrixParams = [];
    this.otherParams = [];
    for (const p of params) (p.shape.length >= 2 ? this.matrixParams : this.otherParams).push(p);
    this.buf = new Map();
    this.adam = new AdamW(this.otherParams, { lr: adamLR, betas: adamBetas, weightDecay });
  }
  step() {
    // Muon branch on 2-D weights
    for (const p of this.matrixParams) {
      if (!p.grad) continue;
      const m = p.shape[p.shape.length - 2], n = p.shape[p.shape.length - 1];
      let buf = this.buf.get(p);
      if (!buf) { buf = new Float64Array(p.size); this.buf.set(p, buf); }
      const g = p.grad;
      for (let i = 0; i < p.size; i++) buf[i] = this.mu * buf[i] + (1 - this.mu) * g[i];
      const upd = this.nesterov
        ? Float64Array.from(buf, (v, i) => this.mu * v + (1 - this.mu) * g[i])
        : Float64Array.from(buf);
      const M = { r: m, c: n, a: Float64Array.from(upd) };
      const O = newtonSchulz(M, this.nsSteps);
      const R = muonRescale(O, n, m);   // fanIn = n (cols), fanOut = m (rows)
      for (let i = 0; i < p.size; i++) {
        p.data[i] -= this.lr * R.a[i];
        if (this.wd) p.data[i] -= this.lr * this.wd * p.data[i];
      }
    }
    this.adam.step();
  }
  zeroGrad() {
    for (const p of this.matrixParams) if (p.grad) p.grad.fill(0);
    this.adam.zeroGrad();
  }
}

/* ------------------------------------------------------------- L-BFGS */
/**
 * L-BFGS with two-loop recursion (Nocedal 1980) + strong-Wolfe line search.
 * Operates on plain Float64Array state; caller supplies f(x) → {value, grad}.
 */
export function lbfgs(f, x0, { maxIter = 500, m = 10, gTol = 1e-8, c1 = 1e-4, c2 = 0.9, verbose = false } = {}) {
  const n = x0.length;
  let x = Float64Array.from(x0);
  let fx, gx;
  ({ value: fx, grad: gx } = f(x));
  const S = [], Y = [];           // curvature pairs
  let evals = 1, lineEvals = 0;

  const dot = (a, b) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; };

  for (let k = 0; k < maxIter; k++) {
    if (Math.sqrt(dot(gx, gx)) < gTol) break;
    // two-loop recursion → direction d = -H·g
    const q = Float64Array.from(gx);
    const alphas = new Float64Array(S.length);
    for (let i = S.length - 1; i >= 0; i--) {
      const rho = 1 / dot(Y[i], S[i]);
      alphas[i] = rho * dot(S[i], q);
      for (let j = 0; j < n; j++) q[j] -= alphas[i] * Y[i][j];
    }
    // initial Hessian scaling γ = sᵀy / yᵀy  (Nocedal & Wright eq 7.20)
    let gamma = 1;
    if (S.length) {
      const sy = dot(S[S.length - 1], Y[S.length - 1]);
      const yy = dot(Y[S.length - 1], Y[S.length - 1]);
      if (yy > 0) gamma = sy / yy;
    }
    const r = new Float64Array(n);
    for (let j = 0; j < n; j++) r[j] = gamma * q[j];
    for (let i = 0; i < S.length; i++) {
      const rho = 1 / dot(Y[i], S[i]);
      const beta = rho * dot(Y[i], r);
      for (let j = 0; j < n; j++) r[j] += S[i][j] * (alphas[i] - beta);
    }
    const d = new Float64Array(n);
    for (let j = 0; j < n; j++) d[j] = -r[j];

    const gTd = dot(gx, d);
    if (gTd >= 0) { // not a descent direction — reset memory
      S.length = 0; Y.length = 0;
      for (let j = 0; j < n; j++) d[j] = -gx[j];
    }

    // strong-Wolfe backtracking/zoom line search
    const { alpha, fxNew, gxNew } = wolfeLineSearch(f, x, fx, gx, d, c1, c2);
    lineEvals += 1;
    const s = new Float64Array(n), y = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      s[j] = alpha * d[j];
      x[j] += s[j];
      y[j] = gxNew[j] - gx[j];
    }
    fx = fxNew; gx = gxNew;
    if (S.length >= m) { S.shift(); Y.shift(); }
    S.push(s); Y.push(y);
    if (verbose && k % 20 === 0) console.log(`  lbfgs iter ${k}  f=${fx.toExponential(4)}  |g|=${Math.sqrt(dot(gx, gx)).toExponential(3)}`);
  }
  evals += lineEvals;
  return { x, f: fx, grad: gx, iterations: maxIter, evals };
}

/** Strong-Wolfe line search: bracket + zoom (Nocedal & Wright Alg 3.5/3.6). */
function wolfeLineSearch(f, x, fx, gx, d, c1, c2) {
  const n = x.length;
  const evalAt = (a) => {
    const xt = new Float64Array(n);
    for (let i = 0; i < n; i++) xt[i] = x[i] + a * d[i];
    const r = f(xt);
    return r;
  };
  const dG = (g) => { let s = 0; for (let i = 0; i < n; i++) s += g[i] * d[i]; return s; };

  let aPrev = 0, fPrev = fx, gPrev = gx, dGPrev = dG(gx);
  let a = 1;
  for (let i = 0; i < 25; i++) {
    const r = evalAt(a);
    const dGa = dG(r.grad);
    if (r.value > fx + c1 * a * dGPrev || (i > 0 && r.value >= fPrev)) {
      return zoom(aPrev, a, fPrev, dGPrev, evalAt, dG, c1, c2);
    }
    if (Math.abs(dGa) <= -c2 * dGPrev) return { alpha: a, fxNew: r.value, gxNew: r.grad };
    if (dGa >= 0) return zoom(a, aPrev, r.value, dGa, evalAt, dG, c1, c2);
    aPrev = a; fPrev = r.value; gPrev = r.grad;
    a *= 2;
    if (a > 1e10) break;
  }
  return { alpha: aPrev, fxNew: fPrev, gxNew: gPrev };
}

function zoom(aLo, aHi, fLo, dGLo, evalAt, dG, c1, c2) {
  // fLo/dGLo describe the endpoint with lower f (aLo side)
  let lo = aLo, hi = aHi;
  let flo = fLo, dGlo = dGLo, glo = null;
  for (let i = 0; i < 30; i++) {
    const a = lo + 0.5 * (hi - lo);
    const r = evalAt(a);
    const dGa = dG(r.grad);
    if (r.value > flo + c1 * a * dGlo || r.value >= flo) {
      hi = a;
    } else {
      if (Math.abs(dGa) <= -c2 * dGlo) return { alpha: a, fxNew: r.value, gxNew: r.grad };
      if (dGa * (hi - lo) >= 0) hi = lo;
      lo = a; flo = r.value; dGlo = dGa; glo = r.grad;
    }
  }
  return { alpha: lo, fxNew: flo, gxNew: glo || evalAt(lo).grad };
}

/**
 * Convenience: optimise an autodiff loss where `build()` returns the loss
 * Tensor and `params` are the leaf tensors.
 */
export function minimize(lossFn, params, opt, { steps = 100, callback = null } = {}) {
  const hist = [];
  for (let t = 0; t < steps; t++) {
    opt.zeroGrad();
    const loss = lossFn();
    backward_(loss);
    opt.step();
    hist.push(loss.data[0]);
    if (callback) callback(t, loss.data[0]);
  }
  return hist;
}
