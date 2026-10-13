/**
 * instance.js — k-NN with KD-tree, Naive Bayes (L6).
 *
 * Canonical references:
 *  - KD-tree (Bentley 1975): O(log n) nearest-neighbour in low dimensions.
 *    Invariant: tree search must return exactly the brute-force result.
 *  - Naive Bayes (McCallum & Nigam 1998): Gaussian NB with log-space
 *    joint + Laplace-style variance smoothing; posteriors sum to 1.
 *
 * @author 晨星
 */

import { logsumexp } from "../core/numeric.js";

/* ------------------------------------------------------------ KD-tree */
export class KDTree {
  constructor(points) {
    this.points = points;
    const d = points[0]?.length ?? 0;
    this.dim = d;
    const ids = Array.from({ length: points.length }, (_, i) => i);
    this.root = this._build(ids, 0);
  }
  _build(ids, depth) {
    if (!ids.length) return null;
    const axis = depth % this.dim;
    ids.sort((a, b) => this.points[a][axis] - this.points[b][axis]);
    const mid = ids.length >> 1;
    return {
      idx: ids[mid],
      axis,
      left: this._build(ids.slice(0, mid), depth + 1),
      right: this._build(ids.slice(mid + 1), depth + 1),
    };
  }
  /** k nearest neighbours; returns indices sorted by distance. */
  knearest(q, k = 1) {
    const heap = [];  // max-heap of [dist2, idx]
    const push = (d2, i) => {
      heap.push([d2, i]);
      let c = heap.length - 1;
      while (c > 0) {
        const p = (c - 1) >> 1;
        if (heap[p][0] >= heap[c][0]) break;
        [heap[p], heap[c]] = [heap[c], heap[p]];
        c = p;
      }
    };
    const popMax = () => {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        let p = 0;
        for (;;) {
          const l = 2 * p + 1, rt = 2 * p + 2;
          let m = p;
          if (l < heap.length && heap[l][0] > heap[m][0]) m = l;
          if (rt < heap.length && heap[rt][0] > heap[m][0]) m = rt;
          if (m === p) break;
          [heap[p], heap[m]] = [heap[m], heap[p]];
          p = m;
        }
      }
      return top;
    };
    const d2 = (i) => {
      let s = 0;
      const p = this.points[i];
      for (let j = 0; j < this.dim; j++) { const t = p[j] - q[j]; s += t * t; }
      return s;
    };
    const visit = (node) => {
      if (!node) return;
      const dist = d2(node.idx);
      if (heap.length < k) push(dist, node.idx);
      else if (dist < heap[0][0]) { popMax(); push(dist, node.idx); }
      const axisDiff = q[node.axis] - this.points[node.idx][node.axis];
      const near = axisDiff < 0 ? node.left : node.right;
      const far = axisDiff < 0 ? node.right : node.left;
      visit(near);
      if (heap.length < k || axisDiff * axisDiff < heap[0][0]) visit(far);
    };
    visit(this.root);
    return heap.map(([, i]) => i).sort((a, b) => d2(a) - d2(b));
  }
}

/** Brute-force reference (also the invariant oracle). */
export function knearestBrute(points, q, k = 1) {
  const idx = points.map((_, i) => i);
  const d2 = (i) => {
    let s = 0;
    const p = points[i];
    for (let j = 0; j < q.length; j++) { const t = p[j] - q[j]; s += t * t; }
    return s;
  };
  idx.sort((a, b) => d2(a) - d2(b));
  return idx.slice(0, k);
}

/** k-NN classifier. */
export class KNNClassifier {
  constructor({ k = 3 } = {}) { this.k = k; }
  fit(X, y) { this.X = X; this.y = y; this.tree = new KDTree(X); return this; }
  predict(x) {
    const nn = this.tree.knearest(x, this.k);
    const votes = new Map();
    for (const i of nn) votes.set(this.y[i], (votes.get(this.y[i]) || 0) + 1);
    let best = null, bv = -1;
    for (const [c, v] of votes) if (v > bv || (v === bv && c < best)) { bv = v; best = c; }
    return best;
  }
  predictAll(X) { return X.map((x) => this.predict(x)); }
}

/* --------------------------------------------------------- Naive Bayes */
export class GaussianNB {
  fit(X, y) {
    const n = X.length, d = X[0].length;
    const classes = [...new Set(y)].sort((a, b) => a - b);
    this.classes = classes;
    this.prior = new Float64Array(classes.length);
    this.mean = [];
    this.var = [];
    for (let ci = 0; ci < classes.length; ci++) {
      const c = classes[ci];
      const idx = y.map((v, i) => (v === c ? i : -1)).filter((i) => i >= 0);
      this.prior[ci] = idx.length / n;
      const m = new Float64Array(d), v = new Float64Array(d);
      for (const i of idx) for (let j = 0; j < d; j++) m[j] += X[i][j];
      for (let j = 0; j < d; j++) m[j] /= idx.length;
      for (const i of idx) for (let j = 0; j < d; j++) v[j] += (X[i][j] - m[j]) ** 2;
      for (let j = 0; j < d; j++) v[j] = v[j] / idx.length + 1e-9; // variance smoothing
      this.mean.push(m); this.var.push(v);
    }
    return this;
  }
  /** log joint P(c) + Σ log P(xⱼ|c) — everything in log space. */
  logJoint(x) {
    const out = new Float64Array(this.classes.length);
    for (let ci = 0; ci < this.classes.length; ci++) {
      let s = Math.log(this.prior[ci]);
      for (let j = 0; j < x.length; j++) {
        const diff = x[j] - this.mean[ci][j];
        s += -0.5 * (diff * diff / this.var[ci][j] + Math.log(2 * Math.PI * this.var[ci][j]));
      }
      out[ci] = s;
    }
    return out;
  }
  predictProba(x) {
    const lj = this.logJoint(x);
    const lse = logsumexp(lj);
    return Float64Array.from(lj, (v) => Math.exp(v - lse));
  }
  predict(x) {
    const lj = this.logJoint(x);
    let bi = 0;
    for (let i = 1; i < lj.length; i++) if (lj[i] > lj[bi]) bi = i;
    return this.classes[bi];
  }
  predictAll(X) { return X.map((x) => this.predict(x)); }
}
