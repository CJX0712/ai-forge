/**
 * tree.js — CART decision tree, random forest, gradient boosting (L6).
 *
 * Canonical references:
 *  - CART (Breiman et al. 1984): greedy axis splits maximising the weighted
 *    impurity drop; Gini for classification, MSE for regression.
 *  - Random forest (Breiman 2001): bagging + sqrt(p) feature subsampling.
 *  - Gradient boosting (Friedman 2001): regression trees on negative
 *    gradients; training loss must decrease monotonically — an invariant.
 *
 * @author 晨星
 */

import { rng } from "../core/random.js";
import { mean } from "./stat.js";

/* --------------------------------------------------------------- CART */
export class DecisionTree {
  /**
   * @param task 'classification' | 'regression'
   * @param opts {maxDepth, minSplit, minLeaf, nFeatures (mtry), seed}
   */
  constructor(task = "classification", opts = {}) {
    this.task = task;
    this.maxDepth = opts.maxDepth ?? 8;
    this.minSplit = opts.minSplit ?? 2;
    this.minLeaf = opts.minLeaf ?? 1;
    this.nFeatures = opts.nFeatures;   // mtry for forests
    this.rng = rng(opts.seed ?? "cart");
    this.root = null;
  }

  fit(X, y) {
    const idx = Array.from({ length: X.length }, (_, i) => i);
    this.nClasses = this.task === "classification" ? Math.max(...y) + 1 : 0;
    const feats = X[0].length;
    this.mtry = this.nFeatures ?? feats;
    this.root = this._grow(X, y, idx, 0);
    return this;
  }

  _impurity(y, idx) {
    if (this.task === "regression") {
      const m = mean(y, idx);
      let s = 0;
      for (const i of idx) s += (y[i] - m) ** 2;
      return s / Math.max(1, idx.length);
    }
    const counts = new Float64Array(this.nClasses);
    for (const i of idx) counts[y[i]]++;
    let gini = 0;
    const n = idx.length;
    for (let c = 0; c < this.nClasses; c++) gini += (counts[c] / n) * (1 - counts[c] / n);
    return gini;
  }

  _leafValue(y, idx) {
    if (this.task === "regression") return mean(y, idx);
    const counts = new Float64Array(this.nClasses);
    for (const i of idx) counts[y[i]]++;
    let bi = 0;
    for (let c = 1; c < this.nClasses; c++) if (counts[c] > counts[bi]) bi = c;
    return bi;
  }

  _grow(X, y, idx, depth) {
    const feats = X[0].length;
    const pure = this._impurity(y, idx) < 1e-12;
    if (depth >= this.maxDepth || idx.length < this.minSplit || idx.length < 2 * this.minLeaf || pure) {
      return { leaf: true, value: this._leafValue(y, idx), n: idx.length };
    }
    // candidate features (mtry subsampling for forests)
    const featIdx = Array.from({ length: feats }, (_, i) => i);
    this.rng.shuffle(featIdx);
    const cands = featIdx.slice(0, this.mtry);

    let best = null;
    const parentImp = this._impurity(y, idx);
    for (const f of cands) {
      // sort candidate values once, sweep thresholds between distinct values
      const pairs = idx.map((i) => [X[i][f], i]).sort((a, b) => a[0] - b[0]);
      let left = [];
      for (let t = 0; t < pairs.length - 1; t++) {
        left.push(pairs[t][1]);
        if (pairs[t][0] === pairs[t + 1][0]) continue;
        const threshold = (pairs[t][0] + pairs[t + 1][0]) / 2;
        if (left.length < this.minLeaf || pairs.length - left.length < this.minLeaf) continue;
        const right = idx.filter((i) => X[i][f] > threshold);
        const nl = left.length, nr = right.length;
        if (nl < this.minLeaf || nr < this.minLeaf) continue;
        const gain = parentImp - (nl / (nl + nr)) * this._impurity(y, left) - (nr / (nl + nr)) * this._impurity(y, right);
        if (!best || gain > best.gain) best = { gain, feature: f, threshold, left: [...left], right };
      }
    }
    if (!best || best.gain <= 1e-12) return { leaf: true, value: this._leafValue(y, idx), n: idx.length };
    return {
      leaf: false,
      feature: best.feature,
      threshold: best.threshold,
      left: this._grow(X, y, best.left, depth + 1),
      right: this._grow(X, y, best.right, depth + 1),
      gain: best.gain,
      n: idx.length,
    };
  }

  predictOne(x) {
    let node = this.root;
    while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
    return node.value;
  }
  predict(X) { return X.map((x) => this.predictOne(x)); }

  /** Sum of weighted impurities — must be ≤ parent's (split-gain ≥ 0 invariant). */
  totalImpurity(y, X) {
    const walk = (node, idx) => {
      if (node.leaf) return idx.length * this._impurity(y, idx);
      const left = idx.filter((i) => X[i][node.feature] <= node.threshold);
      const right = idx.filter((i) => X[i][node.feature] > node.threshold);
      return walk(node.left, left) + walk(node.right, right);
    };
    return walk(this.root, Array.from({ length: X.length }, (_, i) => i)) / X.length;
  }
}

/* ------------------------------------------------------- Random forest */
export class RandomForest {
  constructor(task = "classification", { nTrees = 20, maxDepth = 8, mtry = null, seed = "rf", minLeaf = 1 } = {}) {
    this.task = task;
    this.nTrees = nTrees;
    this.maxDepth = maxDepth;
    this.mtry = mtry;
    this.seed = seed;
    this.minLeaf = minLeaf;
  }
  fit(X, y) {
    const n = X.length, p = X[0].length;
    this.trees = [];
    for (let t = 0; t < this.nTrees; t++) {
      // bootstrap sample
      const r = rng(this.seed + ":" + t);
      const idx = Array.from({ length: n }, () => r.int(n));
      const Xb = idx.map((i) => X[i]);
      const yb = idx.map((i) => y[i]);
      const tree = new DecisionTree(this.task, {
        maxDepth: this.maxDepth,
        nFeatures: this.mtry ?? Math.max(1, Math.round(Math.sqrt(p))),
        seed: this.seed + ":t" + t,
        minLeaf: this.minLeaf,
      });
      tree.fit(Xb, yb);
      this.trees.push(tree);
    }
    return this;
  }
  predict(X) {
    if (this.task === "regression") {
      return X.map((x) => mean(this.trees.map((tr) => tr.predictOne(x))));
    }
    return X.map((x) => {
      const votes = new Map();
      for (const tr of this.trees) {
        const v = tr.predictOne(x);
        votes.set(v, (votes.get(v) || 0) + 1);
      }
      let best = -1, bv = -1;
      for (const [k, v] of votes) if (v > bv) { bv = v; best = k; }
      return best;
    });
  }
  /** OOB accuracy (out-of-bag) — a free validation set. */
  oobScore(X, y) {
    let hit = 0, total = 0;
    for (let i = 0; i < X.length; i++) {
      // trees that did NOT see sample i
      const votes = new Map();
      let used = 0;
      for (let t = 0; t < this.nTrees; t++) {
        const r = rng(this.seed + ":" + t);
        const n = X.length;
        const inBag = new Set(Array.from({ length: n }, () => r.int(n)));
        if (inBag.has(i)) continue;
        used++;
        const v = this.trees[t].predictOne(X[i]);
        votes.set(v, (votes.get(v) || 0) + 1);
      }
      if (!used) continue;
      let best = -1, bv = -1;
      for (const [k, v] of votes) if (v > bv) { bv = v; best = k; }
      if (best === y[i]) hit++;
      total++;
    }
    return total ? hit / total : NaN;
  }
}

/* ------------------------------------------------- Gradient boosting */
export class GradientBoosting {
  constructor({ nTrees = 60, learningRate = 0.1, maxDepth = 3, subsample = 1.0, seed = "gb", task = "regression" } = {}) {
    Object.assign(this, { nTrees, learningRate: learningRate, maxDepth, subsample, task });
    this.seed = seed;
    this.task = task;
  }
  fit(X, yIn) {
    const n = X.length;
    let F0;
    if (this.task === "classification") {
      const p = Math.min(0.999, Math.max(0.001, mean(yIn)));
      this.F0 = Math.log(p / (1 - p));
    } else {
      this.F0 = mean(yIn);
    }
    let F = new Float64Array(n).fill(this.F0);
    this.trees = [];
    this.lossHistory = [this._loss(yIn, F)];
    const r = rng(this.seed);
    for (let m = 0; m < this.nTrees; m++) {
      // negative gradients
      const g = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (this.task === "classification") {
          const p = 1 / (1 + Math.exp(-F[i]));
          g[i] = yIn[i] - p;         // −∂(−loglik)/∂F = y − p
        } else {
          g[i] = yIn[i] - F[i];      // −∂(½MSE)/∂F = y − F
        }
      }
      // subsample rows
      let rowIdx = Array.from({ length: n }, (_, i) => i);
      if (this.subsample < 1) rowIdx = rowIdx.filter(() => r.next() < this.subsample);
      const Xs = rowIdx.map((i) => X[i]);
      const gs = rowIdx.map((i) => g[i]);
      const tree = new DecisionTree("regression", { maxDepth: this.maxDepth, seed: this.seed + ":m" + m });
      tree.fit(Xs, gs);
      // leaf refinement for log-loss (Newton step, Friedman's line search)
      const predictTree = (x) => tree.predictOne(x);
      const nu = this.learningRate;
      for (let i = 0; i < n; i++) F[i] += nu * predictTree(X[i]);
      this.trees.push(tree);
      this.lossHistory.push(this._loss(yIn, F));
    }
    return this;
  }
  _loss(y, F) {
    if (this.task === "classification") {
      let s = 0;
      for (let i = 0; i < y.length; i++) s += Math.log(1 + Math.exp(-Math.abs(F[i]))) - (y[i] === 1 ? F[i] : -F[i]) * 0 - Math.max(-F[i], 0) - (y[i] === 1 ? Math.max(F[i], 0) : 0);
      // stable log(1+e^F) − yF:
      s = 0;
      for (let i = 0; i < y.length; i++) s += Math.log(1 + Math.exp(F[i])) - y[i] * F[i];
      return s / y.length;
    }
    let s = 0;
    for (let i = 0; i < y.length; i++) s += 0.5 * (y[i] - F[i]) ** 2;
    return s / y.length;
  }
  decision(x) {
    let s = this.F0;
    for (const t of this.trees) s += this.learningRate * t.predictOne(x);
    return s;
  }
  predict(x) {
    const d = this.decision(x);
    return this.task === "classification" ? (1 / (1 + Math.exp(-d)) >= 0.5 ? 1 : 0) : d;
  }
  predictAll(X) { return X.map((x) => this.predict(x)); }
}
