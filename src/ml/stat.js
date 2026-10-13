/**
 * stat.js — Small shared statistics helpers for the ML layer.
 * @author 晨星
 */

import { rng } from "../core/random.js";

export function mean(arr, idx = null) {
  if (idx) {
    let s = 0;
    for (const i of idx) s += arr[i];
    return s / idx.length;
  }
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

export function variance(arr, idx = null) {
  const m = mean(arr, idx);
  const n = idx ? idx.length : arr.length;
  let s = 0;
  if (idx) { for (const i of idx) s += (arr[i] - m) ** 2; }
  else { for (let i = 0; i < arr.length; i++) s += (arr[i] - m) ** 2; }
  return s / Math.max(1, n - 1);
}

/** Pearson correlation. */
export function correlation(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy);
}

/** Accuracy, confusion counts, precision/recall/F1 for binary/multiclass. */
export function metrics(yTrue, yPred, nClasses = 2) {
  let acc = 0;
  const confusion = Array.from({ length: nClasses }, () => new Float64Array(nClasses));
  for (let i = 0; i < yTrue.length; i++) {
    confusion[yTrue[i]][yPred[i]]++;
    if (yTrue[i] === yPred[i]) acc++;
  }
  acc /= yTrue.length;
  const perClass = [];
  for (let c = 0; c < nClasses; c++) {
    const tp = confusion[c][c];
    const fp = confusion.reduce((s, row, r) => s + (r === c ? 0 : row[c]), 0);
    const fn = confusion[c].reduce((s, v, cc) => s + (cc === c ? 0 : v), 0);
    const prec = tp + fp ? tp / (tp + fp) : 0;
    const rec = tp + fn ? tp / (tp + fn) : 0;
    perClass.push({ class: c, precision: prec, recall: rec, f1: prec + rec ? 2 * prec * rec / (prec + rec) : 0, support: tp + fn });
  }
  return { accuracy: acc, confusion, perClass };
}

/** R² coefficient of determination. */
export function r2(yTrue, yPred) {
  const m = mean(yTrue);
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < yTrue.length; i++) { ssRes += (yTrue[i] - yPred[i]) ** 2; ssTot += (yTrue[i] - m) ** 2; }
  return 1 - ssRes / ssTot;
}

/** Mean squared error. */
export function mse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
  return s / yTrue.length;
}

/** Deterministic train/test split. */
export function trainTestSplit(X, y, { testFrac = 0.25, seed = "split" } = {}) {
  const r = rng(seed);
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  r.shuffle(idx);
  const nTest = Math.max(1, Math.round(n * testFrac));
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);
  return {
    Xtr: trainIdx.map((i) => X[i]), ytr: trainIdx.map((i) => y[i]),
    Xte: testIdx.map((i) => X[i]), yte: testIdx.map((i) => y[i]),
  };
}
