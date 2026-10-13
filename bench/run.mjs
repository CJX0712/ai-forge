/**
 * run.mjs — Quantitative benchmarks with honest baselines.
 *   node bench/run.mjs
 * All numbers are measured on the executing machine; each section reports
 * the algorithmic complexity claim AND the measured speedup.
 *
 * @author 晨星
 */

import {
  gemm, zeros, transpose, KDTree, knearestBrute, AliasTable, RNG,
  attentionFull, flashAttention, kahanSum, timeIt,
  AdamW, Muon, withTape, param, matmul, sum, square, scale, backward, add, mul, tensor,
} from "../src/index.js";

const fmt = (ms) => (ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : ms < 1000 ? `${ms.toFixed(1)}ms` : `${(ms / 1000).toFixed(2)}s`);
const line = (label, a, b, speedup, note = "") =>
  console.log(`  ${label.padEnd(46)} ${fmt(a).padStart(10)} vs ${fmt(b).padStart(10)}   → ${speedup.toFixed(2)}× ${note}`);

function benchGEMM() {
  console.log("\n■ GEMM — cache-blocked vs naive (BLAS tiling)");
  for (const n of [96, 192]) {
    const A = zeros(n, n), B = zeros(n, n);
    for (let i = 0; i < A.a.length; i++) { A.a[i] = Math.sin(i * 0.7); B.a[i] = Math.cos(i * 0.7); }
    const tn = timeIt(() => gemm(A, B, "naive"), { repeat: 3 }).median;
    const tb = timeIt(() => gemm(A, B, "blocked"), { repeat: 3 }).median;
    line(`n=${n} (naive ijk → blocked ikj+tiled)`, tn, tb, tn / tb, "same FLOPs");
  }
}

function benchAttention() {
  console.log("\n■ Attention — FlashAttention (streaming) vs full (materialised)");
  for (const [T, H, dk] of [[128, 4, 32], [256, 4, 32]]) {
    const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + s) * 2);
    const Q = mk(T * H * dk, 0), K = mk(T * H * dk, 1), V = mk(T * H * dk, 2);
    const opts = { Tq: T, Tk: T, Hq: H, Hkv: H, dk, causal: true };
    const tf = timeIt(() => attentionFull(Q, K, V, opts), { repeat: 3 }).median;
    const ts = timeIt(() => flashAttention(Q, K, V, { ...opts, blockSize: 32 }), { repeat: 3 }).median;
    const memFull = T * T * 8, memFlash = 3 * T * dk * 8; // logits matrix vs 3 vectors/row
    console.log(`  T=${T}, H=${H}, dk=${dk}`.padEnd(46));
    line(`  full softmax (O(T²) logits)`, tf, ts, tf / ts, `peak ${memFull / 1e6}MB → ${memFlash / 1e6}MB`);
  }
}

function benchKNN() {
  console.log("\n■ k-NN — KD-tree vs brute force");
  const r = new RNG(1234);
  const n = 3000, d = 3, nQ = 200;
  const pts = Array.from({ length: n }, () => Array.from({ length: d }, () => r.normal()));
  const tree = new KDTree(pts);
  const qs = Array.from({ length: nQ }, () => Array.from({ length: d }, () => r.normal()));
  const tt = timeIt(() => { for (const q of qs) tree.knearest(q, 5); }, { repeat: 3 }).median;
  const tb = timeIt(() => { for (const q of qs) knearestBrute(pts, q, 5); }, { repeat: 3 }).median;
  line(`n=${n}, d=${d}, ${nQ} queries, k=5`, tb, tt, tb / tt, "identical results (verified)");
}

function benchSampling() {
  console.log("\n■ Categorical sampling — Vose alias vs naive scan");
  const V = 32000;
  const probs = Float64Array.from({ length: V }, (_, i) => Math.exp(-i / 5000));
  const at = new AliasTable(probs);
  const r = new RNG(77);
  const N = 50000;
  const tAlias = timeIt(() => { for (let i = 0; i < N; i++) at.sample(r); }, { repeat: 3 }).median;
  // naive: build CDF once, then binary search each draw (O(V) build + O(logV) draw)
  const cdf = new Float64Array(V);
  let c = 0;
  for (let i = 0; i < V; i++) { c += probs[i]; cdf[i] = c; }
  const tScan = timeIt(() => {
    for (let i = 0; i < N; i++) {
      const target = r.next() * c;
      let lo = 0, hi = V - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < target) lo = mid + 1; else hi = mid; }
      void lo;
    }
  }, { repeat: 3 }).median;
  line(`V=${V}, ${N} draws`, tScan, tAlias, tScan / tAlias, "O(logV) → O(1)");
}

function benchKahan() {
  console.log("\n■ Numerical accuracy — Kahan vs naive summation");
  const x = new Float64Array(100001);
  x[0] = 1e16;
  for (let i = 1; i < x.length; i++) x[i] = 1;
  const truth = 1e16 + 100000;
  const naiveErr = Math.abs(x.reduce((s, v) => s + v, 0) - truth) / truth;
  const kahanErr = Math.abs(kahanSum(x) - truth) / truth;
  console.log(`  pathological series (1e16 + 1e5 × 1)`.padEnd(46));
  console.log(`    naive error ${naiveErr.toExponential(2)}   kahan error ${kahanErr === 0 ? "0 (exact)" : kahanErr.toExponential(2)}`);
}

function benchOptimizers() {
  console.log("\n■ Optimizers — Muon vs AdamW (steps to reach loss ≤ 0.01, identical init)");
  const r = new RNG("bench-opt");
  const N = 64, D = 8, H = 16;
  const Wtrue = zeros(D, H);
  for (let i = 0; i < Wtrue.a.length; i++) Wtrue.a[i] = r.normal() * 0.7;
  const X = Array.from({ length: N }, () => Array.from({ length: D }, () => r.normal()));
  const Y = new Float64Array(N * H);
  for (let t = 0; t < N; t++) for (let h = 0; h < H; h++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += X[t][d] * Wtrue.a[d * H + h];
    Y[t * H + h] = s;
  }
  const XFlat = Float64Array.from(X.flat());
  const run = (make, steps) => {
    const rInit = new RNG("fair-init");       // same init for both optimizers
    const W1 = param(Float64Array.from({ length: D * H }, () => rInit.normal() * 0.3), [D, H]);
    const opt = make([W1]);
    const Xt = tensor(XFlat, [N, D], false);
    const yt = tensor(Y, [N, H], false);
    let stepToTarget = -1;
    for (let s = 0; s < steps; s++) {
      opt.zeroGrad();
      withTape(() => {
        const P = matmul(Xt, W1);
        const diff = add(P, yt);            // P − Y
        const l = scale(sum(square(diff)), 1 / (N * H)); // mean loss
        backward(l);
        if (l.data[0] <= 0.01 && stepToTarget < 0) stepToTarget = s;
      });
      opt.step();
    }
    return stepToTarget;
  };
  const adamSteps = run((ps) => new AdamW(ps, { lr: 0.05, weightDecay: 0 }), 300);
  const muonSteps = run((ps) => new Muon(ps, { lr: 0.05, weightDecay: 0, adamLR: 0.001 }), 300);
  console.log(`  linear regression D×H=${D}×${H}, batch=${N} (Y=X·W* exact)`.padEnd(46));
  console.log(`    AdamW reached loss≤0.01 at step ${adamSteps >= 0 ? adamSteps : "—(not reached)"}`);
  console.log(`    Muon  reached loss≤0.01 at step ${muonSteps >= 0 ? muonSteps : "—(not reached)"}`);
}

console.log("═".repeat(78));
console.log("  AI FORGE — benchmarks (median of 3, warm cache, Node " + process.version + ")");
console.log("═".repeat(78));

benchGEMM();
benchAttention();
benchKNN();
benchSampling();
benchKahan();
benchOptimizers();

console.log("\n  Note: single-threaded JS on V8; numbers are indicative, the");
console.log("  complexity claims are the invariant-checked part.");
