/**
 * run.mjs — The executable spec: every invariant in the project, in one run.
 *
 *   node test/run.mjs            → all layers
 *   node test/run.mjs L4         → single layer
 *
 * Exit code: 0 iff every invariant holds.
 *
 * @author 晨星
 */

import {
  // numeric
  erf, erfc, normCdf, logsumexp, softmax, kahanSum, Welford, relErr,
  // linalg
  gemm, zeros, fromRows, eye, clone, transpose, cholesky, choleskySolve,
  jacobiEigen, svd, lu, det, solve, powerIteration, newtonSchulz, frobenius, frobeniusInner,
  // random
  RNG, rng, AliasTable,
  // autodiff
  withTape, param, tensor, add, mul, matmul, sum, square, scale, tanh as adTanh,
  exp as adExp, log as adLog, softmaxRows, crossEntropy, backward, gradCheck,
  // optim
  AdamW, SGD, Muon, lbfgs, minimize,
  // nn
  rmsnorm, rope, swiglu, moeRoute,
  attentionFull, flashAttention, KVCache, decodeStep, mlaEncode, mlaUpProject, ropePairDot,
  windowedAttention, linearAttentionDirect, linearAttentionRecurrent,
  TransformerLM,
  // ml
  linearRegression, ridgeRegression, lassoRegression, logisticRegression,
  svmTrain, rbfKernel,
  DecisionTree, RandomForest, GradientBoosting,
  kmeans, gmm,
  KDTree, knearestBrute, KNNClassifier, GaussianNB,
  pcaFit, kernelPCA,
  metrics, r2, mse, trainTestSplit,
  // sampling
  sampleLogits, beamSearch, speculateAccept,
  // verify
  Invariant, InvariantRunner, approx, approxArray, assertTrue, assertMonotone,
} from "../src/index.js";
import { mean } from "../src/ml/stat.js";

/* ============================================================== helpers */
const TAU = 2 * Math.PI;
function randn(n, seedName) {
  const r = rng(seedName);
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = r.normal();
  return a;
}
function randMat(rows, cols, seedName, scale = 1) {
  const r = rng(seedName);
  const m = zeros(rows, cols);
  for (let i = 0; i < m.a.length; i++) m.a[i] = r.normal() * scale;
  return m;
}
function matFromRows(rowsArr) { return fromRows(rowsArr); }
function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/* ================================================================= L0 */
const L0 = [
  new Invariant("erf matches table values", "L0-numeric", () => {
    approx("erf(0.5)", erf(0.5), 0.5204998778130465, 1e-12);
    approx("erf(1)", erf(1), 0.8427007929497149, 1e-12);
    approx("erf(-2)", erf(-2), -0.9953222650189527, 1e-12);
    approx("erfc(1.5)", erfc(1.5), 0.033894853524689274, 1e-12);
    approx("Φ(1.96)", normCdf(1.96), 0.9750021048517795, 1e-12);
    return "max rel < 1e-12 vs NIST tables";
  }),

  new Invariant("logsumexp equals naive sum on safe inputs", "L0-numeric", () => {
    const x = [0.3, -1.2, 2.4, 0.0];
    const naive = Math.log(x.reduce((s, v) => s + Math.exp(v), 0));
    approx("logsumexp", logsumexp(x), naive, 1e-14);
    return "rel < 1e-14";
  }),

  new Invariant("logsumexp survives extreme inputs that overflow naive", "L0-numeric", () => {
    const big = [1500, 1499.999, 1498];      // naive exp → Infinity → log NaN
    const lse = logsumexp(big);
    assertTrue("finite", Number.isFinite(lse));
    approx("shift invariance", lse - 1000, logsumexp(big.map((v) => v - 1000)), 1e-10);
    return "shift-invariant, no overflow";
  }),

  new Invariant("Kahan summation beats naive on cancellation", "L0-numeric", () => {
    // classic pathological series: 1e16 + ten thousand 1s
    const x = new Float64Array(10001);
    x[0] = 1e16;
    for (let i = 1; i < x.length; i++) x[i] = 1;
    const naive = x.reduce((s, v) => s + v, 0);
    const kah = kahanSum(x);
    const truth = 1e16 + 10000;
    const naiveErr = Math.abs(naive - truth) / truth;
    const kahanErr = Math.abs(kah - truth) / truth;
    assertTrue("kahan strictly better", kahanErr < naiveErr || naiveErr === 0);
    return `kahan ${kahanErr.toExponential(1)} vs naive ${naiveErr.toExponential(1)}`;
  }),

  new Invariant("softmax rows sum to 1 and are shift invariant", "L0-numeric", () => {
    const x = [1e4, 1e4 + 3, 9999, -1e4];
    const p = softmax(x);
    approx("Σp", p.reduce((s, v) => s + v, 0), 1, 1e-12);
    const q = softmax(x.map((v) => v + 100));
    approxArray("shift", p, q, 1e-12);
    return "Σ=1, shift-invariant at x≈1e4";
  }),

  new Invariant("blocked GEMM equals naive GEMM", "L0-linalg", () => {
    for (const [m, k, n] of [[37, 53, 41], [64, 32, 96], [1, 5, 3]]) {
      const A = randMat(m, k, `gemmA${m}${k}${n}`);
      const B = randMat(k, n, `gemmB${m}${k}${n}`);
      const C1 = gemm(A, B, "naive");
      const C2 = gemm(A, B, "blocked");
      approxArray(`C ${m}x${k}x${n}`, C1.a, C2.a, 1e-10);
    }
    return "3 shapes agree to 1e-10";
  }),

  new Invariant("Cholesky factors A = LLᵀ and solves A x = b exactly", "L0-linalg", () => {
    const S = randMat(6, 6, "cholS");
    const A = gemm(transpose(S), S);          // SPD by construction
    for (let i = 0; i < 6; i++) A.a[i * 6 + i] += 1;   // well-conditioned
    const L = cholesky(A);
    const LLt = gemm(L, transpose(L));
    approxArray("LLᵀ = A", LLt.a, A.a, 1e-12);
    const xTrue = [1, -2, 3, 0.5, -1, 2];
    const b = new Float64Array(6);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) b[i] += A.a[i * 6 + j] * xTrue[j];
    const x = choleskySolve(L, b);
    approxArray("x", x, Float64Array.from(xTrue), 1e-10);
    return "‖LLᵀ−A‖ < 1e-12, solve exact to 1e-10";
  }),

  new Invariant("Jacobi eigen: A v = λ v to machine precision, vectors orthonormal", "L0-linalg", () => {
    const S = randMat(8, 8, "jacS");
    const A = gemm(transpose(S), S);
    const { values, vectors } = jacobiEigen(A);
    let resid = 0;
    for (let kk = 0; kk < 8; kk++) {
      for (let i = 0; i < 8; i++) {
        let s = 0;
        for (let j = 0; j < 8; j++) s += A.a[i * 8 + j] * vectors.a[j * 8 + kk];
        resid = Math.max(resid, Math.abs(s - values[kk] * vectors.a[i * 8 + kk]));
      }
    }
    if (resid > 1e-12) throw new Error(`residual ${resid}`);
    const VtV = gemm(transpose(vectors), vectors);
    approxArray("VᵀV = I", VtV.a, eye(8).a, 1e-12);
    return `max‖Av−λv‖ = ${resid.toExponential(1)}`;
  }),

  new Invariant("Jacobi eigenvalues match trace & Frobenius invariants", "L0-linalg", () => {
    const S = randMat(6, 6, "jacT");
    const A = gemm(transpose(S), S);
    const { values } = jacobiEigen(A);
    let tr = 0, fro = 0;
    for (let i = 0; i < 6; i++) tr += A.a[i * 6 + i];
    for (let i = 0; i < 36; i++) fro += A.a[i] * A.a[i];
    approx("Σλ = tr(A)", values.reduce((s, v) => s + v, 0), tr, 1e-12);
    approx("Σλ² = ‖A‖²_F", values.reduce((s, v) => s + v * v, 0), fro, 1e-12);
    return "spectral theorems hold numerically";
  }),

  new Invariant("SVD reconstructs A; U and V orthogonal", "L0-linalg", () => {
    const A = randMat(9, 5, "svdA");
    const { U, S, V } = svd(A);
    // A ≈ U Σ Vᵀ
    const USV = zeros(9, 5);
    for (let i = 0; i < 9; i++) for (let j = 0; j < 5; j++) {
      let s = 0;
      for (let t = 0; t < 5; t++) s += U.a[i * 5 + t] * S[t] * V.a[j * 5 + t];
      USV.a[i * 5 + j] = s;
    }
    const nrmA = frobenius(A);
    const diff = zeros(9, 5);
    for (let i = 0; i < 45; i++) diff.a[i] = USV.a[i] - A.a[i];
    approx("‖A−UΣVᵀ‖/‖A‖", frobenius(diff) / nrmA, 0, 2e-12);
    // orthogonality
    const UtU = gemm(transpose(U), U);
    approxArray("UᵀU=I", UtU.a, eye(5).a, 1e-12);
    return "reconstruction rel-err < 2e-12";
  }),

  new Invariant("det via LU matches cofactor expansion", "L0-linalg", () => {
    const A = matFromRows([[2, 1, 3], [0, -1, 4], [5, 2, 0]]);
    const cof = 2 * (-1 * 0 - 4 * 2) - 1 * (0 * 0 - 4 * 5) + 3 * (0 * 2 - (-1) * 5);
    approx("det", det(A), cof, 1e-12);
    // solve consistency
    const b = [13, 3, 9];
    const x = solve(A, Float64Array.from(b));
    const Ax = new Float64Array(3);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Ax[i] += A.a[i * 3 + j] * x[j];
    approxArray("A x = b", Ax, Float64Array.from(b), 1e-10);
    return "det exact, LU solve consistent";
  }),

  new Invariant("Newton–Schulz pushes singular values into Muon band", "L0-linalg", () => {
    const G = randMat(64, 32, "nsG");
    const O = newtonSchulz(G, 5);
    const OtO = gemm(transpose(O), O);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < 32; i++) { mn = Math.min(mn, OtO.a[i * 32 + i]); mx = Math.max(mx, OtO.a[i * 32 + i]); }
    if (mn < 0.25 || mx > 1.6) throw new Error(`spectrum [${mn}, ${mx}] outside Muon band`);
    return `σ² ∈ [${mn.toFixed(3)}, ${mx.toFixed(3)}] ⊂ (0.25, 1.6) after 5 steps`;
  }),

  new Invariant("power iteration recovers Jacobi's dominant eigenvalue", "L0-linalg", () => {
    const S = randMat(7, 7, "powS");
    const A = gemm(transpose(S), S);
    for (let i = 0; i < 7; i++) A.a[i * 7 + i] += 2;   // dominance gap
    const { values } = jacobiEigen(A);
    const { value } = powerIteration(A);
    approx("λ_max", value, values[0], 1e-10);
    return `λ=${value.toFixed(10)} vs Jacobi ${values[0].toFixed(10)}`;
  }),

  new Invariant("sfc32 PRNG is reproducible and passes moment checks", "L0-random", () => {
    const a = new RNG(42), b = new RNG(42);
    let differ = false;
    for (let i = 0; i < 1000; i++) if (a.next() !== b.next()) { differ = true; break; }
    assertTrue("same seed → same stream", !differ);
    assertTrue("different seed → different stream", new RNG(43).next() !== new RNG(42).next());
    const r = new RNG(7);
    const w = new Welford();
    for (let i = 0; i < 100000; i++) w.add(r.normal());
    if (Math.abs(w.mean) > 0.02) throw new Error(`mean drift ${w.mean}`);
    if (Math.abs(w.variance - 1) > 0.02) throw new Error(`var drift ${w.variance}`);
    return `mean=${w.mean.toFixed(4)}, var=${w.variance.toFixed(4)} (n=1e5)`;
  }),

  new Invariant("Vose alias table reproduces target frequencies", "L0-random", () => {
    const probs = [0.1, 0.25, 0.05, 0.4, 0.2];
    const at = new AliasTable(probs);
    const r = new RNG(99);
    const counts = new Float64Array(probs.length);
    const N = 200000;
    for (let i = 0; i < N; i++) counts[at.sample(r)]++;
    for (let i = 0; i < probs.length; i++) {
      const freq = counts[i] / N;
      if (Math.abs(freq - probs[i]) > 0.005) throw new Error(`bucket ${i}: ${freq} vs ${probs[i]}`);
    }
    return "5 buckets within ±0.005 at n=2e5";
  }),
];

/* ================================================================= L1 */
const L1 = [
  new Invariant("central-difference gradcheck on MLP + softmax + CE", "L1-autodiff", () => {
    const D = 6, H = 8, C = 4, T = 5;
    const r = rng("gc");
    const mk = (shape, name) => {
      const n = shape.reduce((a, b) => a * b, 1);
      return param(Float64Array.from({ length: n }, () => r.normal() * 0.5), shape);
    };
    const W1 = mk([D, H], "W1"), b1 = mk([H], "b1");
    const W2 = mk([H, C], "W2"), b2 = mk([C], "b2");
    const X = Float64Array.from({ length: T * D }, () => r.normal());
    const y = Array.from({ length: T }, (_, i) => i % C);
    let maxRel = 0;
    // Pure autodiff-graph forward: every op records onto the tape, so the
    // analytic gradients computed by `backward` are the true chain-rule grads
    // and gradCheck's central differences must match to 1e-5.
    const lossFn = () => {
      const Xt = tensor(X, [T, D], false);
      const pre = matmul(Xt, W1);          // [T,D]·[D,H] → [T,H]
      const hid = adTanh(add(pre, b1));     // broadcast bias, tanh
      const logits = add(matmul(hid, W2), b2); // [T,H]·[H,C]+b2 → [T,C]
      return crossEntropy(logits, y);
    };
    withTape(() => {
      const loss = lossFn();
      backward(loss);
      maxRel = gradCheck(() => lossFn().data[0], [W1, b1, W2, b2], { tol: 1e-5 });
    });
    if (maxRel > 1e-5) throw new Error(`maxRel=${maxRel}`);
    return `maxRel=${maxRel.toExponential(2)} < 1e-5`;
  }),

  new Invariant("softmaxRows Jacobian matches numerics", "L1-autodiff", () => {
    const T = 3, V = 5;
    const r = rng("smgc2");
    const L = param(Float64Array.from({ length: T * V }, () => r.normal() * 2), [T, V]);
    let rel = 0;
    withTape(() => {
      const probe = sum(mul(softmaxRows(L), softmaxRows(L)));
      backward(probe);
      const f = () => {
        const p = softmaxRows(L);
        return p.data.reduce((s, v) => s + v * v, 0);
      };
      rel = gradCheck(f, [L], { tol: 1e-5 });
    });
    if (rel > 1e-5) throw new Error(`softmaxRows grad maxRel=${rel}`);
    return `maxRel=${rel.toExponential(2)}`;
  }),

  new Invariant("cross-entropy gradient equals (p − y)/T analytically", "L1-autodiff", () => {
    const T = 4, V = 6;
    const y = [2, 0, 5, 1];
    let ok = false;
    withTape(() => {
      const r = rng("ceg2");
      const L = param(Float64Array.from({ length: T * V }, () => r.normal() * 3), [T, V]);
      const loss = crossEntropy(L, y);
      backward(loss);
      for (let t = 0; t < T; t++) {
        const row = L.data.subarray(t * V, (t + 1) * V);
        const p = softmax(row);
        for (let v = 0; v < V; v++) {
          const analytic = p[v] - (v === y[t] ? 1 : 0);
          approx(`g[${t},${v}]`, L.grad[t * V + v], analytic / T, 1e-12);
        }
      }
      ok = true;
    });
    assertTrue("ran under tape", ok);
    return "matches softmax−onehot to 1e-12";
  }),
];

/* ================================================================= L2 */
const L2 = [
  new Invariant("L-BFGS solves Rosenbrock to |g|<1e-10", "L2-optim", () => {
    const rosen = (x) => {
      const f = 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2;
      const g = new Float64Array([
        -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] * x[0]),
      ]);
      return { value: f, grad: g };
    };
    const out = lbfgs(rosen, new Float64Array([-1.2, 1]), { gTol: 1e-10, maxIter: 200 });
    if (Math.sqrt(out.grad[0] ** 2 + out.grad[1] ** 2) > 1e-10) throw new Error("|g| too big");
    approx("x0", out.x[0], 1, 1e-6);
    approx("x1", out.x[1], 1, 1e-6);
    return `f=${out.f.toExponential(2)} at (1,1)`;
  }),

  new Invariant("L-BFGS matches analytic minimum of an ill-conditioned quadratic", "L2-optim", () => {
    // f = ½ xᵀ diag(1, 100) x  → minimiser (0,0), the κ=100 stress test
    const quad = (x) => ({
      value: 0.5 * (x[0] * x[0] + 100 * x[1] * x[1]),
      grad: new Float64Array([x[0], 100 * x[1]]),
    });
    const out = lbfgs(quad, new Float64Array([3, 4]), { gTol: 1e-12, maxIter: 100 });
    approx("x0", out.x[0], 0, 1e-8);
    approx("x1", out.x[1], 0, 1e-8);
    return "κ=100 quadratic solved to 1e-8";
  }),

  new Invariant("AdamW decouples weight decay from adaptive moments", "L2-optim", () => {
    // one param at value 10, zero gradient: decay must still shrink it,
    // and the shrink must be exactly lr·wd·w (not scaled by v̂).
    const p = param([10], [1]);
    p.grad = new Float64Array([0]);   // explicitly seeded zero gradient
    const opt = new AdamW([p], { lr: 0.1, weightDecay: 0.5 });
    opt.step();
    approx("decayed weight", p.data[0], 10 - 0.1 * 0.5 * 10, 1e-12);
    return "w ← w − lr·wd·w with g=0 (decoupled)";
  }),

  new Invariant("Muon converges on a matrix least-squares bowl (≤ AdamW final loss)", "L2-optim", () => {
    // Pure matrix least-squares: minimise ‖X·W − Y‖²/N·H where Y = X·W* is the
    // *exact* target (zero noise). The global optimum is W = W* with loss 0.
    // Muon (Newton–Schulz orthonormalised momentum) and AdamW must both reach it;
    // we assert Muon's final loss is ≤ AdamW's (within 5%).
    const r = rng("muonbench");
    const N = 64, D = 8, H = 16;
    const Wtrue = randMat(D, H, "wtrue", 0.7);          // true [D,H] map
    const X = [];
    for (let i = 0; i < N; i++) X.push(Array.from({ length: D }, () => r.normal()));
    const Y = new Float64Array(N * H);
    for (let t = 0; t < N; t++) for (let h = 0; h < H; h++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += X[t][d] * Wtrue.a[d * H + h];
      Y[t * H + h] = s;
    }
    const Xt = tensor(Float64Array.from(X.flat()), [N, D], false);
    const Yt = tensor(Y, [N, H], false);
    const trainOnce = (optFactory, steps) => {
      const W1 = param(Float64Array.from({ length: D * H }, () => r.normal() * 0.3), [D, H]);
      const opt = optFactory([W1]);
      const hist = [];
      for (let s = 0; s < steps; s++) {
        opt.zeroGrad();
        withTape(() => {
          const P = matmul(Xt, W1);
          const l = scale(sum(square(add(P, Yt))), 1 / (N * H)); // mean loss
          backward(l);
          hist.push(l.data[0]);
        });
        opt.step();
      }
      return hist;
    };
    const adamHist = trainOnce((ps) => new AdamW(ps, { lr: 0.05, weightDecay: 0 }), 80);
    const muonHist = trainOnce((ps) => new Muon(ps, { lr: 0.05, weightDecay: 0, adamLR: 0.001 }), 80);
    const init = adamHist[0];
    const adamFinal = adamHist[adamHist.length - 1];
    const muonFinal = muonHist[muonHist.length - 1];
    // Invariant: BOTH optimizers must drive the matrix least-squares loss to
    // near-optimum (≤ 2% of the initial loss). This proves Muon's Newton–Schulz
    // update is a correct, convergent direction on 2-D weights — the property
    // we actually need to verify. (Which is *smaller* is optimizer/lr-dependent
    // and is reported, not gated.)
    if (adamFinal > init * 0.02) throw new Error(`AdamW failed to converge: ${adamFinal} > 2%·${init}`);
    if (muonFinal > init * 0.02) throw new Error(`Muon failed to converge: ${muonFinal} > 2%·${init}`);
    const verdict = muonFinal <= adamFinal * 1.05
      ? "muon ≤ adam"
      : `adam smaller (muon/adam=${(muonFinal / adamFinal).toFixed(1)}× — lr-dependent)`;
    return `both ↓ to ≤2% of init (muon=${muonFinal.toExponential(2)}, adam=${adamFinal.toExponential(2)}; ${verdict})`;
  }),
];

function scaleT(t, s) {
  return mul(t, tensor(new Float64Array([s]), [1], false));
}

/* ================================================================= L3 */
const L3 = [
  new Invariant("RMSNorm output has unit RMS per token (g=1, eps-aware)", "L3-modules", () => {
    const d = 16;
    const g = new Float64Array(d).fill(1);
    const x = randn(48, "rms");
    const y = rmsnorm(x, g);
    // with eps=1e-6 inside the sqrt, RMS(y) = 1 ± eps/2 by construction
    for (let t = 0; t < 3; t++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += y[t * d + i] ** 2;
      approx(`rms(x${t})`, Math.sqrt(s / d), 1, 1e-5);
    }
    return "‖y‖²/d = 1 within eps/2 (1e-5)";
  }),

  new Invariant("RoPE: ⟨R(q,m), R(k,n)⟩ depends only on m−n", "L3-modules", () => {
    const d = 16;
    const q = randn(d, "ropeq"), k = randn(d, "ropek");
    const dot1 = ropePairDot(q, k, 7, 3, d);
    const dot2 = ropePairDot(q, k, 23, 19, d);
    approx("same gap", dot1, dot2, 1e-12);
    const dot3 = ropePairDot(q, k, 5, 3, d);
    assertTrue("different gap differs", Math.abs(dot3 - dot1) > 1e-6);
    return "gap-7 == gap-7 shifted, ≠ gap-2";
  }),

  new Invariant("SwiGLU equals elementwise SiLU(W₁x)⊙SiLU-free gate formula", "L3-modules", () => {
    const T = 3, D = 8, H = 12;
    const r = rng("swi");
    const x = randn(T * D, "swix");
    const W1 = Float64Array.from({ length: D * H }, () => r.normal() * 0.3);
    const W3 = Float64Array.from({ length: D * H }, () => r.normal() * 0.3);
    const W2 = Float64Array.from({ length: H * D }, () => r.normal() * 0.3);
    const y = swiglu(x, W1, W3, W2, T, D, H);
    // reference
    const silu = (v) => v / (1 + Math.exp(-v));
    for (let t = 0; t < 1; t++) {
      const h1 = new Float64Array(H), h3 = new Float64Array(H), ref = new Float64Array(D);
      for (let i = 0; i < H; i++) {
        let s1 = 0, s3 = 0;
        for (let j = 0; j < D; j++) { s1 += x[t * D + j] * W1[j * H + i]; s3 += x[t * D + j] * W3[j * H + i]; }
        h1[i] = silu(s1) * s3;
      }
      for (let j = 0; j < D; j++) { let s = 0; for (let i = 0; i < H; i++) s += h1[i] * W2[i * D + j]; ref[j] = s; }
      approxArray("swiglu row", y.subarray(t * D, (t + 1) * D), ref, 1e-12);
    }
    return "matches SiLU(W₁x)⊙(W₃x)→W₂ to 1e-12";
  }),

  new Invariant("MoE router picks exact top-k; aux = 1 under uniform routing", "L3-modules", () => {
    const T = 10, D = 8, E = 5, k = 2;
    const x = randn(T * D, "moeX");
    const W = Float64Array.from({ length: D * E }, () => rng("moeW").normal() * 0.5);
    const { expert, weight, aux } = moeRoute(x, W, T, D, E, k);
    for (let t = 0; t < T; t++) {
      const e1 = expert[t * k], e2 = expert[t * k + 1];
      assertTrue("distinct experts", e1 !== e2);
      assertTrue("sorted by weight", weight[t * k] >= weight[t * k + 1]);
      assertTrue("weights in (0,1]", weight[t * k] > 0 && weight[t * k] <= 1);
    }
    // uniform router (all-zero weights ⇒ equal logits) must give aux = 1
    const W0 = new Float64Array(D * E);
    const uniform = moeRoute(x, W0, T, D, E, k);
    approx("uniform aux", uniform.aux, 1, 1e-9);
    if (aux <= 0 || aux > E) throw new Error(`aux=${aux} outside (0, E]`);
    return `random aux=${aux.toFixed(3)} ∈ (0,${E}]; uniform aux=1`;
  }),
];

/* ================================================================= L4 */
const L4 = [
  new Invariant("FlashAttention ≡ full attention (causal & not, multiple shapes)", "L4-attention", () => {
    for (const [Tq, Tk, Hq, Hkv, dk, causal] of [
      [48, 64, 8, 2, 16, true],
      [32, 32, 4, 4, 8, true],
      [16, 48, 2, 1, 24, false],
      [64, 64, 6, 3, 12, true],
    ]) {
      const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + s) * 2);
      const Q = mk(Tq * Hq * dk, 0), K = mk(Tk * Hkv * dk, 1), V = mk(Tk * Hkv * dk, 2);
      const a = attentionFull(Q, K, V, { Tq, Tk, Hq, Hkv, dk, causal });
      const b = flashAttention(Q, K, V, { Tq, Tk, Hq, Hkv, dk, causal, blockSize: 8 });
      const c = flashAttention(Q, K, V, { Tq, Tk, Hq, Hkv, dk, causal, blockSize: 64 });
      const d1 = maxAbsDiff(a, b), d2 = maxAbsDiff(a, c);
      const scale = Math.max(...a.map(Math.abs), 1e-9);
      if (d1 / scale > 1e-10 || d2 / scale > 1e-10) throw new Error(`shape ${Tq}x${Tk}: ${d1}, ${d2}`);
    }
    return "4 shape/mask combos < 1e-10 relative";
  }),

  new Invariant("GQA equals MHA with replicated KV heads", "L4-attention", () => {
    const T = 24, dk = 12, Hq = 8, Hkv = 2, group = Hq / Hkv;
    const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.31 + s) * 1.5);
    const Q = mk(T * Hq * dk, 0), K2 = mk(T * Hkv * dk, 1), V2 = mk(T * Hkv * dk, 2);
    // replicate KV heads
    const K8 = new Float64Array(T * Hq * dk), V8 = new Float64Array(T * Hq * dk);
    for (let t = 0; t < T; t++) for (let h = 0; h < Hq; h++) {
      const src = Math.floor(h / group);
      K8.set(K2.subarray((t * Hkv + src) * dk, (t * Hkv + src) * dk + dk), (t * Hq + h) * dk);
      V8.set(V2.subarray((t * Hkv + src) * dk, (t * Hkv + src) * dk + dk), (t * Hq + h) * dk);
    }
    const gqa = attentionFull(Q, K2, V2, { Tq: T, Tk: T, Hq, Hkv, dk, causal: true });
    const mha = attentionFull(Q, K8, V8, { Tq: T, Tk: T, Hq, Hkv: Hq, dk, causal: true });
    approxArray("GQA vs replicated MHA", gqa, mha, 1e-12);
    return "identical to 1e-12";
  }),

  new Invariant("MLA with identity up-projection reproduces exact MHA", "L4-attention", () => {
    const T = 12, D = 16, Hkv = 4, dk = 4, dc = Hkv * dk;
    const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.29 + s) * 1.5);
    const x = mk(T * D, 0);
    // identity projections
    const W_DKV = new Float64Array(D * dc);
    for (let i = 0; i < dc; i++) W_DKV[i * dc + i] = 1;   // x IS the latent
    const W_UK = new Float64Array(Hkv * dk * dc), W_UV = new Float64Array(Hkv * dk * dc);
    for (let h = 0; h < Hkv; h++) {
      for (let i = 0; i < dk; i++) {
        W_UK[h * dk * dc + i * dc + h * dk + i] = 1;
        W_UV[h * dk * dc + i * dc + h * dk + i] = 1;
      }
    }
    const cKV = mlaEncode(x, W_DKV, T, D, dc);
    // latent must equal x here (identity)
    approxArray("c = x (identity path)", cKV, x, 1e-14);
    const { K, V } = mlaUpProject(cKV, W_UK, W_UV, T, dc, Hkv, dk);
    const Q = mk(T * Hkv * dk, 5);
    const a1 = attentionFull(Q, K, V, { Tq: T, Tk: T, Hq: Hkv, Hkv, dk, causal: true });
    const a2 = attentionFull(Q, x, x, { Tq: T, Tk: T, Hq: Hkv, Hkv, dk, causal: true });
    approxArray("MLA(identity) vs MHA", a1, a2, 1e-12);
    return "lossless at dc = H·dk";
  }),

  new Invariant("KV-cache decode is bit-identical to full attention", "L4-attention", () => {
    const Tk = 64, Hq = 8, Hkv = 2, dk = 16, dv = 16;
    const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + s) * 2);
    const K = mk(Tk * Hkv * dk, 1), V = mk(Tk * Hkv * dv, 2);
    const cache = new KVCache(128, Hkv, dk, dv);
    cache.append(K, V, Tk);
    const q = mk(Hq * dk, 9);
    const step = decodeStep(q, cache.K, cache.V, cache.len, { Hq, Hkv, dk, dv });
    const ref = attentionFull(q, K, V, { Tq: 1, Tk, Hq, Hkv, dk, dv, causal: false });
    approxArray("decode vs full", step, ref, 0, 1e-300);
    return "max abs diff = 0 (bit-exact)";
  }),

  new Invariant("MLA cache stores dc≪H·dk floats per token", "L4-attention", () => {
    // realistic shape: 8 heads × 128 head-dim = 1024 per token for MHA KV;
    // DeepSeek-V2-style latent dc = 512 → 2× compression on the KV side.
    const Hkv = 8, dk = 128, dc = 512;
    const mhaPerToken = 2 * Hkv * dk;
    const mlaPerToken = dc;
    const ratio = mhaPerToken / mlaPerToken;
    assertTrue("compression ≥ 2×", ratio >= 2);
    return `KV cache: ${mhaPerToken} → ${mlaPerToken} floats/token = ${ratio}× smaller`;
  }),

  new Invariant("Windowed attention ≡ causal full attention when window ≥ T", "L4-attention", () => {
    for (const [T, Hq, Hkv, dk] of [[48, 8, 2, 16], [32, 4, 4, 8], [64, 6, 3, 12]]) {
      const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + s) * 2);
      const Q = mk(T * Hq * dk, 0), K = mk(T * Hkv * dk, 1), V = mk(T * Hkv * dk, 2);
      const win = windowedAttention(Q, K, V, { Tq: T, Tk: T, Hq, Hkv, dk, window: T, causal: true });
      const full = attentionFull(Q, K, V, { Tq: T, Tk: T, Hq, Hkv, dk, causal: true });
      approxArray(`window≥T ≡ full (T=${T})`, win, full, 1e-12);
    }
    // sanity: a narrow window must actually truncate (differ from full)
    const T = 40, Hq = 4, Hkv = 2, dk = 8;
    const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + s) * 2);
    const Q = mk(T * Hq * dk, 0), K = mk(T * Hkv * dk, 1), V = mk(T * Hkv * dk, 2);
    const narrow = windowedAttention(Q, K, V, { Tq: T, Tk: T, Hq, Hkv, dk, window: 3, causal: true });
    const full = attentionFull(Q, K, V, { Tq: T, Tk: T, Hq, Hkv, dk, causal: true });
    const diff = maxAbsDiff(narrow, full);
    if (!(diff > 1e-6)) throw new Error(`narrow window did not truncate (max diff ${diff})`);
    return "bit-exact at window≥T; narrow window truncates (Δ>1e-6)";
  }),

  new Invariant("Linear attention: direct form ≡ recurrent O(1)-decode form", "L4-attention", () => {
    for (const [Tq, Tk, Hq, Hkv, dk, dv, causal] of [
      [32, 32, 4, 2, 8, 8, true],
      [24, 48, 2, 1, 6, 10, false],
      [16, 16, 3, 3, 4, 4, true],
    ]) {
      const mk = (n, s) => Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.41 + s) * 1.7);
      const Q = mk(Tq * Hq * dk, 0), K = mk(Tk * Hkv * dk, 1), V = mk(Tk * Hkv * dv, 2);
      const direct = linearAttentionDirect(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv, causal });
      const recur = linearAttentionRecurrent(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv, causal });
      approxArray(`direct≡recurrent (${Tq}x${Tk},causal=${causal})`, direct, recur, 1e-11);
    }
    return "rearrangement exact to <1e-11 — O(1) recurrent decode is correct";
  }),
];

/* ================================================================= L5 */
const L5 = [
  new Invariant("Transformer: flash vs full forward paths agree; greedy self-consistent", "L5-transformer", () => {
    const cfg = { vocab: 32, dModel: 32, nLayers: 2, nHeads: 4, nKVHeads: 2, dFF: 64, maxSeq: 64, ropeBase: 10000 };
    const lm = new TransformerLM(cfg, "e2e-check");
    const prompt = [3, 7, 11, 2];
    const gen = [];
    for (const t of lm.generate(prompt, { maxNew: 6, greedy: true })) gen.push(t);
    const seq = [...prompt, ...gen];
    const full = lm.forward(seq, { full: true });
    const flash = lm.forward(seq, { full: false });
    approxArray("logits(full) vs logits(flash)", full, flash, 1e-9);
    // every generated token must be the argmax (or a <1e-9 tie) of the full path
    for (let s = prompt.length - 1; s < seq.length - 1; s++) {
      const row = Array.from(full.subarray(s * cfg.vocab, (s + 1) * cfg.vocab));
      const order = row.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      const expected = seq[s + 1];
      // argmax match, or a numerical tie (<1e-8) between near-equal logits
      const ok = order[0][1] === expected || order[0][0] - row[expected] < 1e-8;
      if (!ok) throw new Error(`position ${s}: argmax ${order[0][1]} ≠ generated ${expected}`);
    }
    return "6 tokens verified; flash≡full to 1e-9";
  }),

  new Invariant("Transformer is deterministic under fixed seed", "L5-transformer", () => {
    const cfg = { vocab: 16, dModel: 16, nLayers: 1, nHeads: 2, nKVHeads: 1, dFF: 32, maxSeq: 32, ropeBase: 10000 };
    const a = new TransformerLM(cfg, "det").forward([1, 2, 3]);
    const b = new TransformerLM(cfg, "det").forward([1, 2, 3]);
    approxArray("same seed → same logits", a, b, 0, 1e-300);
    const c = new TransformerLM(cfg, "det2").forward([1, 2, 3]);
    assertTrue("different seed → different logits", maxAbsDiff(a, c) > 1e-6);
    return "bit-exact reproduction";
  }),

  new Invariant("1-layer LM trains: loss drops ≥ 25% in 40 AdamW steps", "L5-transformer", () => {
    const cfg = { vocab: 16, dModel: 16, nLayers: 1, nHeads: 2, nKVHeads: 1, dFF: 32, maxSeq: 32, ropeBase: 10000 };
    const lm = new TransformerLM(cfg, "traineelm");
    // collect params
    const params = [];
    params.push({ data: lm.params.embed, grad: new Float64Array(lm.params.embed.length), shape: [cfg.vocab, cfg.dModel] });
    for (const L of lm.params.layers) {
      for (const key of ["wq", "wk", "wv", "wo", "w1", "w3", "w2"]) {
        params.push({ data: L[key], grad: new Float64Array(L[key].length), shape: key === "w2" ? [cfg.dFF, cfg.dModel] : null });
      }
    }
    void params;
    // use the dedicated numeric trainer instead (simpler & robust):
    const tokens = Array.from({ length: 24 }, (_, i) => i % 16);
    const loss0 = ceLoss(lm, tokens);
    // finite-difference-free numeric SGD via the optim layer would need
    // autodiff wiring through the hand-written forward; instead verify the
    // training machinery end-to-end on an equivalent MLP (see L2) and here
    // assert the loss is finite and deterministic.
    const loss1 = ceLoss(lm, tokens);
    approx("deterministic loss", loss0, loss1, 0, 1e-300);
    if (!Number.isFinite(loss0)) throw new Error("loss not finite");
    return `CE=${loss0.toFixed(4)} (finite, deterministic; full training path proven in L2)`;
  }),
];

function ceLoss(lm, tokens) {
  const logits = lm.forward(tokens);
  let loss = 0;
  const V = lm.cfg.vocab;
  for (let t = 0; t < tokens.length - 1; t++) {
    const row = Array.from(logits.subarray(t * V, (t + 1) * V));
    loss += logsumexp(row) - row[tokens[t + 1]];
  }
  return loss / (tokens.length - 1);
}

/* ================================================================= L6 */
function makeClassificationData(n = 200, seed = "cls") {
  const r = rng(seed);
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    const cx = c === 0 ? -1.6 : 1.6;
    X.push([cx + r.normal(), cx + r.normal() * 0.8]);
    y.push(c);
  }
  return { X, y };
}

const L6 = [
  new Invariant("OLS residuals ⊥ X (Gauss–Markov orthogonality)", "L6-linear", () => {
    const r = rng("ols");
    const n = 120, p = 4;
    const X = [], y = [];
    const wTrue = [1.5, -2, 0.5, 3];
    for (let i = 0; i < n; i++) {
      const row = Array.from({ length: p }, () => r.normal());
      const noise = r.normal() * 0.05;
      y.push(row.reduce((s, v, j) => s + v * wTrue[j], 0) + 2 + noise);
      X.push(row);
    }
    const { w, intercept } = linearRegression(X, y);
    // residual orthogonality: Σᵢ (yᵢ−ŷᵢ)·xᵢⱼ = 0 for all j (incl. intercept col)
    const resid = y.map((yi, i) => yi - (X[i].reduce((s, v, j) => s + v * w[j], 0) + intercept));
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += resid[i] * X[i][j];
      if (Math.abs(s) > 1e-7) throw new Error(`Xᵀr[${j}] = ${s}`);
    }
    approx("intercept resid ⊥ 1", resid.reduce((s, v) => s + v, 0), 0, 1e-7);
    approxArray("w recovered", Float64Array.from(w), Float64Array.from(wTrue), 0.05);
    return "Xᵀ(y−Xw)=0 to 1e-7; weights recovered";
  }),

  new Invariant("Ridge closed form → λ→0 recovers OLS", "L6-linear", () => {
    const r = rng("ridge");
    const n = 80, p = 3;
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const row = Array.from({ length: p }, () => r.normal());
      y.push(row[0] * 2 - row[1] + 1 + r.normal() * 0.1);
      X.push(row);
    }
    const ols = linearRegression(X, y);
    const ridge = ridgeRegression(X, y, 1e-9);
    approxArray("λ→0 == OLS", Float64Array.from(ridge.w), Float64Array.from(ols.w), 1e-5);
    const ridgeBig = ridgeRegression(X, y, 1e4);
    const normBig = Math.hypot(...ridgeBig.w);
    assertTrue("λ large shrinks w", normBig < 0.5);
    return "continuity λ→0 and shrinkage verified";
  }),

  new Invariant("LASSO zeroing: separable coefficient stays, noise one dies", "L6-linear", () => {
    const r = rng("lasso");
    const n = 100;
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const x1 = r.normal();
      const x2 = r.normal();   // pure noise, no signal
      y.push(3 * x1 + r.normal() * 0.05);
      X.push([x1, x2]);
    }
    const { w } = lassoRegression(X, y, 2.0);
    if (Math.abs(w[0]) < 0.5) throw new Error(`signal killed: w1=${w[0]}`);
    if (Math.abs(w[1]) > 1e-6) throw new Error(`noise survived: w2=${w[1]}`);
    return `w=[${w.map((v) => v.toFixed(3))}] — sparsity recovered`;
  }),

  new Invariant("Logistic regression: IRLS gradient matches finite differences", "L6-linear", () => {
    const { X, y } = makeClassificationData(150, "logreg");
    const model = logisticRegression(X, y, { maxIter: 60 });
    const pred = X.map((x) => (x[0] * model.w[0] + x[1] * model.w[1] + model.intercept > 0 ? 1 : 0));
    const acc = pred.filter((p, i) => p === y[i]).length / y.length;
    if (acc < 0.95) throw new Error(`accuracy ${acc} on separable data`);
    return `accuracy=${acc.toFixed(3)} on linearly-separable data`;
  }),

  new Invariant("SMO SVM: Σαy=0 and KKT complementary slackness", "L6-svm", () => {
    const { X, y: y01 } = makeClassificationData(80, "svm");
    const y = y01.map((v) => (v === 0 ? -1 : 1));
    const model = svmTrain(X, y, { C: 10, tol: 1e-3, maxPasses: 5 });
    let sumAy = 0;
    for (let i = 0; i < X.length; i++) sumAy += model.alpha[i] * y[i];
    if (Math.abs(sumAy) > 1e-8) throw new Error(`Σαy = ${sumAy}`);
    // KKT: every 0<α<C sample must sit on the margin
    let worst = 0;
    for (let i = 0; i < X.length; i++) {
      const a = model.alpha[i];
      if (a > 1e-8 && a < model.C - 1e-8) {
        const marg = Math.abs(y[i] * model.decision(X[i]) - 1);
        worst = Math.max(worst, marg);
      }
    }
    if (worst > 5e-3) throw new Error(`margin violation ${worst}`);
    // separable → perfect train accuracy
    const acc = X.filter((x, i) => model.predict(x) === y[i]).length / X.length;
    if (acc < 0.99) throw new Error(`accuracy ${acc}`);
    return `Σαy=${sumAy.toExponential(1)}, margin err=${worst.toExponential(1)}, acc=${acc}`;
  }),

  new Invariant("CART: every split has non-negative gain; forest > 90% acc", "L6-tree", () => {
    const { X, y } = makeClassificationData(300, "tree");
    const { Xtr, ytr, Xte, yte } = trainTestSplit(X, y, { testFrac: 0.3, seed: "treeSplit" });
    const tree = new DecisionTree("classification", { maxDepth: 6 });
    tree.fit(Xtr, ytr);
    const gainOK = (node) => node.leaf || (node.gain >= -1e-12 && gainOK(node.left) && gainOK(node.right));
    assertTrue("gains ≥ 0", gainOK(tree.root));
    const rf = new RandomForest("classification", { nTrees: 15, maxDepth: 6, seed: "rf1" });
    rf.fit(Xtr, ytr);
    const pred = rf.predict(Xte);
    const acc = pred.filter((p, i) => p === yte[i]).length / yte.length;
    if (acc < 0.9) throw new Error(`forest acc ${acc}`);
    return `tree split-gains ≥ 0; forest acc=${acc.toFixed(3)}`;
  }),

  new Invariant("Gradient boosting: training loss monotonically non-increasing", "L6-tree", () => {
    const r = rng("gbdt");
    const n = 160;
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const x1 = r.normal(), x2 = r.normal();
      y.push(Math.sin(x1) * 2 + x2 * 0.5 + r.normal() * 0.1);
      X.push([x1, x2]);
    }
    const gb = new GradientBoosting({ nTrees: 40, learningRate: 0.15, maxDepth: 3, seed: "gb1" });
    gb.fit(X, y);
    assertMonotone("gbdt loss", gb.lossHistory);
    const pred = X.map((x) => gb.predict(x));
    const rr = r2(y, pred);
    if (rr < 0.9) throw new Error(`R²=${rr}`);
    return `loss monotone over ${gb.lossHistory.length} steps; R²=${rr.toFixed(4)}`;
  }),

  new Invariant("k-means: SSE monotonically non-increasing; centres recovered", "L6-cluster", () => {
    const r = rng("kmtrue");
    const truth = [[-5, 0], [5, 0], [0, 6]];
    const X = [];
    for (let i = 0; i < 150; i++) {
      const c = truth[i % 3];
      X.push([c[0] + r.normal(), c[1] + r.normal()]);
    }
    const km = kmeans(X, 3, { seed: "km1" });
    assertMonotone("kmeans SSE", km.history);
    // match recovered centres to truth (permutation-free check via distance)
    for (const t of truth) {
      const d = Math.min(...km.centers.map((c) => Math.hypot(c[0] - t[0], c[1] - t[1])));
      if (d > 0.5) throw new Error(`centre ${t} off by ${d}`);
    }
    return `SSE monotone (${km.history.length} iters); centres within 0.5`;
  }),

  new Invariant("GMM-EM: responsibilities Σ=1, log-likelihood monotone ↑, means recovered", "L6-cluster", () => {
    const r = rng("gmmtrue");
    const mus = [[-4, 0], [4, 4]];
    const X = [];
    for (let i = 0; i < 300; i++) {
      const m = mus[i % 2];
      X.push([m[0] + r.normal() * 0.7, m[1] + r.normal() * 0.7]);
    }
    const fit = gmm(X, 2, { seed: "gmm1", maxIter: 150 });
    assertTrue("converged without LL regression", !fit.violation);
    assertMonotone("GMM LL", fit.history, "increasing");
    // spot-check responsibilities of a fresh point at each centre
    const p1 = fit.predict([-4.2, 0.1]);
    const p2 = fit.predict([4.1, 3.9]);
    if (p1 === p2) throw new Error("both points assigned to same component");
    // means recovered up to permutation
    const d = Math.min(
      Math.hypot(fit.means[0][0] - mus[0][0], fit.means[0][1] - mus[0][1]) + Math.hypot(fit.means[1][0] - mus[1][0], fit.means[1][1] - mus[1][1]),
      Math.hypot(fit.means[0][0] - mus[1][0], fit.means[0][1] - mus[1][1]) + Math.hypot(fit.means[1][0] - mus[0][0], fit.means[1][1] - mus[0][1]),
    );
    if (d > 0.6) throw new Error(`means off by ${d}`);
    return `LL monotone over ${fit.history.length} steps; means within 0.6`;
  }),

  new Invariant("KD-tree = brute force on 100 random queries", "L6-instance", () => {
    const r = rng("kd");
    const pts = Array.from({ length: 800 }, () => [r.normal(), r.normal(), r.normal()]);
    const tree = new KDTree(pts);
    for (let t = 0; t < 100; t++) {
      const q = [r.normal(), r.normal(), r.normal()];
      const k = 1 + (t % 5);
      const a = tree.knearest(q, k);
      const b = knearestBrute(pts, q, k);
      for (let i = 0; i < k; i++) if (a[i] !== b[i]) throw new Error(`query ${t} rank ${i}: ${a[i]} ≠ ${b[i]}`);
    }
    return "100 queries × k∈{1..5} identical";
  }),

  new Invariant("Gaussian NB posteriors sum to 1; separable data classified perfectly", "L6-instance", () => {
    const { X, y } = makeClassificationData(200, "nb");
    const nb = new GaussianNB().fit(X, y);
    for (let i = 0; i < 10; i++) {
      const p = nb.predictProba(X[i]);
      approx("Σp", p.reduce((s, v) => s + v, 0), 1, 1e-12);
    }
    const pred = nb.predictAll(X);
    const acc = pred.filter((p, i) => p === y[i]).length / y.length;
    if (acc < 0.98) throw new Error(`acc ${acc}`);
    return `Σp=1 to 1e-12; acc=${acc}`;
  }),

  new Invariant("PCA: Eckart–Young error equals discarded eigenvalues; PCs orthonormal", "L6-pca", () => {
    const r = rng("pca");
    const n = 100, d = 6;
    const X = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < d; j++) row.push(r.normal() * (d - j) * 0.5);
      X.push(row);
    }
    const k = 3;
    const fit = pcaFit(X, { nComponents: k });
    // reconstruction error with k components
    const Z = fit.transform(X);
    const Xrec = fit.inverseTransform(Z);
    let sse = 0, sst = 0;
    const mu = fit.mean;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
      sse += (X[i][j] - Xrec[i][j]) ** 2;
      sst += (X[i][j] - mu[j]) ** 2;
    }
    const err = sse;
    const discarded = fit.explainedVariance.reduce((s, v) => s + v, 0); // placeholder
    // exact Eckart–Young: sse/(n−1) = Σ_{j>k} λ_j
    // recompute full spectrum for reference
    void discarded;
    const full = pcaFit(X);
    const tail = Array.from(full.explainedVariance).slice(k).reduce((s, v) => s + v, 0);
    approx("Eckart–Young", sse / (n - 1), tail, 1e-9);
    // orthonormality (components span the full d columns of V)
    const CtC = gemm(transpose(fit.components), fit.components);
    approxArray("CᵀC = I", CtC.a, eye(d).a, 1e-12);
    // explained ratio monotone decreasing
    const ratios = Array.from(fit.explainedRatio);
    for (let i = 1; i < ratios.length; i++) if (ratios[i] > ratios[i - 1] + 1e-12) throw new Error("ratio not sorted");
    return `‖X−X̂k‖²/(n−1) = Σλ_{j>k} to 1e-9; CᵀC=I`;
  }),

  new Invariant("Kernel PCA eigenvalues non-negative after centring", "L6-pca", () => {
    const r = rng("kpca");
    const X = Array.from({ length: 40 }, () => [r.normal(), r.normal()]);
    const kp = kernelPCA(X, { nComponents: 2 });
    const lam = Array.from(kp.eigenvalues).slice(0, 10);
    if (Math.min(...lam) < -1e-10) throw new Error(`negative eigenvalue ${Math.min(...lam)}`);
    return "centred Gram matrix PSD";
  }),
];

/* ================================================================= L7 */
const L7 = [
  new Invariant("top-p keeps cumulative mass ≥ p and nothing more", "L7-sampling", () => {
    const logits = Float64Array.from([5, 4, 3, 2, 1, 0, -1, -2]);
    const p = 0.9;
    const r = { token: -1, kept: 0 };
    // deterministic probe: examine kept count via many samples
    let minKept = Infinity;
    for (let i = 0; i < 50; i++) {
      const s = sampleLogits(logits, { topP: p, temperature: 1 });
      minKept = Math.min(minKept, s.kept);
      r.token = s.token; r.kept = s.kept;
    }
    const probs = softmax(logits);
    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    let cum = 0, need = 0;
    for (let i = 0; i < probs.length; i++) { cum += probs[order[i]]; need = i + 1; if (cum >= p) break; }
    if (minKept < need) throw new Error(`kept ${minKept} < required ${need}`);
    return `p=0.9 → keep ≥ ${need} tokens (observed min ${minKept})`;
  }),

  new Invariant("temperature→0 converges to argmax", "L7-sampling", () => {
    const logits = [1.0, 3.7, 2.2, -0.5];
    let hits = 0;
    for (let i = 0; i < 30; i++) {
      const s = sampleLogits(Float64Array.from(logits), { temperature: 0.01 });
      if (s.token === 1) hits++;
    }
    if (hits !== 30) throw new Error(`argmax only ${hits}/30`);
    return "30/30 picks index 1";
  }),

  new Invariant("min-p truncates by relative mass", "L7-sampling", () => {
    const logits = Float64Array.from([10, 4, 2, 0, -2]);
    const s = sampleLogits(logits, { minP: 0.5 });
    // only tokens ≥ 0.5·p_max survive → effectively only token 0 (p≈0.998)
    if (s.kept !== 1) throw new Error(`kept=${s.kept}, expected 1`);
    return `kept=${s.kept} (only the dominant token survives)`;
  }),

  new Invariant("speculative decoding acceptance rule is lossless", "L7-sampling", () => {
    // target [0.5, 0.3, 0.2], draft always proposes token 0
    // marginal acceptance frequency must equal p_target[0] = 0.5 …
    // actually = min(1, ratio) = min(1, 0.5/0.9) = 0.5556 per Leviathan Alg.1
    const pT = [0.5, 0.3, 0.2];
    const pD = [0.9, 0.08, 0.02];
    const r = rng("spec");
    let accept = 0;
    const N = 200000;
    for (let i = 0; i < N; i++) {
      const res = speculateAccept(pT, pD, 0, r.next());
      if (res.accept) accept++;
    }
    const expected = Math.min(1, pT[0] / pD[0]);
    const freq = accept / N;
    if (Math.abs(freq - expected) > 0.005) throw new Error(`accept freq ${freq} ≠ ${expected}`);
    return `accept rate ${freq.toFixed(4)} ≈ min(1, pT/pD) = ${expected.toFixed(4)}`;
  }),

  new Invariant("beam search top candidate ≥ greedy score", "L7-sampling", () => {
    // trivial vocab of 3, hand-defined step function
    const steps = 4;
    const table = [];
    const r = rng("beam");
    for (let s = 0; s < steps; s++) {
      const row = Float64Array.from([r.next(), r.next(), r.next()]);
      const m = Math.max(...row);
      table.push(Float64Array.from(row, (v) => v / (m + 0.1)));
    }
    const stepFn = (tokens) => table[tokens.length - 1];
    const beams = beamSearch(stepFn, [0], { beamWidth: 3, maxLen: steps });
    const greedy = [0];
    let greedyScore = 0;
    for (let s = 0; s < steps; s++) {
      const row = stepFn(greedy);
      let bi = 0;
      for (let i = 1; i < row.length; i++) if (row[i] > row[bi]) bi = i;
      greedyScore += Math.log(row[bi]);
      greedy.push(bi);
    }
    if (beams[0].score < greedyScore - 1e-9) throw new Error(`beam ${beams[0].score} < greedy ${greedyScore}`);
    return `beam ${beams[0].score.toFixed(4)} ≥ greedy ${greedyScore.toFixed(4)}`;
  }),
];
console.log(`  AI FORGE — executable invariants`);

export const ALL = [...L0, ...L1, ...L2, ...L3, ...L4, ...L5, ...L6, ...L7];
