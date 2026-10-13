/**
 * linalg.js — BLAS/LAPACK-class kernels in pure JS (L0).
 *
 * Zero-rewrite policy: every routine is the canonical, battle-tested
 * formulation from the numerical-linear-algebra literature, ported to
 * cache-friendly typed arrays:
 *
 *  - GEMM: ikj loop order + cache blocking (Goto/BLIS micro-kernel idea)
 *  - Cholesky A = L Lᵀ  (used by Ridge / Kalman / GMM covariance)
 *  - Householder QR + least squares (LAPACK dgeqrf/dormqr idea)
 *  - Cyclic Jacobi eigen-decomposition (Rutishauser; gives *orthogonal*
 *    eigenvectors to machine precision — the reason PCA uses it)
 *  - One-sided Jacobi SVD (Demmel–Veselić; higher accuracy than Golub–Kahan
 *    for graded matrices)
 *  - LU with partial pivoting, determinant, inverse
 *  - Newton–Schulz quintic iteration (Jordan 2024 / Moonshot 2025) —
 *    the orthogonalisation core of the Muon optimizer
 *
 * Matrix representation: row-major Float64Array with {r, c}.
 *
 * @author 晨星
 */

export function zeros(r, c) { return { r, c, a: new Float64Array(r * c) }; }
export function fromRows(rows) {
  const r = rows.length, c = rows[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m.a[i * c + j] = rows[i][j];
  return m;
}
export function eye(n) {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m.a[i * n + i] = 1;
  return m;
}
export function clone(A) { return { r: A.r, c: A.c, a: Float64Array.from(A.a) }; }
export function get(A, i, j) { return A.a[i * A.c + j]; }
export function set(A, i, j, v) { A.a[i * A.c + j] = v; return A; }

export function transpose(A) {
  const T = zeros(A.c, A.r);
  const { r, c, a } = A; const ta = T.a;
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) ta[j * r + i] = a[i * c + j];
  return T;
}

/** C = A·B. `mode`: 'naive' (ijk), 'blocked' (ikj + tiling). */
export function gemm(A, B, mode = "blocked") {
  const m = A.r, k = A.c, n = B.c;
  if (B.r !== k) throw new Error(`gemm: shape mismatch ${m}x${k} · ${B.r}x${n}`);
  const C = zeros(m, n);
  const a = A.a, b = B.a, cc = C.a;
  if (mode === "naive") {
    for (let i = 0; i < m; i++) {
      const ai = i * k;
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) s += a[ai + p] * b[p * n + j];
        cc[i * n + j] = s;
      }
    }
    return C;
  }
  // blocked ikj: streaming access on B row and C row; tile k for cache
  const BK = 32;
  for (let kk = 0; kk < k; kk += BK) {
    const kEnd = Math.min(kk + BK, k);
    for (let i = 0; i < m; i++) {
      const ai = i * k, ci = i * n;
      for (let p = kk; p < kEnd; p++) {
        const av = a[ai + p];
        if (av === 0) continue;
        const bp = p * n;
        const ccRow = ci;
        for (let j = 0; j < n; j++) cc[ccRow + j] += av * b[bp + j];
      }
    }
  }
  return C;
}

/** C = Aᵀ·B (k = A.r). Avoids materialising the transpose. */
export function gemmAT(A, B) {
  const k = A.r, m = A.c, n = B.c;
  const C = zeros(m, n);
  const a = A.a, b = B.a, cc = C.a;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a[p * m + i] * b[p * n + j];
      cc[i * n + j] = s;
    }
  }
  return C;
}

/** y = A·x */
export function matVec(A, x, out) {
  const o = out || new Float64Array(A.r);
  for (let i = 0; i < A.r; i++) {
    let s = 0;
    const ai = i * A.c;
    for (let j = 0; j < A.c; j++) s += A.a[ai + j] * x[j];
    o[i] = s;
  }
  return o;
}
/** y = Aᵀ·x */
export function matVecT(A, x, out) {
  const o = out || new Float64Array(A.c);
  o.fill(0);
  for (let i = 0; i < A.r; i++) {
    const ai = i * A.c, xi = x[i];
    for (let j = 0; j < A.c; j++) o[j] += A.a[ai + j] * xi;
  }
  return o;
}

export const trace = (A) => { let s = 0; const n = Math.min(A.r, A.c); for (let i = 0; i < n; i++) s += A.a[i * A.c + i]; return s; };
export function frobenius(A) {
  let s = 0;
  for (let i = 0; i < A.a.length; i++) s += A.a[i] * A.a[i];
  return Math.sqrt(s);
}
export function maxAbs(A) { let m = 0; for (let i = 0; i < A.a.length; i++) { const v = Math.abs(A.a[i]); if (v > m) m = v; } return m; }

export function add(A, B) {
  const C = zeros(A.r, A.c);
  for (let i = 0; i < A.a.length; i++) C.a[i] = A.a[i] + B.a[i];
  return C;
}
export function sub(A, B) {
  const C = zeros(A.r, A.c);
  for (let i = 0; i < A.a.length; i++) C.a[i] = A.a[i] - B.a[i];
  return C;
}
export function scale(A, s) {
  const C = zeros(A.r, A.c);
  for (let i = 0; i < A.a.length; i++) C.a[i] = A.a[i] * s;
  return C;
}

/* ------------------------------------------------------------------ */
/* Cholesky: A = L Lᵀ (A symmetric positive definite)                  */
/* ------------------------------------------------------------------ */
export function cholesky(A) {
  const n = A.r; const L = zeros(n, n); const l = L.a, a = A.a;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j];
      for (let k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
      if (i === j) {
        if (s <= 0) throw new Error("cholesky: matrix is not positive definite");
        l[i * n + j] = Math.sqrt(s);
      } else {
        l[i * n + j] = s / l[j * n + j];
      }
    }
  }
  return L;
}
/** Solve A x = b given Cholesky factor L (A = L Lᵀ). */
export function choleskySolve(L, b) {
  const n = L.r; const y = new Float64Array(n); const l = L.a;
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= l[i * n + k] * y[k];
    y[i] = s / l[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
  }
  return x;
}

/* ------------------------------------------------------------------ */
/* Householder QR  →  least squares                                    */
/* ------------------------------------------------------------------ */
export function qr(A) {
  const m = A.r, n = A.c;
  const R = clone(A); const a = R.a;
  const vs = [];
  for (let k = 0; k < n; k++) {
    // norm of the sub-column
    let normx = 0;
    for (let i = k; i < m; i++) normx += a[i * n + k] * a[i * n + k];
    normx = Math.sqrt(normx);
    if (normx === 0) { vs.push(null); continue; }
    const alpha = a[k * n + k] > 0 ? -normx : normx;
    const v = new Float64Array(m);
    for (let i = k; i < m; i++) v[i] = a[i * n + k];
    v[k] -= alpha;
    let vnorm = 0;
    for (let i = k; i < m; i++) vnorm += v[i] * v[i];
    vnorm = Math.sqrt(vnorm);
    if (vnorm < 1e-300) { vs.push(null); continue; }
    for (let i = k; i < m; i++) v[i] /= vnorm;
    // apply H = I - 2 v vᵀ to columns k..n-1
    for (let j = k; j < n; j++) {
      let s = 0;
      for (let i = k; i < m; i++) s += v[i] * a[i * n + j];
      s *= 2;
      for (let i = k; i < m; i++) a[i * n + j] -= s * v[i];
    }
    vs.push(v);
  }
  // Build Q by applying the same reflectors to I
  const Q = eye(m); const q = Q.a;
  for (let k = n - 1; k >= 0; k--) {
    const v = vs[k]; if (!v) continue;
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let i = k; i < m; i++) s += v[i] * q[i * m + j];
      s *= 2;
      for (let i = k; i < m; i++) q[i * m + j] -= s * v[i];
    }
  }
  return { Q, R };
}

/** Least squares min‖Ax − b‖ via QR (numerically safer than normal eqs). */
export function lstsq(A, b) {
  const { Q, R } = qr(A);
  const n = A.c;
  const qt = matVecT(Q, b);
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = qt[i];
    for (let j = i + 1; j < n; j++) s -= R.a[i * n + j] * x[j];
    x[i] = s / R.a[i * n + i];
  }
  return x;
}

/* ------------------------------------------------------------------ */
/* Cyclic Jacobi eigen-decomposition of a symmetric matrix             */
/* ------------------------------------------------------------------ */
export function jacobiEigen(Ain, maxSweeps = 100, tol = 1e-30) {
  const n = Ain.r;
  const A = clone(Ain).a;
  const V = eye(n).a;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
    if (Math.sqrt(off) <= tol) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        // rotate A (rows/cols p,q)
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        // accumulate V
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => values[j] - values[i]);
  const sorted = new Float64Array(n);
  const Vec = zeros(n, n);
  order.forEach((oi, k) => { sorted[k] = values[oi]; for (let r = 0; r < n; r++) Vec.a[r * n + k] = V[r * n + oi]; });
  return { values: sorted, vectors: Vec };
}

/* ------------------------------------------------------------------ */
/* One-sided Jacobi SVD  →  A = U Σ Vᵀ                                 */
/* ------------------------------------------------------------------ */
export function svd(Ain, maxSweeps = 60) {
  const m = Ain.r, n = Ain.c;
  const X = clone(Ain).a;
  const V = eye(n).a;
  const eps = 2.220446049250313e-16;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let converged = true;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) {
          const xp = X[i * n + p], xq = X[i * n + q];
          alpha += xp * xp; beta += xq * xq; gamma += xp * xq;
        }
        if (Math.abs(gamma) > eps * Math.sqrt(alpha * beta) && gamma !== 0) {
          converged = false;
          const zeta = (beta - alpha) / (2 * gamma);
          const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let i = 0; i < m; i++) {
            const xp = X[i * n + p], xq = X[i * n + q];
            X[i * n + p] = c * xp - s * xq;
            X[i * n + q] = s * xp + c * xq;
          }
          for (let i = 0; i < n; i++) {
            const vp = V[i * n + p], vq = V[i * n + q];
            V[i * n + p] = c * vp - s * vq;
            V[i * n + q] = s * vp + c * vq;
          }
        }
      }
    }
    if (converged) break;
  }
  const svals = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < m; i++) s += X[i * n + j] * X[i * n + j];
    svals[j] = Math.sqrt(s);
  }
  const U = zeros(m, n);
  for (let j = 0; j < n; j++) {
    const sv = svals[j] > 1e-300 ? 1 / svals[j] : 0;
    for (let i = 0; i < m; i++) U.a[i * n + j] = X[i * n + j] * sv;
  }
  return { U, S: svals, V: { r: n, c: n, a: V } };
}

/** Truncated SVD reconstruction (Eckart–Young optimal rank-k). */
export function lowRankApprox(Ain, k) {
  const { U, S, V } = svd(Ain);
  const { r, c } = Ain;
  const out = zeros(r, c);
  for (let t = 0; t < Math.min(k, S.length); t++) {
    const s = S[t];
    for (let i = 0; i < r; i++) {
      const u = U.a[i * U.c + t];
      if (u === 0) continue;
      for (let j = 0; j < c; j++) out.a[i * c + j] += u * s * V.a[j * V.c + t];
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* LU with partial pivoting → det / inverse / solve                    */
/* ------------------------------------------------------------------ */
export function lu(Ain) {
  const n = Ain.r;
  const A = clone(Ain).a;
  const piv = new Int32Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  let sign = 1;
  for (let k = 0; k < n; k++) {
    let p = k, mx = Math.abs(A[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(A[i * n + k]); if (v > mx) { mx = v; p = i; } }
    if (mx === 0) continue;
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; }
      const t = piv[k]; piv[k] = piv[p]; piv[p] = t;
      sign = -sign;
    }
    const akk = A[k * n + k];
    for (let i = k + 1; i < n; i++) {
      const f = A[i * n + k] / akk;
      A[i * n + k] = f;
      for (let j = k + 1; j < n; j++) A[i * n + j] -= f * A[k * n + j];
    }
  }
  return { LU: { r: n, c: n, a: A }, piv, sign };
}
export function det(A) {
  const { LU: L, sign } = lu(A);
  let d = sign;
  for (let i = 0; i < A.r; i++) d *= L.a[i * A.c + i];
  return d;
}
export function solve(A, b) {
  const { LU: L, piv } = lu(A);
  const n = A.r;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = b[piv[i]];
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) x[i] -= L.a[i * n + j] * x[j];
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= L.a[i * n + j] * x[j];
    x[i] = s / L.a[i * n + i];
  }
  return x;
}
export function inv(A) {
  const n = A.r;
  const I = eye(n);
  const out = zeros(n, n);
  for (let j = 0; j < n; j++) {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = I.a[i * n + j];
    const x = solve(A, col);
    for (let i = 0; i < n; i++) out.a[i * n + j] = x[i];
  }
  return out;
}

/** Power iteration for the dominant eigenpair (Rayleigh quotient). */
export function powerIteration(A, iters = 500, tol = 1e-14) {
  const n = A.r;
  let v = new Float64Array(n).fill(1 / Math.sqrt(n));
  let lambda = 0;
  for (let t = 0; t < iters; t++) {
    const w = matVec(A, v);
    const nrm = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
    if (nrm === 0) break;
    for (let i = 0; i < n; i++) w[i] /= nrm;
    const nl = matVec(A, w).reduce((s, x, i) => s + x * w[i], 0);
    if (Math.abs(nl - lambda) < tol) { v = w; lambda = nl; break; }
    v = w; lambda = nl;
  }
  return { value: lambda, vector: v };
}

/* ------------------------------------------------------------------ */
/* Newton–Schulz orthogonalisation — the Muon core                     */
/* ------------------------------------------------------------------ */
/**
 * Quintic Newton–Schulz (Jordan 2024; coefficients tuned to maximise the
 * slope at 0 so 5 steps suffice even in bf16):
 *   G ← aG + b·G(GᵀG) + c·G(GᵀG)²,  (a,b,c) = (3.4445, −4.7750, 2.0315)
 * Converges to the closest semi-orthogonal matrix (polar factor).
 */
export function newtonSchulz(G, steps = 5, eps = 1e-12) {
  const { r, c } = G;
  let X = clone(G);
  const f = frobenius(X);
  if (f < eps) return X;                       // degenerate: nothing to orthogonalise
  for (let i = 0; i < X.a.length; i++) X.a[i] /= f;

  // Work with the tall side up so the Gram matrix is the smaller one.
  const flipped = r < c;
  let W = flipped ? transpose(X) : X;          // n×m, n >= m
  const n = W.r, m = W.c;
  const a = 3.4445, b = -4.7750, cc = 2.0315;

  for (let t = 0; t < steps; t++) {
    const A = gemmAT(W, W);                    // WᵀW  (m×m)
    const A2 = gemm(A, A, "naive");            // (WᵀW)² (m×m)
    // M = a·I + b·A + c·A²
    const M = zeros(m, m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) M.a[i * m + j] = b * A.a[i * m + j] + cc * A2.a[i * m + j];
      M.a[i * m + i] += a;
    }
    // W ← W · M   (equiv. aW + bW(WᵀW) + cW(WᵀW)²)
    const out = zeros(n, m);
    for (let i = 0; i < n; i++) {
      const wi = i * m, oi = i * m;
      for (let k = 0; k < m; k++) {
        const w = W.a[wi + k];
        if (w === 0) continue;
        const mk = k * m;
        for (let j = 0; j < m; j++) out.a[oi + j] += w * M.a[mk + j];
      }
    }
    W = out;
  }
  return flipped ? transpose(W) : W;
}

/** RMS-matched rescaling: make ‖Update‖_rms ≈ 0.2·sqrt(max(1, fanOut/fanIn)). */
export function muonRescale(O, fanIn, fanOut) {
  const scale = 0.2 * Math.sqrt(Math.max(1, fanOut / fanIn));
  const s = scale / (Math.sqrt(O.a.reduce((t, x) => t + x * x, 0) / O.a.length) + 1e-12);
  const out = zeros(O.r, O.c);
  for (let i = 0; i < O.a.length; i++) out.a[i] = O.a[i] * s;
  return out;
}

/** Frobenius inner product ⟨A,B⟩ = tr(AᵀB). */
export function frobeniusInner(A, B) {
  let s = 0;
  for (let i = 0; i < A.a.length; i++) s += A.a[i] * B.a[i];
  return s;
}

/** Symmetric matrix from a flat upper-triangle builder (used by tests). */
export function symmetricFrom(n, fn) {
  const M = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const v = fn(i, j); M.a[i * n + j] = v; M.a[j * n + i] = v; }
  return M;
}
