/**
 * attention.js — Attention engines (L4).
 *
 * Zero-rewrite — the three modern attention regimes, in one file:
 *
 *  1. Full softmax attention (Vaswani 2017) — the reference semantics.
 *  2. FlashAttention (Dao et al. 2022): online softmax streaming over KV
 *     blocks; peak extra memory drops from O(T²) to O(T). The *numerical
 *     equivalence* of (1) and (2) to <1e-12 is an executable invariant —
 *     FlashAttention is an execution strategy, not a different function.
 *  3. GQA (Ainslie et al. 2023): Hq query heads share Hkv KV heads.
 *  4. KV-cache decoding: incremental attention where cache appending must
 *     reproduce full recompute bit-for-bit — the strongest invariant here.
 *  5. MLA (DeepSeek-V2 2024): keys/values live in a low-rank latent c_KV
 *     (dc ≪ H·dk); the cache stores only the latent, attention up-projects.
 *
 * @author 晨星
 */

import { softmax } from "../core/numeric.js";

/**
 * Full softmax attention (reference implementation).
 * Q: [Tq, Hq, dk] · K,V: [Tk, Hkv, dk] → out: [Tq, Hq, dv]
 * GQA: head h uses KV head floor(h / (Hq/Hkv)).
 */
export function attentionFull(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv = dk, causal = true, offset = 0, scale = null }) {
  const group = Hq / Hkv;
  const sc = scale ?? 1 / Math.sqrt(dk);
  const out = new Float64Array(Tq * Hq * dv);
  const logits = new Float64Array(Tk);
  const probs = new Float64Array(Tk);
  for (let h = 0; h < Hq; h++) {
    const kvh = Math.floor(h / group);
    for (let t = 0; t < Tq; t++) {
      const qOff = (t * Hq + h) * dk;
      // logits for this query row
      let nValid = 0;
      for (let s = 0; s < Tk; s++) {
        if (causal && s > t + offset) { logits[s] = -Infinity; continue; }
        let dot = 0;
        const kOff = (s * Hkv + kvh) * dk;
        for (let i = 0; i < dk; i++) dot += Q[qOff + i] * K[kOff + i];
        logits[s] = dot * sc;
        nValid++;
      }
      if (nValid === 0) continue;
      softmax(logits, probs);   // stable, handles -inf entries
      const oOff = (t * Hq + h) * dv;
      for (let s = 0; s < Tk; s++) {
        const p = probs[s];
        if (p === 0) continue;
        const vOff = (s * Hkv + kvh) * dv;
        for (let i = 0; i < dv; i++) out[oOff + i] += p * V[vOff + i];
      }
    }
  }
  return out;
}

/**
 * FlashAttention-style streaming attention: identical math, O(1) extra
 * memory per query row (online softmax: running max m, running sum l,
 * running output accumulator). Blocked over KV in chunks of `blockSize`.
 */
export function flashAttention(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv = dk, causal = true, offset = 0, scale = null, blockSize = 64 }) {
  const group = Hq / Hkv;
  const sc = scale ?? 1 / Math.sqrt(dk);
  const out = new Float64Array(Tq * Hq * dv);
  const acc = new Float64Array(dv);
  const sBuf = new Float64Array(blockSize);
  for (let h = 0; h < Hq; h++) {
    const kvh = Math.floor(h / group);
    for (let t = 0; t < Tq; t++) {
      const qOff = (t * Hq + h) * dk;
      let m = -Infinity, l = 0;
      acc.fill(0);
      const maxS = causal ? Math.min(Tk, t + offset + 1) : Tk;
      for (let bStart = 0; bStart < maxS; bStart += blockSize) {
        const bEnd = Math.min(bStart + blockSize, maxS);
        // block logits + block max
        let mBlock = -Infinity;
        for (let s = bStart; s < bEnd; s++) {
          let dot = 0;
          const kOff = (s * Hkv + kvh) * dk;
          for (let i = 0; i < dk; i++) dot += Q[qOff + i] * K[kOff + i];
          const sv = dot * sc;
          sBuf[s - bStart] = sv;
          if (sv > mBlock) mBlock = sv;
        }
        const mNew = Math.max(m, mBlock);
        const corr = m === -Infinity ? 0 : Math.exp(m - mNew);
        let lBlock = 0;
        for (let s = bStart; s < bEnd; s++) {
          const p = Math.exp(sBuf[s - bStart] - mNew);
          sBuf[s - bStart] = p;
          lBlock += p;
        }
        l = l * corr + lBlock;
        for (let i = 0; i < dv; i++) acc[i] *= corr;
        for (let s = bStart; s < bEnd; s++) {
          const p = sBuf[s - bStart];
          const vOff = (s * Hkv + kvh) * dv;
          for (let i = 0; i < dv; i++) acc[i] += p * V[vOff + i];
        }
        m = mNew;
      }
      const oOff = (t * Hq + h) * dv;
      if (l > 0) for (let i = 0; i < dv; i++) out[oOff + i] = acc[i] / l;
    }
  }
  return out;
}

/* ------------------------------------------------------------ KV cache */
/** Preallocated KV cache for incremental decoding (paged layout: [slot, Hkv, d]). */
export class KVCache {
  constructor(maxTokens, Hkv, dk, dv = dk) {
    this.maxTokens = maxTokens;
    this.Hkv = Hkv; this.dk = dk; this.dv = dv;
    this.K = new Float64Array(maxTokens * Hkv * dk);
    this.V = new Float64Array(maxTokens * Hkv * dv);
    this.len = 0;
  }
  /** Append prefill tokens. K: [T, Hkv, dk] */
  append(K, V, T) {
    const perK = this.Hkv * this.dk, perV = this.Hkv * this.dv;
    this.K.set(K.subarray(0, T * perK), this.len * perK);
    this.V.set(V.subarray(0, T * perV), this.len * perV);
    this.len += T;
    return this.len;
  }
  view() { return { K: this.K, V: this.V }; }
  reset() { this.len = 0; }
}

/**
 * One decode step: a single new query attends over the cache.
 * Mathematically identical to running full attention over all T+1 tokens
 * and keeping the last row — invariant-checked to 1e-12.
 */
export function decodeStep(q, Kc, Vc, cacheLen, { Hq, Hkv, dk, dv = dk, scale = null }) {
  const group = Hq / Hkv;
  const sc = scale ?? 1 / Math.sqrt(dk);
  const out = new Float64Array(Hq * dv);
  const logits = new Float64Array(cacheLen);
  const probs = new Float64Array(cacheLen);
  for (let h = 0; h < Hq; h++) {
    const kvh = Math.floor(h / group);
    for (let s = 0; s < cacheLen; s++) {
      let dot = 0;
      const kOff = (s * Hkv + kvh) * dk;
      const qOff = h * dk;
      for (let i = 0; i < dk; i++) dot += q[qOff + i] * Kc[kOff + i];
      logits[s] = dot * sc;
    }
    softmax(logits, probs);
    const oOff = h * dv;
    for (let s = 0; s < cacheLen; s++) {
      const p = probs[s];
      const vOff = (s * Hkv + kvh) * dv;
      for (let i = 0; i < dv; i++) out[oOff + i] += p * Vc[vOff + i];
    }
  }
  return out;
}

/* ---------------------------------------------------------------- MLA */
/**
 * Multi-head Latent Attention (DeepSeek-V2). KV compression:
 *   c_KV = W_DKV · x                 (dc-dim latent — the only cached part)
 *   k_h  = W_UK[h] · c_KV            (up-projection per head)
 *   v_h  = W_UV[h] · c_KV
 * Identity up-projections with dc = H·dk must reproduce exact MHA — invariant.
 */
export function mlaEncode(x, W_DKV, T, D, dc) {
  const c = new Float64Array(T * dc);
  linearRef(x, W_DKV, c, T, D, dc);
  return c;
}
export function mlaUpProject(cKV, W_UK, W_UV, T, dc, Hkv, dk, dv = dk) {
  const K = new Float64Array(T * Hkv * dk);
  const V = new Float64Array(T * Hkv * dv);
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < Hkv; h++) {
      const cOff = t * dc;
      const kOff = (t * Hkv + h) * dk;
      const vOff = (t * Hkv + h) * dv;
      const uk = h * dk * dc, uv = h * dv * dc;
      for (let i = 0; i < dk; i++) {
        let s = 0;
        for (let j = 0; j < dc; j++) s += W_UK[uk + i * dc + j] * cKV[cOff + j];
        K[kOff + i] = s;
      }
      for (let i = 0; i < dv; i++) {
        let s = 0;
        for (let j = 0; j < dc; j++) s += W_UV[uv + i * dc + j] * cKV[cOff + j];
        V[vOff + i] = s;
      }
    }
  }
  return { K, V };
}

/** Plain linear without bias (small, local). */
function linearRef(x, W, y, T, m, n) {
  for (let t = 0; t < T; t++) {
    const xo = t * m, yo = t * n;
    y.fill(0, yo, yo + n);
    for (let i = 0; i < m; i++) {
      const v = x[xo + i];
      if (v === 0) continue;
      const wo = i * n;
      for (let j = 0; j < n; j++) y[yo + j] += v * W[wo + j];
    }
  }
  return y;
}

/** Attention logit ⟨RoPE(q,m), RoPE(k,n)⟩ must depend only on m−n. */
export function ropePairDot(qk, kn, m, n, D, base = 10000.0) {
  let dot = 0;
  const half = D / 2;
  for (let i = 0; i < half; i++) {
    const thetaM = m * Math.pow(base, -2 * i / D);
    const thetaN = n * Math.pow(base, -2 * i / D);
    const cm = Math.cos(thetaM), sm = Math.sin(thetaM);
    const cn = Math.cos(thetaN), sn = Math.sin(thetaN);
    const q0 = qk[2 * i], q1 = qk[2 * i + 1];
    const k0 = kn[2 * i], k1 = kn[2 * i + 1];
    const qr0 = q0 * cm - q1 * sm, qr1 = q0 * sm + q1 * cm;
    const kr0 = k0 * cn - k1 * sn, kr1 = k0 * sn + k1 * cn;
    dot += qr0 * kr0 + qr1 * kr1;
  }
  return dot;
}

/* ------------------------------------------- 滑动窗口注意力 (Mistral-style) */
/**
 * Sliding-window attention (Beltagy et al. 2020 / Mistral 2023): each query t
 * attends only to keys inside a window of width `window`. Causal form restricts
 * to [t−window+1, t]; bidirectional form to [t−window+1, t+window−1].
 *
 * Invariant (tested): with window ≥ T this must equal causal `attentionFull`
 * bit-for-bit — the window imposes no restriction, so the two independent
 * implementations must agree to 1e-12.
 */
export function windowedAttention(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv = dk, window, causal = true }) {
  const group = Hq / Hkv;
  const sc = 1 / Math.sqrt(dk);
  const out = new Float64Array(Tq * Hq * dv);
  const logits = new Float64Array(Tk);
  const probs = new Float64Array(Tk);
  for (let h = 0; h < Hq; h++) {
    const kvh = Math.floor(h / group);
    for (let t = 0; t < Tq; t++) {
      let lo, hi;
      if (causal) { lo = Math.max(0, t - window + 1); hi = t; }
      else { lo = Math.max(0, t - window + 1); hi = Math.min(Tk - 1, t + window - 1); }
      if (hi < lo) continue;
      // windowed logits
      const qOff = (t * Hq + h) * dk;
      for (let s = lo; s <= hi; s++) {
        let dot = 0;
        const kOff = (s * Hkv + kvh) * dk;
        for (let i = 0; i < dk; i++) dot += Q[qOff + i] * K[kOff + i];
        logits[s] = dot * sc;
      }
      // stable softmax over [lo, hi]
      let m = -Infinity;
      for (let s = lo; s <= hi; s++) m = Math.max(m, logits[s]);
      let sum = 0;
      for (let s = lo; s <= hi; s++) { const e = Math.exp(logits[s] - m); probs[s] = e; sum += e; }
      for (let s = lo; s <= hi; s++) probs[s] /= sum;
      const oOff = (t * Hq + h) * dv;
      for (let s = lo; s <= hi; s++) {
        const p = probs[s];
        if (p === 0) continue;
        const vOff = (s * Hkv + kvh) * dv;
        for (let i = 0; i < dv; i++) out[oOff + i] += p * V[vOff + i];
      }
    }
  }
  return out;
}

/* ------------------------------- 线性注意力 (RetNet/Mamba 风格 kernel 注意力) */
/**
 * Linear (kernel-based) attention replaces softmax with a positive feature map
 * φ(·), giving O(T·d²) complexity instead of O(T²·d) and an O(1) recurrent
 * decode — the backbone of RetNet / linear-attention / state-space models.
 *
 *   S_t = Σ_{s≤t} φ(k_s) ⊗ v_s ,  z_t = Σ_{s≤t} φ(k_s)
 *   out_t = φ(q_t)ᵀ S_t / (φ(q_t)ᵀ z_t)
 *
 * This is a *rearrangement* of the softmax-free kernel attention, so the direct
 * form and the recurrent form must be bit-identical — the executable invariant.
 */
function _eluPlusOne(x) { return (x >= 0 ? x : Math.expm1(x)) + 1; }

/** Direct O(T²) reference form. */
export function linearAttentionDirect(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv = dk, causal = true, featureMap = null }) {
  const phi = featureMap || _eluPlusOne;
  const group = Hq / Hkv;
  const out = new Float64Array(Tq * Hq * dv);
  for (let h = 0; h < Hq; h++) {
    const kvh = Math.floor(h / group);
    for (let t = 0; t < Tq; t++) {
      const qOff = (t * Hq + h) * dk;
      let denom = 0;
      const num = new Float64Array(dv);
      const sEnd = causal ? t : Tk - 1;
      for (let s = 0; s <= sEnd; s++) {
        const kOff = (s * Hkv + kvh) * dk;
        const vOff = (s * Hkv + kvh) * dv;
        let sim = 0;
        for (let i = 0; i < dk; i++) sim += phi(Q[qOff + i]) * phi(K[kOff + i]);
        denom += sim;
        for (let j = 0; j < dv; j++) num[j] += sim * V[vOff + j];
      }
      const oOff = (t * Hq + h) * dv;
      if (denom > 0) for (let j = 0; j < dv; j++) out[oOff + j] = num[j] / denom;
    }
  }
  return out;
}

/** Recurrent O(T) streaming form — the O(1)-decode variant. */
export function linearAttentionRecurrent(Q, K, V, { Tq, Tk, Hq, Hkv, dk, dv = dk, causal = true, featureMap = null }) {
  const phi = featureMap || _eluPlusOne;
  const group = Hq / Hkv;
  const out = new Float64Array(Tq * Hq * dv);
  for (let h = 0; h < Hq; h++) {
    const kvh = Math.floor(h / group);
    const S = new Float64Array(dk * dv);   // Σ φ(k) ⊗ v
    const z = new Float64Array(dk);          // Σ φ(k)
    if (!causal) {
      // non-causal: accumulate the full KV context once
      for (let s = 0; s < Tk; s++) {
        const kOff = (s * Hkv + kvh) * dk;
        const vOff = (s * Hkv + kvh) * dv;
        for (let i = 0; i < dk; i++) {
          const ki = phi(K[kOff + i]);
          z[i] += ki;
          for (let j = 0; j < dv; j++) S[i * dv + j] += ki * V[vOff + j];
        }
      }
    }
    for (let t = 0; t < Tq; t++) {
      if (causal) {
        const kOff = (t * Hkv + kvh) * dk;
        const vOff = (t * Hkv + kvh) * dv;
        for (let i = 0; i < dk; i++) {
          const ki = phi(K[kOff + i]);
          z[i] += ki;
          for (let j = 0; j < dv; j++) S[i * dv + j] += ki * V[vOff + j];
        }
      }
      const qOff = (t * Hq + h) * dk;
      let denom = 0;
      for (let i = 0; i < dk; i++) denom += phi(Q[qOff + i]) * z[i];
      const oOff = (t * Hq + h) * dv;
      if (denom > 0) {
        for (let j = 0; j < dv; j++) {
          let s = 0;
          for (let i = 0; i < dk; i++) s += phi(Q[qOff + i]) * S[i * dv + j];
          out[oOff + j] = s / denom;
        }
      }
    }
  }
  return out;
}
