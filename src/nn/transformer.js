/**
 * transformer.js — Decoder-only LLM (L5).
 *
 * Architecture = the modern Llama/DeepSeek lineage, assembled from the
 * L3/L4 blocks: token embedding → N × [RMSNorm → RoPE-GQA-FlashAttention
 * → residual; RMSNorm → SwiGLU → residual] → RMSNorm → tied output head.
 * Pre-norm (Xiong 2020) for gradient stability; no biases anywhere.
 *
 * Supports:
 *  - prefill with causal FlashAttention,
 *  - incremental generation via KVCache (bit-exact with full recompute),
 *  - optional top-k MoE feed-forward (DeepSeek-style sparse FFN).
 *
 * @author 晨星
 */

import { rmsnorm, rope, swiglu, linear, moeRoute, initLinear } from "./modules.js";
import { flashAttention, attentionFull, KVCache, decodeStep } from "./attention.js";
import { softmax } from "../core/numeric.js";

export class TransformerLM {
  /**
   * @param cfg {vocab, dModel, nLayers, nHeads, nKVHeads, dFF, maxSeq, ropeBase, moeExperts, moeTopK}
   */
  constructor(cfg, seed = "forge-lm") {
    this.cfg = cfg;
    const { vocab, dModel: D, nLayers, nHeads: H, nKVHeads: Hkv, dFF: F, maxSeq } = cfg;
    const dk = D / H;
    this.dk = dk;
    const seedRng = () => seed;
    this.params = {
      embed: new Float64Array(vocab * D),
      layers: [],
      norm: new Float64Array(D).fill(1),
      head: null,  // tied to embed by default (weight tying, Press & Wolf 2017)
    };
    initLinear(this.params.embed, vocab, D, { std: 0.02, seed: seedRng() + ":embed" });
    for (let l = 0; l < nLayers; l++) {
      const layer = {
        wq: new Float64Array(D * D),
        wk: new Float64Array(D * Hkv * dk),
        wv: new Float64Array(D * Hkv * dk),
        wo: new Float64Array(D * D),
        attnNorm: new Float64Array(D).fill(1),
        ffnNorm: new Float64Array(D).fill(1),
      };
      initLinear(layer.wq, D, D, { seed: seedRng() + `:wq${l}` });
      initLinear(layer.wk, D, Hkv * dk, { seed: seedRng() + `:wk${l}` });
      initLinear(layer.wv, D, Hkv * dk, { seed: seedRng() + `:wv${l}` });
      // output projection scaled per GPT-2 residual convention (1/√(2L))
      initLinear(layer.wo, D, D, { seed: seedRng() + `:wo${l}`, scaleStd: 0.02 / Math.sqrt(2 * nLayers) });
      if (cfg.moeExperts) {
        const E = cfg.moeExperts;
        layer.router = new Float64Array(D * E);
        initLinear(layer.router, D, E, { seed: seedRng() + `:rtr${l}` });
        layer.w1 = []; layer.w3 = []; layer.w2 = [];
        for (let e = 0; e < E; e++) {
          const a = new Float64Array(D * F), c = new Float64Array(D * F), b = new Float64Array(F * D);
          initLinear(a, D, F, { seed: seedRng() + `:w1_${l}_${e}` });
          initLinear(c, D, F, { seed: seedRng() + `:w3_${l}_${e}` });
          initLinear(b, F, D, { seed: seedRng() + `:w2_${l}_${e}`, scaleStd: 0.02 / Math.sqrt(2 * nLayers) });
          layer.w1.push(a); layer.w3.push(c); layer.w2.push(b);
        }
      } else {
        layer.w1 = new Float64Array(D * F);
        layer.w3 = new Float64Array(D * F);
        layer.w2 = new Float64Array(F * D);
        initLinear(layer.w1, D, F, { seed: seedRng() + `:w1${l}` });
        initLinear(layer.w3, D, F, { seed: seedRng() + `:w3${l}` });
        initLinear(layer.w2, F, D, { seed: seedRng() + `:w2${l}`, scaleStd: 0.02 / Math.sqrt(2 * nLayers) });
      }
      this.params.layers.push(layer);
    }
  }

  /** Embed tokens → [T, D] */
  embedTokens(tokens) {
    const D = this.cfg.dModel;
    const x = new Float64Array(tokens.length * D);
    for (let t = 0; t < tokens.length; t++) {
      x.set(this.params.embed.subarray(tokens[t] * D, tokens[t] * D + D), t * D);
    }
    return x;
  }

  /** One layer forward. Returns {x, K, V} (post-RoPE keys/values [T, Hkv, dk]). */
  layerForward(l, x, positions, useFullAttention = false) {
    const { dModel: D, nHeads: H, nKVHeads: Hkv, ropeBase } = this.cfg;
    const dk = this.dk;
    const T = positions.length;
    const L = this.params.layers[l];

    // --- attention sub-block (pre-norm) ---
    const normed = rmsnorm(x, L.attnNorm);
    const q = new Float64Array(T * D);
    const k = new Float64Array(T * Hkv * dk);
    const v = new Float64Array(T * Hkv * dk);
    linear(normed, L.wq, q, T, D, D);
    linear(normed, L.wk, k, T, D, Hkv * dk);
    linear(normed, L.wv, v, T, D, Hkv * dk);
    const qR = rope(q, T, D, positions, ropeBase);
    const kR = rope(k, T, Hkv * dk, positions, ropeBase);
    const attnOpts = { Tq: T, Tk: T, Hq: H, Hkv, dk, dv: dk, causal: true };
    const attn = (useFullAttention ? attentionFull : flashAttention)(qR, kR, v, attnOpts);
    const attnOut = new Float64Array(T * D);
    linear(attn, L.wo, attnOut, T, D, D);
    const x1 = new Float64Array(T * D);
    for (let i = 0; i < x1.length; i++) x1[i] = x[i] + attnOut[i];

    // --- feed-forward sub-block (pre-norm) ---
    const normed2 = rmsnorm(x1, L.ffnNorm);
    let ffn;
    if (L.router) {
      const { expert, weight } = moeRoute(normed2, L.router, T, D, this.cfg.moeExperts, this.cfg.moeTopK ?? 2);
      ffn = new Float64Array(T * D);
      const F = this.cfg.dFF;
      const acc = new Float64Array(T * D);
      for (let e = 0; e < this.cfg.moeExperts; e++) {
        // gather tokens routed to expert e (weighted)
        let any = false;
        const idx = [];
        const wts = [];
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < (this.cfg.moeTopK ?? 2); j++) {
            if (expert[t * (this.cfg.moeTopK ?? 2) + j] === e) { idx.push(t); wts.push(weight[t * (this.cfg.moeTopK ?? 2) + j]); }
          }
        }
        if (!idx.length) continue;
        any = true;
        const sub = new Float64Array(idx.length * D);
        idx.forEach((t, i) => sub.set(normed2.subarray(t * D, t * D + D), i * D));
        const y = swiglu(sub, L.w1[e], L.w3[e], L.w2[e], idx.length, D, F);
        idx.forEach((t, i) => {
          const w = wts[i];
          for (let d2 = 0; d2 < D; d2++) acc[t * D + d2] += w * y[i * D + d2];
        });
        void any;
      }
      ffn = acc;
    } else {
      ffn = swiglu(normed2, L.w1, L.w3, L.w2, T, D, this.cfg.dFF);
    }
    const x2 = new Float64Array(T * D);
    for (let i = 0; i < x2.length; i++) x2[i] = x1[i] + ffn[i];
    return { x: x2, K: kR, V: v };
  }

  /**
   * One decode step for a single token at position `pos`, attending over
   * the KV-cache. Mathematically identical to the corresponding causal row
   * of a full forward (invariant-checked in L4/L5).
   */
  layerDecode(l, xVec, pos, cache) {
    const { dModel: D, nHeads: H, nKVHeads: Hkv, ropeBase } = this.cfg;
    const dk = this.dk;
    const L = this.params.layers[l];
    const normed = rmsnorm(xVec, L.attnNorm);
    const q = new Float64Array(D), k = new Float64Array(Hkv * dk), v = new Float64Array(Hkv * dk);
    linear(normed, L.wq, q, 1, D, D);
    linear(normed, L.wk, k, 1, D, Hkv * dk);
    linear(normed, L.wv, v, 1, D, Hkv * dk);
    const qR = rope(q, 1, D, [pos], ropeBase);
    const kR = rope(k, 1, Hkv * dk, [pos], ropeBase);
    cache.append(kR, v, 1);
    const attn = decodeStep(qR, cache.K, cache.V, cache.len, { Hq: H, Hkv, dk, dv: dk });
    const attnOut = new Float64Array(D);
    linear(attn, L.wo, attnOut, 1, D, D);
    const x1 = new Float64Array(D);
    for (let i = 0; i < D; i++) x1[i] = xVec[i] + attnOut[i];
    const normed2 = rmsnorm(x1, L.ffnNorm);
    const ffn = swiglu(normed2, L.w1, L.w3, L.w2, 1, D, this.cfg.dFF);
    const x2 = new Float64Array(D);
    for (let i = 0; i < D; i++) x2[i] = x1[i] + ffn[i];
    return x2;
  }

  /** Full forward → logits [T, vocab]. useFullAttention=true uses the reference path. */
  forward(tokens, { full = false } = {}) {
    const D = this.cfg.dModel;
    const T = tokens.length;
    let x = this.embedTokens(tokens);
    const positions = Array.from({ length: T }, (_, i) => i);
    for (let l = 0; l < this.cfg.nLayers; l++) {
      const r = this.layerForward(l, x, positions, full);
      x = r.x;
    }
    const normed = rmsnorm(x, this.params.norm);
    const logits = new Float64Array(T * this.cfg.vocab);
    linear(normed, this.params.embed, logits, T, D, this.cfg.vocab); // tied head
    return logits;
  }

  /**
   * Incremental generation with KV-cache. Returns generated token ids.
   * Invariant: every step equals the corresponding row of a full recompute.
   */
  * generate(promptTokens, { maxNew = 16, temperature = 1.0, topK = 0, topP = 0, rngStream = null, greedy = true }) {
    const cache = [];
    for (let l = 0; l < this.cfg.nLayers; l++) {
      cache.push(new KVCache(this.cfg.maxSeq, this.cfg.nKVHeads, this.dk));
    }
    const D = this.cfg.dModel;
    const tokens = [...promptTokens];
    let last = null;
    // prefill
    let x = this.embedTokens(tokens);
    let positions = tokens.map((_, i) => i);
    for (let l = 0; l < this.cfg.nLayers; l++) {
      const r = this.layerForward(l, x, positions, false);
      cache[l].append(r.K, r.V, tokens.length);
      x = r.x;
    }
    last = rmsnorm(x.subarray((tokens.length - 1) * D, tokens.length * D), this.params.norm);
    let next = this.sampleFrom(last, { temperature, topK, topP, rngStream, greedy });
    for (let step = 0; step < maxNew; step++) {
      tokens.push(next);
      yield next;
      if (step === maxNew - 1) break;
      // decode one token: each layer attends over its KV-cache
      const pos = tokens.length - 1;
      let xi = this.embedTokens([next]).subarray(0, D);
      for (let l = 0; l < this.cfg.nLayers; l++) {
        xi = this.layerDecode(l, xi, pos, cache[l]);
      }
      const normed = rmsnorm(xi, this.params.norm);
      next = this.sampleFrom(normed, { temperature, topK, topP, rngStream, greedy });
    }
    return tokens;
  }

  sampleFrom(normedRow, { temperature, topK, topP, rngStream, greedy }) {
    const logits = new Float64Array(this.cfg.vocab);
    linear(normedRow, this.params.embed, logits, 1, this.cfg.dModel, this.cfg.vocab);
    if (temperature !== 1) for (let i = 0; i < logits.length; i++) logits[i] /= temperature;
    const probs = softmax(logits);
    if (greedy) {
      let bi = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bi]) bi = i;
      return bi;
    }
    // top-k / top-p filtering then sample
    const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    let keep = idx.length;
    if (topK > 0) keep = Math.min(keep, topK);
    if (topP > 0) {
      let cum = 0;
      keep = Math.min(keep, idx.length);
      for (let i = 0; i < idx.length; i++) { cum += probs[idx[i]]; if (cum >= topP) { keep = Math.min(keep, i + 1); break; } }
    }
    let cum = 0;
    const r = rngStream ? rngStream.next() : Math.random();
    for (let i = 0; i < keep; i++) {
      cum += probs[idx[i]];
      if (r < cum) return idx[i];
    }
    return idx[keep - 1];
  }
}
