/**
 * sampling.js — Token sampling strategies (L7).
 *
 * Canonical references:
 *  - temperature / top-k / nucleus (top-p) sampling (Holtzman 2019),
 *    plus min-p (Nguyen 2024) as used in modern inference stacks.
 *  - Speculative decoding (Leviathan 2023 / Chen 2023): draft model
 *    proposes k tokens, verifier accepts/rejects via the residual rule
 *    min(1, p_target/p_draft) — mathematically lossless w.r.t. the target.
 *
 * @author 晨星
 */

import { softmax } from "../core/numeric.js";

/**
 * Filter + sample from logits. Returns {token, prob, kept}.
 * Order of operations: temperature → top-k → top-p → min-p → multinomial.
 */
export function sampleLogits(logits, { temperature = 1, topK = 0, topP = 0, minP = 0, rng = null } = {}) {
  const V = logits.length;
  const l = Float64Array.from(logits);
  if (temperature !== 1 && temperature > 0) for (let i = 0; i < V; i++) l[i] /= temperature;
  const probs = softmax(l);
  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);

  let cutoff = V;
  if (topK > 0) cutoff = Math.min(cutoff, topK);
  if (topP > 0) {
    let cum = 0;
    for (let i = 0; i < V; i++) { cum += probs[order[i]]; if (cum >= topP) { cutoff = Math.min(cutoff, i + 1); break; } }
  }
  if (minP > 0) {
    const pMax = probs[order[0]];
    let i = 0;
    while (i < cutoff && probs[order[i]] >= minP * pMax) i++;
    cutoff = Math.max(1, i);
  }
  // renormalise kept region
  const kept = new Float64Array(cutoff);
  let total = 0;
  for (let i = 0; i < cutoff; i++) { kept[i] = probs[order[i]]; total += kept[i]; }
  for (let i = 0; i < cutoff; i++) kept[i] /= total;
  // multinomial draw
  const r = rng ? rng() : Math.random();
  let cum = 0;
  for (let i = 0; i < cutoff; i++) { cum += kept[i]; if (r < cum) return { token: order[i], prob: kept[i], kept: cutoff }; }
  return { token: order[cutoff - 1], prob: kept[cutoff - 1], kept: cutoff };
}

/**
 * Greedy beam search (log-prob scoring). Returns beam candidates sorted by
 * cumulative log-probability.
 */
export function beamSearch(stepFn, startTokens, { beamWidth = 4, maxLen = 16, eos = null } = {}) {
  let beams = [{ tokens: [...startTokens], score: 0 }];
  const completed = [];
  for (let step = 0; step < maxLen; step++) {
    const candidates = [];
    for (const beam of beams) {
      const dist = stepFn(beam.tokens);         // Float64Array over vocab
      const logp = Float64Array.from(dist, Math.log);
      const order = Array.from(logp.keys()).sort((a, b) => logp[b] - logp[a]);
      for (let i = 0; i < Math.min(beamWidth, order.length); i++) {
        const tok = order[i];
        if (eos !== null && tok === eos) { completed.push(beam); continue; }
        candidates.push({ tokens: [...beam.tokens, tok], score: beam.score + logp[tok] });
      }
    }
    if (!candidates.length) break;
    candidates.sort((a, b) => b.score - a.score);
    beams = candidates.slice(0, beamWidth);
    if (completed.length >= beamWidth) break;
  }
  const all = [...completed, ...beams];
  all.sort((a, b) => b.score - a.score);
  return all.slice(0, beamWidth);
}

/**
 * Speculative decoding verifier. Given the target and draft distributions
 * over one position and the drafted token, decide accept/reject by the
 * residual-sampling rule (Leviathan 2023, Alg. 1) — provably lossless.
 */
export function speculateAccept(pTarget, pDraft, draftedToken, u01 = Math.random()) {
  const ratio = pTarget[draftedToken] / Math.max(1e-300, pDraft[draftedToken]);
  if (u01 <= Math.min(1, ratio)) return { accept: true, token: draftedToken };
  // residual: sample from normalised max(0, p_target − p_draft)
  const resid = Float64Array.from(pTarget, (v, i) => Math.max(0, v - pDraft[i]));
  let total = 0;
  for (const v of resid) total += v;
  if (total <= 0) return { accept: false, token: -1 };
  let cum = 0;
  for (let i = 0; i < resid.length; i++) { cum += resid[i] / total; if (u01 < cum) return { accept: false, token: i }; }
  return { accept: false, token: resid.length - 1 };
}

/** Perplexity of a language model over held-out logits. */
export function perplexity(logProbs) {
  let s = 0;
  for (const lp of logProbs) s += lp;
  return Math.exp(-s / logProbs.length);
}
