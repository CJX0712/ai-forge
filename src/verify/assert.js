/**
 * assert.js — Invariant verification framework (L8).
 *
 * The philosophy: an AI library you cannot *prove* is a liability. Every
 * module ships with executable invariants — mathematical statements that
 * must hold to machine precision — checked here with meaningful tolerances
 * derived from the analysis in each module's header.
 *
 * @author 晨星
 */

export class Invariant {
  constructor(name, layer, fn, { tolerance = null, doc = "" } = {}) {
    this.name = name; this.layer = layer; this.fn = fn; this.doc = doc; this.tolerance = tolerance;
  }
}

export class InvariantRunner {
  constructor({ verbose = true } = {}) {
    this.invariants = [];
    this.verbose = verbose;
  }
  register(...invariants) { this.invariants.push(...invariants); return this; }

  async run(filter = null) {
    const t0 = performance.now();
    const results = [];
    let pass = 0, fail = 0;
    for (const inv of this.invariants) {
      if (filter && !inv.name.includes(filter) && !inv.layer.includes(filter)) continue;
      const ts = performance.now();
      let status = "PASS", detail = "", error = null;
      try {
        const r = inv.fn();
        if (r === false) status = "FAIL";
        else if (typeof r === "string") { detail = r; }
        else if (r && typeof r === "object") {
          if (r.ok === false) { status = "FAIL"; }
          detail = r.detail || "";
        }
      } catch (e) {
        status = "ERROR"; error = e.message;
      }
      const ms = performance.now() - ts;
      if (status === "PASS") pass++; else fail++;
      results.push({ name: inv.name, layer: inv.layer, status, detail, error, ms });
      if (this.verbose) {
        const icon = status === "PASS" ? "✅" : status === "ERROR" ? "💥" : "❌";
        console.log(`${icon} [${inv.layer}] ${inv.name}  ${detail ? "(" + detail + ")" : ""}  ${ms.toFixed(1)}ms${error ? " — " + error : ""}`);
      }
    }
    const total = performance.now() - t0;
    return { results, pass, fail, total };
  }
}

/* -------------------------------------------------------- assertions */
/**
 * Mixed absolute/relative check (numpy.allclose semantics):
 *   |a − b| ≤ tol · max(1, |a|, |b|)
 * so values near zero are judged by absolute error and large values by
 * relative error — the only scale-free definition that treats "≈0" fairly.
 */
export function approx(name, a, b, tol = 1e-10) {
  const d = Math.abs(a - b);
  const scale = Math.max(1, Math.abs(a), Math.abs(b));
  const mixed = d / scale;
  if (mixed > tol) throw new Error(`${name}: |${a} − ${b}| = ${d} (mixed ${mixed.toExponential(2)} > ${tol})`);
  return `${d.toExponential(2)} ≤ ${tol}`;
}

/** Max mixed error over arrays. */
export function approxArray(name, A, B, tol = 1e-10) {
  let maxE = 0, badIdx = -1;
  const n = Math.min(A.length, B.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(A[i] - B[i]);
    const scale = Math.max(1, Math.abs(A[i]), Math.abs(B[i]));
    const mixed = d / scale;
    if (mixed > maxE) { maxE = mixed; badIdx = i; }
  }
  if (maxE > tol) throw new Error(`${name}: max err ${maxE.toExponential(2)} @${badIdx} > ${tol}`);
  return `maxErr=${maxE.toExponential(2)} ≤ ${tol}`;
}

export function assertTrue(name, cond) {
  if (!cond) throw new Error(`${name}: expected true`);
  return "ok";
}

export function assertMonotone(name, seq, direction = "decreasing") {
  for (let i = 1; i < seq.length; i++) {
    const bad = direction === "decreasing" ? seq[i] > seq[i - 1] + 1e-9 : seq[i] < seq[i - 1] - 1e-9;
    if (bad) throw new Error(`${name}: ${direction} violated at step ${i} (${seq[i - 1]} → ${seq[i]})`);
  }
  return `${seq.length} steps monotone ${direction}`;
}

/** Deterministic timing helper for benchmarks. */
export function timeIt(fn, { repeat = 3, warmup = 1 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const times = [];
  for (let i = 0; i < repeat; i++) {
    const t = performance.now();
    fn();
    times.push(performance.now() - t);
  }
  return { min: Math.min(...times), median: times.sort((a, b) => a - b)[Math.floor(times.length / 2)], all: times };
}
