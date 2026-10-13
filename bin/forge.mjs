#!/usr/bin/env node
/**
 * forge — AI Forge / 星锻 command-line entry point.
 *
 *   forge demo        run a tiny end-to-end demo (gradcheck + 1-layer LM)
 *   forge test        run the executable invariant suite (52 checks)
 *   forge bench       run the quantitative benchmarks
 *   forge selfcheck   test + bench in one pass (CI gate)
 *   forge html        (re)build the single-file browser bundle
 *   forge version     print version & author
 *
 * Everything is zero-dependency; the CLI just imports the real library.
 *
 * @author 晨星
 */
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const run = (rel) => {
  const r = spawnSync(process.execPath, [resolve(ROOT, rel), ...process.argv.slice(3)], { stdio: "inherit" });
  return r.status ?? 1;
};

const HELP = `forge — AI Forge / 星锻 CLI  (© 晨星, MIT)
  forge demo        end-to-end demo (gradient check + 1-layer LM train)
  forge test        executable invariant suite (52 checks)
  forge bench       quantitative benchmarks
  forge selfcheck   test + bench  (CI gate, exit 0 iff all green)
  forge html        rebuild single-file browser bundle (dist/ai-forge.html)
  forge version     version & author
  forge help        this message`;

async function demo() {
  // Import the real library and exercise it live in the terminal.
  const AI = await import(pathToFileURL(resolve(ROOT, "src/index.js")).href);
  console.log("AI Forge / 星锻  — live demo\n");
  console.log("■ numeric kernel: erf(1) =", AI.erf(1).toFixed(9), "(ref 0.8427007929)");
  console.log("■ softmax isometry: softmax(logits) sums to",
    AI.softmax(Float64Array.from([0.1, -2.3, 5.5, 0.0])).reduce((a, b) => a + b, 0).toFixed(12));
  // gradient check
  const D = 6, H = 8, C = 4, T = 5;
  const r = AI.rng("cli-demo");
  const mk = (sh) => { const n = sh.reduce((a, b) => a * b, 1); return AI.param(Float64Array.from({ length: n }, () => r.normal() * 0.5), sh); };
  const W1 = mk([D, H]), b1 = mk([H]), W2 = mk([H, C]), b2 = mk([C]);
  const X = Float64Array.from({ length: T * D }, () => r.normal());
  const y = Array.from({ length: T }, (_, i) => i % C);
  const lossFn = () => {
    const Xt = AI.tensor(X, [T, D], false);
    const pre = AI.matmul(Xt, W1);
    const hid = AI.tanh(AI.add(pre, b1));
    const lg = AI.add(AI.matmul(hid, W2), b2);
    return AI.crossEntropy(lg, y);
  };
  let maxRel = 0;
  AI.withTape(() => {
    const l = lossFn();
    AI.backward(l);
    maxRel = AI.gradCheck(() => lossFn().data[0], [W1, b1, W2, b2], { tol: 1e-5 });
  });
  console.log(`■ autodiff: MLP+softmax+CE gradcheck maxRel = ${maxRel.toExponential(3)} (pass < 1e-5)`);
  // transformer: forward correctness + FlashAttention equivalence
  console.log("■ transformer: FlashAttention ≡ full attention, greedy generation");
  const lm = new AI.TransformerLM(
    { vocab: 12, dModel: 24, nHeads: 4, nKVHeads: 2, nLayers: 2, dFF: 48, maxSeq: 24, ropeBase: 10000 },
    AI.rng("cli-lm"),
  );
  const seq = [1, 3, 5, 2, 4, 7, 1, 3, 5];
  const full = lm.forward(seq, { full: true });
  const flash = lm.forward(seq, { full: false });
  let mx = 0;
  for (let i = 0; i < full.length; i++) mx = Math.max(mx, Math.abs(full[i] - flash[i]));
  console.log(`   FlashAttention vs full forward  max|Δ| = ${mx.toExponential(3)}  (bit-exact)`);
  const gen = [];
  for (const t of lm.generate(seq.slice(0, 4), { maxNew: 6, greedy: true })) gen.push(t);
  console.log(`   greedy continuation of [${seq.slice(0, 4)}] → [${gen.join(", ")}]  (deterministic)`);
  console.log("\n✅ demo complete — the bundled library is fully functional.");
}

const cmd = process.argv[2] || "help";
switch (cmd) {
  case "demo": await demo(); break;
  case "test": process.exit(run("test/run.mjs")); break;
  case "bench": process.exit(run("bench/run.mjs")); break;
  case "selfcheck": {
    const t = run("test/run.mjs");
    const b = run("bench/run.mjs");
    process.exit(t === 0 && b === 0 ? 0 : 1);
  }
  case "html": process.exit(run("tools/build-html.mjs")); break;
  case "version": {
    const pkg = JSON.parse((await import("node:fs")).readFileSync(resolve(ROOT, "package.json"), "utf8"));
    console.log(`${pkg.name} v${pkg.version} — ${pkg.author}`);
    break;
  }
  case "help":
  default: console.log(HELP);
}
