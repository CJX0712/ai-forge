/**
 * run.mjs — CLI entry: run the executable spec.
 *   node test/run.mjs [filter]
 * @author 晨星
 */
import { ALL } from "./invariants.mjs";
import { InvariantRunner } from "../src/index.js";

const filter = process.argv[2] || null;
const runner = new InvariantRunner({ verbose: true });
runner.register(...ALL);
const { results, pass, fail, total } = await runner.run(filter);

console.log("\n" + "═".repeat(64));
console.log("  AI FORGE — executable invariants");
console.log(`  ✅ ${pass} passed   ${fail ? "❌ " + fail + " failed" : "no failures"}   ${(total / 1000).toFixed(2)}s`);
console.log("═".repeat(64));
if (fail) {
  console.log("\nFailed invariants:");
  for (const r of results.filter((x) => x.status !== "PASS")) {
    console.log(`  ❌ [${r.layer}] ${r.name}${r.error ? " — " + r.error : ""}`);
  }
  process.exit(1);
}
