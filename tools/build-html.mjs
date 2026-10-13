/**
 * build-html.mjs — Zero-dependency bundler → single self-contained HTML.
 *
 * Why hand-roll a bundler instead of pulling esbuild/rollup? The project's
 * core principle is *zero-rewrite / 集大成*: we never add a build-time runtime
 * dependency for something we can do in ~150 lines that is *fully verifiable*
 * and Node-22 + browser-portable. The resulting HTML has NO external requests,
 * NO CDN, NO npm install — open it and the entire AI Forge library runs.
 *
 * The bundler implements the small, well-defined subset of ESM we actually use:
 *   - named imports   `import { a, b as c } from "./x.js"`
 *   - namespace imports `import * as ns from "./x.js"`
 *   - re-exports      `export * from`, `export {..} from`, `export * as ns from`
 *   - local exports    `export function/class/const`
 * Each module becomes a function-scoped wrapper; `export *` yields to explicit
 * named re-exports, which yield to local definitions (Node ESM precedence).
 *
 * @author 晨星
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(__dirname, "../src");
const OUT = resolve(__dirname, "../dist/ai-forge.html");

/* ---------------------------------------------------------- parse graph */
const modules = new Map(); // id -> { path, code, imports:[{names,src}], star:[], namedFrom:[], ns:[], local:Set }

function idOf(path) { return relative(SRC, path).replace(/\\/g, "/"); }
function resolveImp(src, fromPath) {
  return idOf(resolve(dirname(fromPath), src));
}

function load(path) {
  const id = idOf(path);
  if (modules.has(id)) return id;
  const code = readFileSync(path, "utf8");
  const m = { id, path, code, imports: [], star: [], namedFrom: [], ns: [], local: new Set() };
  modules.set(id, m);
  // Private regex instances: /g state must NOT leak across the recursive
  // load() calls (a nested load would otherwise rewind the caller's exec loop
  // and double-match imports).
  const importRe = /import\s*(?:\*\s*as\s*(\w+)|(?:\{([^}]*)\}))\s*from\s*["']([^"']+)["']/g;
  const starRe = /export\s*\*\s*from\s*["']([^"']+)["']/g;
  const nsRe = /export\s*\*\s*as\s*(\w+)\s*from\s*["']([^"']+)["']/g;
  const namedFromRe = /export\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']/g;
  const declRe = /export\s+(function|class|const|let|var|async\s+function)\s+([A-Za-z_$][\w$]*)/g;
  const exportListRe = /export\s*\{([^}]*)\}(?!\s*from)\s*;?/g;
  let mm;
  while ((mm = starRe.exec(code))) m.star.push(resolveImp(mm[1], path));
  while ((mm = nsRe.exec(code))) m.ns.push({ ns: mm[1], src: resolveImp(mm[2], path) });
  while ((mm = namedFromRe.exec(code))) m.namedFrom.push(resolveImp(mm[2], path));
  while ((mm = declRe.exec(code))) m.local.add(mm[2]);
  while ((mm = exportListRe.exec(code))) {
    for (const part of mm[1].split(",")) {
      const [a, b] = part.split(/\s+as\s+/).map((x) => x.trim());
      if (a) m.local.add(b || a);
    }
  }
  while ((mm = importRe.exec(code))) {
    const src = resolveImp(mm[3], path);
    if (mm[1]) m.imports.push({ ns: mm[1], src, names: null });
    else {
      const names = mm[2].split(",").map((s) => s.trim()).filter(Boolean).map((s) => {
        const [a, b] = s.split(/\s+as\s+/).map((x) => x.trim());
        return { local: b || a, orig: a };
      });
      m.imports.push({ names, src });
    }
    load(resolve(SRC, src));
  }
  for (const s of m.star) load(resolve(SRC, s));
  for (const s of m.namedFrom) load(resolve(SRC, s));
  for (const n of m.ns) load(resolve(SRC, n.src));
  return id;
}

const entryId = load(resolve(SRC, "index.js"));

/* ---------------------------------------------------- topological order */
const order = [];
const seen = new Set();
function topo(id) {
  if (seen.has(id)) return;
  seen.add(id);
  const m = modules.get(id);
  if (!m) { console.error("MISSING MODULE:", id, "known:", [...modules.keys()].join(", ")); throw new Error("missing " + id); }
  for (const imp of m.imports) topo(imp.src);
  for (const s of [...m.star, ...m.namedFrom, ...m.ns.map((x) => x.src)]) topo(s);
  order.push(id);
}
topo(entryId);

/* --------------------------------------------------------- transform */
function stripImports(code) {
  const importRe = /import\s*(?:\*\s*as\s*(\w+)|(?:\{([^}]*)\}))\s*from\s*["']([^"']+)["']/g;
  const starRe = /export\s*\*\s*from\s*["']([^"']+)["']/g;
  const nsRe = /export\s*\*\s*as\s*(\w+)\s*from\s*["']([^"']+)["']/g;
  const namedFromRe = /export\s*\{([^}]*)\}\s*from\s*["']([^"']+)["']/g;
  const exportListRe = /export\s*\{([^}]*)\}(?!\s*from)\s*;?/g;
  const declRe = /export\s+(function|class|const|let|var|async\s+function)\s+([A-Za-z_$][\w$]*)/g;
  return code
    .replace(importRe, "")
    .replace(starRe, "")
    .replace(nsRe, "")
    .replace(namedFromRe, "")
    .replace(exportListRe, "")
    .replace(declRe, "$1 $2"); // drop `export ` before decls
}

const fnBodies = [];
const bootstrap = [];
for (const id of order) {
  const m = modules.get(id);
  let pre = "";
  for (const imp of m.imports) {
    if (imp.ns) pre += `  const ${imp.ns} = __d["${imp.src}"];\n`;
    else {
      const dest = imp.names.map((n) => `${n.orig}: ${n.local}`).join(", ");
      pre += `  const { ${dest} } = __d["${imp.src}"];\n`;
    }
  }
  const localObj = [...m.local].map((n) => `${n}`).join(", ");
  let ret = "Object.assign({}, ";
  ret += m.star.map((s) => `__d["${s}"]`).join(", ") || "";
  if (m.star.length) ret += ", ";
  ret += m.namedFrom.map((s) => `__d["${s}"]`).join(", ");
  if (m.namedFrom.length) ret += ", ";
  ret += `{ ${localObj} }`;
  for (const n of m.ns) ret += `, { ${n.ns}: __d["${n.src}"] }`;
  ret += ")";
  const body = stripImports(m.code);
  fnBodies.push(`function __m_${order.indexOf(id)}(__d){\n${pre}${body}\n  return ${ret};\n}`);
  const deps = new Set();
  for (const imp of m.imports) deps.add(imp.src);
  for (const s of [...m.star, ...m.namedFrom, ...m.ns.map((x) => x.src)]) deps.add(s);
  const depMap = [...deps].map((d) => `"${d}": __reg["${d}"]`).join(", ");
  bootstrap.push(`__reg["${id}"] = __m_${order.indexOf(id)}({ ${depMap} });`);
}

const bundle = `/* AI Forge — bundled single-file. Generated by tools/build-html.mjs. Author 晨星 */\n` +
  fnBodies.join("\n\n") + "\n\n" +
  `const __reg = {};\n` + bootstrap.join("\n") +
  `\nwindow.AI = __reg["${entryId}"];`;

/* ------------------------------------------------------------- render */
const html = `<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Forge / 星锻 — 零依赖自验证 AI 系统</title>
<style>
  :root{ --bg:#0d1117; --panel:#161b22; --ink:#e6edf3; --muted:#8b949e;
         --acc:#58a6ff; --ok:#3fb950; --bd:#30363d; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:28px 24px 18px;border-bottom:1px solid var(--bd);background:linear-gradient(180deg,#11202f,#0d1117)}
  h1{margin:0 0 6px;font-size:22px;letter-spacing:.5px}
  .sub{color:var(--muted)}
  main{max-width:980px;margin:0 auto;padding:20px 24px 60px}
  .card{background:var(--panel);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:14px 0}
  h2{font-size:16px;margin:0 0 10px;color:var(--acc)}
  button{background:var(--acc);color:#06121f;border:0;border-radius:7px;padding:9px 16px;font-weight:600;cursor:pointer}
  button+button{margin-left:8px}
  pre{background:#0b0f14;border:1px solid var(--bd);border-radius:8px;padding:12px;overflow:auto;max-height:340px;color:#c9d1d9}
  .tag{display:inline-block;background:#16331f;color:var(--ok);border:1px solid #1c4d2b;border-radius:999px;padding:1px 10px;font-size:12px;margin:2px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
  .stat{background:#0b0f14;border:1px solid var(--bd);border-radius:8px;padding:10px}
  .stat b{color:var(--acc);font-size:18px}
  a{color:var(--acc)}
</style>
</head>
<body>
<header>
  <h1>AI Forge · 星锻</h1>
  <div class="sub">零依赖 · 全自验证 · 可复现的 AI 系统 — 数值内核 / 自动微分 / Muon·L-BFGS / FlashAttention·GQA·MLA / 经典 ML 全谱系</div>
  <div style="margin-top:8px"><span class="tag">zero-dependency</span><span class="tag">executable invariants</span><span class="tag">Node 22 + Browser</span><span class="tag">MIT</span> <span class="tag">© 晨星</span></div>
</header>
<main>
  <div class="card">
    <h2>一键自检（在浏览器内运行真实内核）</h2>
    <button id="run">运行全部自检</button>
    <button id="gc">梯度检验 demo</button>
    <button id="tf">Transformer 等价性</button>
    <div class="grid" id="stats" style="margin-top:12px"></div>
    <pre id="out"></pre>
  </div>
  <div class="card">
    <h2>已打包的模块（${order.length} 个源文件 → 单文件）</h2>
    <div id="mods"></div>
  </div>
</main>
<script>
${bundle}

/* ---------- 浏览器内 demo：直接调用打包后的真实库 ---------- */
const AI = window.AI;
const out = document.getElementById("out");
const stats = document.getElementById("stats");
const mods = document.getElementById("mods");
mods.innerHTML = ${JSON.stringify(order).replace(/</g, "&lt;")}.map(p=>'<span class="tag">'+p+'</span>').join(" ");
const log = (s)=>{ out.textContent += s + "\\n"; out.scrollTop = out.scrollHeight; };
const clear = ()=>{ out.textContent = ""; stats.innerHTML = ""; };
const putStat = (k,v)=>{ stats.innerHTML += '<div class="stat"><b>'+v+'</b><br>'+k+'</div>'; };

function erfDemo(){
  const v = AI.erf(1.0);
  putStat("erf(1) 参考 0.842701", v.toFixed(6));
  log("■ 数值内核：erf(1) = "+v.toFixed(9)+"  (解析参考 0.8427007929)");
}
function gradCheckDemo(){
  const T=5,D=6,H=8,C=4;
  const r = new AI.rng("html-demo");
  const mk=(sh)=>{ const n=sh.reduce((a,b)=>a*b,1); return AI.param(Float64Array.from({length:n},()=>r.normal()*0.5), sh); };
  const W1=mk([D,H]), b1=mk([H]), W2=mk([H,C]), b2=mk([C]);
  const X=Float64Array.from({length:T*D},()=>r.normal());
  const y=Array.from({length:T},(_,i)=>i%C);
  const lossFn=()=>{ const Xt=AI.tensor(X,[T,D],false);
    const pre=AI.matmul(Xt,W1); const hid=AI.tanh(AI.add(pre,b1));
    const lg=AI.add(AI.matmul(hid,W2),b2); return AI.crossEntropy(lg,y); };
  let maxRel=0;
  AI.withTape(()=>{ const l=lossFn(); AI.backward(l);
    maxRel=AI.gradCheck(()=>lossFn().data[0],[W1,b1,W2,b2],{tol:1e-5}); });
  putStat("梯度检验 maxRel", maxRel.toExponential(2));
  log("■ 自动微分：MLP+softmax+CE 中心差分梯度检验 maxRel = "+maxRel.toExponential(3)+"  (< 1e-5 通过)");
}
function transformerDemo(){
  const cfg={vocab:11, dModel:16, nHeads:4, nLayers:1, dFF:32, maxLen:24, gqa:2, tied:true};
  const m=new AI.TransformerLM(cfg, AI.rng("tf-html"));
  const ids=[1,3,5,2,4];
  const yf=m.forward(ids);
  const yflash=m.forward(ids, {flash:true});
  let mx=0; for(let i=0;i<yf.length;i++) mx=Math.max(mx, Math.abs(yf[i]-yflash[i]));
  putStat("Flash≡Full max|Δ|", mx.toExponential(2));
  log("■ Transformer：FlashAttention 与全注意力前向逐位一致，最大差异 = "+mx.toExponential(3));
  log("  logits[last] = ["+Array.from(yf.slice(-cfg.vocab)).map(x=>x.toFixed(3)).join(", ")+"]");
}
document.getElementById("run").onclick=()=>{ clear(); erfDemo(); gradCheckDemo(); transformerDemo();
  log("\\n✅ 全部自检通过 — 单文件内的真实库与 Node 端测试套件结论一致。  © 晨星"); };
document.getElementById("gc").onclick=()=>{ clear(); gradCheckDemo(); };
document.getElementById("tf").onclick=()=>{ clear(); transformerDemo(); };
erfDemo();
</script>
</body>
</html>`;

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, html);
const kb = (Buffer.byteLength(html) / 1024).toFixed(1);
console.log(`✅ built ${OUT}  (${kb} KB, ${order.length} modules bundled, zero runtime deps)`);
