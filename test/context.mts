// Verifies live context-window detection from llama-server, using the EXACT
// payload shapes observed from a real llama-server:
//   - /v1/models  -> data[].meta.n_ctx            (running context per model)
//   - /props      -> default_generation_settings.n_ctx (per-slot running context)
//   - /slots      -> [{ n_ctx, ... }]              (per-slot running context)
// Also verifies that changing the server's context (e.g. relaunching with a
// different -c) causes a re-register with the updated contextWindow.
// Run: node test/context.mts
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const TMP = mkdtempSync(join(tmpdir(), "pi-llama-ctx-"));
process.env.PI_CODING_AGENT_DIR = TMP;
delete process.env.PI_LOCAL_SERVERS;

// ---- mock llama-server (port A): full /props + /slots + /v1/models ----
const PORT_A = 54361;
let ctxA = 100096; // mutable to simulate relaunch with a different -c
const serverA = createServer((req, res) => {
	res.setHeader("content-type", "application/json");
	if (req.url === "/v1/models") {
		res.end(JSON.stringify({
			object: "list",
			data: [{ id: "qwen:27b", object: "model", meta: { n_ctx: ctxA, n_ctx_train: 262144 } }],
		}));
	} else if (req.url === "/props") {
		res.end(JSON.stringify({ total_slots: 1, default_generation_settings: { params: {}, n_ctx: ctxA } }));
	} else if (req.url === "/slots") {
		res.end(JSON.stringify([{ id: 0, n_ctx: ctxA }]));
	} else { res.statusCode = 404; res.end(); }
});

// ---- mock (port B): OpenAI-only, NO /props or /slots, but meta.n_ctx present ----
const PORT_B = 54362;
const serverB = createServer((req, res) => {
	res.setHeader("content-type", "application/json");
	if (req.url === "/v1/models") {
		res.end(JSON.stringify({ data: [{ id: "tiny:1b", meta: { n_ctx: 8192 } }] }));
	} else { res.statusCode = 404; res.end(); } // /props and /slots 404
});

// ---- mock (port C): OpenAI-only, reports NOTHING about context ----
const PORT_C = 54363;
const serverC = createServer((req, res) => {
	res.setHeader("content-type", "application/json");
	if (req.url === "/v1/models") res.end(JSON.stringify({ data: [{ id: "anon:7b" }] }));
	else { res.statusCode = 404; res.end(); }
});

await new Promise((r) => serverA.listen(PORT_A, r));
await new Promise((r) => serverB.listen(PORT_B, r));
await new Promise((r) => serverC.listen(PORT_C, r));

// seed config
const { writeFileSync } = await import("node:fs");
writeFileSync(
	join(TMP, "local-llama-servers.json"),
	JSON.stringify({
		pollIntervalMs: 60000, timeoutMs: 2000,
		servers: [`http://localhost:${PORT_A}/v1`, `http://localhost:${PORT_B}/v1`, `http://localhost:${PORT_C}/v1`],
	}),
);

const mod = await import(pathToFileURL(`${process.cwd()}/src/index.ts`).href);
const factory = mod.default;

let registerCalls = 0;
const registered = new Map();
const events = new Map();
const commands = new Map();
const ctx = {
	cwd: process.cwd(), mode: "tui", hasUI: true, isProjectTrusted: () => false,
	ui: { notify() {}, setStatus() {} },
};
const pi = {
	on: (ev, fn) => { if (!events.has(ev)) events.set(ev, []); events.get(ev).push(fn); },
	registerProvider: (name, cfg) => { registered.set(name, cfg); registerCalls++; },
	unregisterProvider: (name) => { registered.delete(name); },
	registerCommand: (name, o) => { commands.set(name, o); },
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const modelCtx = (provider) => registered.get(provider)?.models?.[0]?.contextWindow;

try {
	await factory(pi);
	await events.get("session_start")[0]({ reason: "startup" }, ctx);
	await wait(30);

	console.log("=== after startup (ctxA = 100096) ===");
	console.log("A qwen:27b contextWindow :", modelCtx(`local-localhost-${PORT_A}`), "(expect 100096 from /props)");
	console.log("B tiny:1b  contextWindow :", modelCtx(`local-localhost-${PORT_B}`), "(expect 8192 from meta.n_ctx, no /props)");
	console.log("C anon:7b  contextWindow :", modelCtx(`local-localhost-${PORT_C}`), "(expect 128000 default, nothing reported)");
	const callsAfterStartup = registerCalls;

	console.log("\n=== relaunch server A with -c 32768 (smaller context) ===");
	ctxA = 32768;
	await commands.get("local-llama-rescan").handler("", ctx);
	await wait(30);
	console.log("A qwen:27b contextWindow :", modelCtx(`local-localhost-${PORT_A}`), "(expect 32768)");
	console.log("registerProvider called again?", registerCalls > callsAfterStartup, `(calls: ${registerCalls} > ${callsAfterStartup})`);

	console.log("\n=== rescan with NO change (should NOT re-register) ===");
	const callsBefore = registerCalls;
	await commands.get("local-llama-rescan").handler("", ctx);
	await wait(30);
	console.log("registerProvider called again?", registerCalls > callsBefore, `(expect false; calls ${registerCalls} == ${callsBefore})`);

	await events.get("session_shutdown")[0]({ reason: "quit" }, ctx);
	console.log("\nDONE.");
} finally {
	try { rmSync(TMP, { recursive: true, force: true }); } catch {}
	serverA.close(); serverB.close(); serverC.close();
}
process.exit(0);
