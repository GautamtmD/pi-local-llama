// Verifies the per-server `reasoning`, `maxTokens` and `thinkingLevelMap`
// options: a server marked `reasoning` registers thinking models that send
// reasoning_effort, and `maxTokens` overrides the 8192 default. A plain server
// keeps the safe defaults.
// Run: node test/reasoning.mts
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const TMP = mkdtempSync(join(tmpdir(), "pi-llama-reason-"));
process.env.PI_CODING_AGENT_DIR = TMP;
delete process.env.PI_LOCAL_SERVERS;

const mock = (id) => createServer((req, res) => {
	res.setHeader("content-type", "application/json");
	if (req.url === "/v1/models") res.end(JSON.stringify({ data: [{ id, meta: { n_ctx: 204800 } }] }));
	else { res.statusCode = 404; res.end(); }
});
const PORT_A = 54371; // thinking server
const PORT_B = 54372; // plain server
const serverA = mock("thinker");
const serverB = mock("plain");
await new Promise((r) => serverA.listen(PORT_A, r));
await new Promise((r) => serverB.listen(PORT_B, r));

const levelMap = { off: "none", minimal: "low" };
writeFileSync(
	join(TMP, "local-llama-servers.json"),
	JSON.stringify({
		pollIntervalMs: 60000, timeoutMs: 2000,
		servers: [
			{ baseUrl: `http://localhost:${PORT_A}/v1`, reasoning: true, maxTokens: 32768, thinkingLevelMap: levelMap },
			`http://localhost:${PORT_B}/v1`,
		],
	}),
);

const mod = await import(pathToFileURL(`${process.cwd()}/src/index.ts`).href);
const registered = new Map();
const events = new Map();
const ctx = {
	cwd: process.cwd(), mode: "tui", hasUI: true, isProjectTrusted: () => false,
	ui: { notify() {}, setStatus() {} },
};
const pi = {
	on: (ev, fn) => { if (!events.has(ev)) events.set(ev, []); events.get(ev).push(fn); },
	registerProvider: (name, cfg) => { registered.set(name, cfg); },
	unregisterProvider: (name) => { registered.delete(name); },
	registerCommand: () => {},
};
const model = (port) => registered.get(`local-localhost-${port}`)?.models?.[0];

try {
	await mod.default(pi);
	await events.get("session_start")[0]({ reason: "startup" }, ctx);
	await new Promise((r) => setTimeout(r, 30));

	const a = model(PORT_A);
	assert.equal(a.reasoning, true);
	assert.equal(a.maxTokens, 32768);
	assert.deepEqual(a.thinkingLevelMap, levelMap);
	assert.equal(a.compat.supportsReasoningEffort, true);

	const b = model(PORT_B);
	assert.equal(b.reasoning, false);
	assert.equal(b.maxTokens, 8192);
	assert.equal(b.thinkingLevelMap, undefined);
	assert.equal(b.compat.supportsReasoningEffort, false);

	await events.get("session_shutdown")[0]({ reason: "quit" }, ctx);
	console.log("PASS");
} finally {
	try { rmSync(TMP, { recursive: true, force: true }); } catch {}
	serverA.close(); serverB.close();
}
process.exit(0);
