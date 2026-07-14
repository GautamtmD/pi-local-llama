// Verifies /local-llama-servers add|list|remove against a mock server.
// Run: node test/servers.mts
import { createServer } from "node:http";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const PORT = 54351;
const TMP = mkdtempSync(join(tmpdir(), "pi-llama-srv-"));
process.env.PI_CODING_AGENT_DIR = TMP;
delete process.env.PI_LOCAL_SERVERS; // use only the config file

// mock server
const server = createServer((req, res) => {
	if (req.url === "/v1/models") {
		res.setHeader("content-type", "application/json");
		res.end(JSON.stringify({ data: [{ id: "mock-llama:7b" }] }));
	} else { res.statusCode = 404; res.end(); }
});
await new Promise((r) => server.listen(PORT, r));

const mod = await import(pathToFileURL(`${process.cwd()}/src/index.ts`).href);
const factory = mod.default;

const registered = new Map();
const events = new Map();
const commands = new Map();
const notifies = [];
const push = (m, t) => notifies.push(`[${t}] ${m}`);
const ctx = {
	cwd: process.cwd(), mode: "tui", hasUI: true, isProjectTrusted: () => false,
	ui: { notify: push, setStatus() {} },
};
const pi = {
	on: (ev, fn) => { if (!events.has(ev)) events.set(ev, []); events.get(ev).push(fn); },
	registerProvider: (n, c) => { registered.set(n, c); },
	unregisterProvider: (n) => { registered.delete(n); },
	registerCommand: (n, o) => { commands.set(n, o); },
};
const cfgPath = join(TMP, "local-llama-servers.json");
const readCfg = () => { try { return JSON.parse(readFileSync(cfgPath, "utf8")); } catch { return null; } };

try {
	await factory(pi);
	await events.get("session_start")[0]({ reason: "startup" }, ctx);

	console.log("=== /local-llama-servers add localhost:" + PORT + " (bare host, expect /v1 appended) ===");
	await commands.get("local-llama-servers").handler(`add localhost:${PORT}`, ctx);
	console.log("config file:", JSON.stringify(readCfg()));
	console.log("registered:", [...registered.keys()]);
	console.log("notifies:", notifies.slice(-3));

	console.log("\n=== /local-llama-servers add localhost:" + PORT + " (again, expect 'already configured') ===");
	notifies.length = 0;
	await commands.get("local-llama-servers").handler(`add localhost:${PORT}`, ctx);
	console.log("notifies:", notifies);

	console.log("\n=== /local-llama-servers list ===");
	notifies.length = 0;
	await commands.get("local-llama-servers").handler("list", ctx);
	console.log("notifies:", notifies);

	console.log("\n=== /local-llama-servers remove localhost:" + PORT + " ===");
	notifies.length = 0;
	await commands.get("local-llama-servers").handler(`remove localhost:${PORT}`, ctx);
	console.log("config file:", JSON.stringify(readCfg()));
	console.log("registered (expect empty):", [...registered.keys()]);
	console.log("notifies:", notifies.slice(-2));

	await events.get("session_shutdown")[0]({ reason: "quit" }, ctx);
} finally {
	try { rmSync(TMP, { recursive: true, force: true }); } catch {}
	server.close();
}
process.exit(0);
