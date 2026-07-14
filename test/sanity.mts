// Runtime sanity check for src/index.ts using a mock OpenAI-compatible server.
// Run: node test/sanity.mts
import { createServer } from "node:http";
import { pathToFileURL } from "node:url";

// Inject config via env before importing the extension (it reads env at load).
const PORT = 54329;
const BASE = `http://localhost:${PORT}/v1`;
process.env.PI_LOCAL_SERVERS = BASE;

const mod = await import(pathToFileURL(`${process.cwd()}/src/index.ts`).href);
const factory = mod.default;

// ---- mock pi ----
const registered = new Map();
const events = new Map();
const commands = new Map();
const fakeCtx = () => ({
	cwd: process.cwd(),
	mode: "tui",
	hasUI: true,
	isProjectTrusted: () => false,
	ui: {
		notify: (m, t) => console.log(`[notify:${t}] ${m}`),
		setStatus: (k, v) => console.log(`[status:${k}] ${v ?? "(cleared)"}`),
	},
});
const pi = {
	on: (ev, fn) => { if (!events.has(ev)) events.set(ev, []); events.get(ev).push(fn); },
	registerProvider: (name, cfg) => { registered.set(name, cfg); },
	unregisterProvider: (name) => { registered.delete(name); },
	registerCommand: (name, opts) => { commands.set(name, opts); },
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- mock server ----
let up = true;
const server = createServer((req, res) => {
	if (!up) { req.socket.destroy(); return; } // simulate "killed": drop connection
	if (req.url === "/v1/models") {
		res.setHeader("content-type", "application/json");
		res.end(JSON.stringify({ data: [{ id: "mock-llama:7b", context_window: 8192 }, { id: "mock-qwen:14b" }] }));
	} else {
		res.statusCode = 404; res.end();
	}
});

function start() { return new Promise((r) => server.listen(PORT, r)); }
function close() { return new Promise((r) => server.close(r)); }

// ---- run ----
await start();
await factory(pi);

console.log("\n== after startup discovery ==");
console.log("registered providers:", [...registered.keys()]);
const cfg = registered.get(`local-localhost-${PORT}`);
console.log("provider name local-localhost-" + PORT + " models:", cfg?.models?.map((m) => m.id));
console.log("compat sample:", cfg?.models?.[0]?.compat);

// fire session_start
console.log("\n== firing session_start ==");
await events.get("session_start")[0]({ reason: "startup" }, fakeCtx());
await wait(50);

// now kill the server and wait for a poll cycle (pollIntervalMs defaults to 5000)
console.log("\n== killing mock server, waiting for poll cycle ==");
up = false;
await close();
// force an immediate rescan via the command instead of waiting 5s
await commands.get("local-llama-rescan").handler("", fakeCtx());
await wait(50);

console.log("\n== after server killed + rescan ==");
console.log("registered providers:", [...registered.keys()]);

// shutdown
await events.get("session_shutdown")[0]({ reason: "quit" }, fakeCtx());
console.log("\nDONE. Expectation: 1 provider before kill, 0 after kill.");
