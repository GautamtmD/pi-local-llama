// Run: node test/auth.mts — missing/rejected keys, persistence and provider refresh.
import { strict as assert } from "node:assert";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, rmSync, writeFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const tmp = mkdtempSync(join(tmpdir(), "pi-llama-auth-"));
process.env.PI_CODING_AGENT_DIR = tmp;
delete process.env.PI_LOCAL_SERVERS;
const server = createServer((req, res) => {
	if (req.headers.authorization !== `Bearer ${expectedKey}` && (!alternateKey || req.headers.authorization !== `Bearer ${alternateKey}`)) {
		res.writeHead(401); res.end(); return;
	}
	res.setHeader("content-type", "application/json");
	if (req.url === "/v1/models") res.end(JSON.stringify({ data: [{ id: "protected-model" }] }));
	else if (req.url === "/v1/props") res.end(JSON.stringify({ default_generation_settings: { n_ctx: 4096 } }));
	else { res.writeHead(404); res.end(); }
});
let expectedKey = "correct-key";
let alternateKey;
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
const name = `local-127-0-0-1-${server.address().port}`;
const configPath = join(tmp, "local-llama-servers.json");
const keyPath = join(tmp, "local-llama-credentials.json");
writeFileSync(configPath, JSON.stringify({ servers: [baseUrl], pollIntervalMs: 60000 }));
const factory = (await import(pathToFileURL(`${process.cwd()}/src/index.ts`).href)).default;
let inputValue;
const notices = [];
const statuses = [];
const providers = new Map();
let registrations = 0;
const ctx = {
	cwd: tmp, hasUI: true, mode: "tui", isProjectTrusted: () => false,
	ui: {
		notify: (message, level) => notices.push({ message, level }),
		setStatus: (_, text) => statuses.push(text),
		input: async () => inputValue,
	},
};
async function instance() {
	const events = new Map(), commands = new Map();
	const pi = {
		on: (n, fn) => events.set(n, fn),
		registerProvider: (n, config) => { providers.set(n, config); registrations++; },
		unregisterProvider: (n) => providers.delete(n),
		registerCommand: (n, command) => commands.set(n, command),
	};
	await factory(pi);
	await events.get("session_start")({ reason: "startup" }, ctx);
	return { events, commands };
}
const warning = (text) => notices.some((n) => n.level === "warning" && n.message.includes(text));
try {
	let { events, commands } = await instance();
	assert.equal(providers.size, 0);
	assert.ok(warning("requires an API key"));
	assert.ok(statuses.at(-1).includes("needs API key"));
	const warningsBefore = notices.filter((n) => n.message.includes("requires an API key")).length;
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(notices.filter((n) => n.message.includes("requires an API key")).length, warningsBefore);
	inputValue = undefined;
	await commands.get("local-llama-servers").handler(`key ${baseUrl}`, ctx);
	assert.throws(() => readFileSync(keyPath));
	inputValue = "wrong-key";
	await commands.get("local-llama-servers").handler(`key ${baseUrl}`, ctx);
	assert.ok(warning("rejected its API key"));
	assert.equal(providers.size, 0);
	assert.equal(JSON.parse(readFileSync(keyPath, "utf8"))[baseUrl], "wrong-key");
	if (process.platform !== "win32") assert.equal(statSync(keyPath).mode & 0o777, 0o600);
	await commands.get("local-llama-servers").handler("list", ctx);
	assert.ok(notices.at(-1).message.includes("key rejected"));
	inputValue = expectedKey;
	await commands.get("local-llama-servers").handler(`key ${baseUrl}`, ctx);
	assert.equal(providers.get(name).apiKey, expectedKey);
	assert.equal(providers.get(name).models[0].contextWindow, 4096);
	assert.ok(!statuses.at(-1).includes("needs API key"));
	const callsBefore = registrations;
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(registrations, callsBefore);
	// A replacement key must refresh the provider even if the old key still works.
	alternateKey = expectedKey;
	expectedKey = "rotated-key";
	inputValue = expectedKey;
	await commands.get("local-llama-servers").handler(`key ${baseUrl}`, ctx);
	assert.equal(registrations, callsBefore + 1);
	assert.equal(providers.get(name).apiKey, expectedKey);
	// Reject a previously valid key, notify once, and remove the stale provider.
	alternateKey = undefined;
	expectedKey = "second-rotated-key";
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(providers.size, 0);
	assert.ok(warning("rejected its API key"));
	assert.ok(statuses.at(-1).includes("key rejected"));
	const rejectionsBefore = notices.filter((n) => n.message.includes("rejected its API key")).length;
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(notices.filter((n) => n.message.includes("rejected its API key")).length, rejectionsBefore);
	// Enter the new key; it must be used immediately and after restart.
	inputValue = expectedKey;
	await commands.get("local-llama-servers").handler(`key ${baseUrl}`, ctx);
	assert.equal(registrations, callsBefore + 2);
	assert.equal(providers.get(name).apiKey, expectedKey);
	assert.ok(!JSON.stringify(notices).includes("second-rotated-key"));
	await events.get("session_shutdown")();
	providers.clear();
	({ events, commands } = await instance());
	assert.equal(providers.get(name).apiKey, expectedKey);
	await commands.get("local-llama-servers").handler(`key-remove ${baseUrl}`, ctx);
	assert.equal(providers.size, 0);
	assert.ok(warning("requires an API key"));
	assert.equal(JSON.parse(readFileSync(keyPath, "utf8"))[baseUrl], undefined);
	// Explicit server config keys take precedence over saved credentials.
	writeFileSync(configPath, JSON.stringify({ servers: [{ baseUrl, apiKey: expectedKey }] }));
	await events.get("session_shutdown")();
	providers.clear();
	({ events, commands } = await instance());
	assert.equal(providers.get(name).apiKey, expectedKey);
	inputValue = "wrong-key";
	await commands.get("local-llama-servers").handler(`key ${baseUrl}`, ctx);
	assert.equal(providers.get(name).apiKey, expectedKey);
	assert.ok(warning("takes precedence"));
	assert.ok(!JSON.stringify(notices).includes("wrong-key"));
	await events.get("session_shutdown")();
	console.log("auth: OK");
} finally {
	server.close();
	rmSync(tmp, { recursive: true, force: true });
}
