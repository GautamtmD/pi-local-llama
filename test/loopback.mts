// Run: node test/loopback.mts — a `localhost` server shadowed by another program
// on the IPv6 loopback (e.g. `python -m http.server <port>` bound to [::]) must
// fall back to 127.0.0.1, and register THAT url so chat requests go there too.
import { strict as assert } from "node:assert";
import { createServer } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const tmp = mkdtempSync(join(tmpdir(), "pi-llama-loopback-"));
process.env.PI_CODING_AGENT_DIR = tmp;
delete process.env.PI_LOCAL_SERVERS;

// The real model server, IPv4 loopback only (like llama-server's default).
const real = createServer((req, res) => {
	res.setHeader("content-type", "application/json");
	res.setHeader("connection", "close"); // no keep-alive reuse: each poll re-resolves localhost
	if (req.url === "/v1/models") res.end(JSON.stringify({ data: [{ id: "real-model" }] }));
	else if (req.url === "/v1/props") res.end(JSON.stringify({ default_generation_settings: { n_ctx: 4096 } }));
	else { res.writeHead(404); res.end(); }
});
// The impostor on the same port, IPv6 loopback: a static file server.
let impostorStatus = 404;
const impostor = createServer((req, res) => {
	res.writeHead(impostorStatus, { "content-type": "text/html" });
	res.end("<!DOCTYPE HTML><html><body><h1>Error response</h1></body></html>");
});

await new Promise((r) => real.listen(0, "127.0.0.1", r));
const port = real.address().port;
const ipv6 = await new Promise((resolve) => {
	impostor.once("error", () => resolve(false));
	impostor.listen({ port, host: "::1", ipv6Only: true }, () => resolve(true));
});
if (!ipv6) {
	console.log("SKIP loopback: IPv6 loopback unavailable on this machine");
	real.close();
	rmSync(tmp, { recursive: true, force: true });
	process.exit(0);
}

const configured = `http://localhost:${port}/v1`;
const name = `local-localhost-${port}`;
writeFileSync(join(tmp, "local-llama-servers.json"), JSON.stringify({ servers: [configured], pollIntervalMs: 60000 }));

const factory = (await import(pathToFileURL(`${process.cwd()}/src/index.ts`).href)).default;
const providers = new Map();
const notices = [];
const events = new Map(), commands = new Map();
const pi = {
	on: (n, fn) => events.set(n, fn),
	registerProvider: (n, config) => providers.set(n, config),
	unregisterProvider: (n) => providers.delete(n),
	registerCommand: (n, command) => commands.set(n, command),
};
const ctx = {
	cwd: tmp, hasUI: true, mode: "tui", isProjectTrusted: () => false,
	ui: { notify: (message, level) => notices.push({ message, level }), setStatus: () => {} },
};

try {
	// Sanity: plain localhost really does hit the impostor in this setup.
	const direct = await fetch(`${configured}/models`).catch(() => undefined);
	assert.ok(direct && direct.status === 404, "localhost should reach the IPv6 impostor first");

	await factory(pi);
	await events.get("session_start")({ reason: "startup" }, ctx);
	const cfg = providers.get(name);
	assert.ok(cfg, "provider keeps its localhost-derived name");
	assert.equal(cfg.baseUrl, `http://127.0.0.1:${port}/v1`, "chat requests must go to the real server");
	assert.deepEqual(cfg.models.map((m) => m.id), ["real-model"]);
	assert.equal(cfg.models[0].contextWindow, 4096, "context comes from the real server's /props");
	const warn = notices.find((n) => n.level === "warning" && n.message.includes("using http://127.0.0.1"));
	assert.ok(warn, "user is told about the reroute at startup");

	// Unchanged on the next poll: no duplicate warning.
	const before = notices.length;
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(notices.slice(before).filter((n) => n.message.includes("using http://127.0.0.1")).length, 0);

	// A 200 HTML page is also not a model list.
	impostorStatus = 200;
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(providers.get(name).baseUrl, `http://127.0.0.1:${port}/v1`);

	// Impostor goes away: localhost works again and the configured URL is restored.
	await new Promise((r) => impostor.close(r));
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(providers.get(name).baseUrl, configured);

	// Impostor reappears mid-session: rerouted again, with the reason shown.
	impostorStatus = 404;
	await new Promise((r) => impostor.listen({ port, host: "::1", ipv6Only: true }, r));
	const mark = notices.length;
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(providers.get(name).baseUrl, `http://127.0.0.1:${port}/v1`);
	assert.ok(notices.slice(mark).some((n) => n.level === "warning" && n.message.includes("[HTTP 404]")));
	await new Promise((r) => impostor.close(r));

	// Real server down too: provider is removed as before.
	await new Promise((r) => real.close(r));
	await commands.get("local-llama-rescan").handler("", ctx);
	assert.equal(providers.has(name), false);

	console.log("PASS loopback fallback");
} finally {
	events.get("session_shutdown")?.({}, ctx);
	real.close();
	impostor.close();
	rmSync(tmp, { recursive: true, force: true });
}
