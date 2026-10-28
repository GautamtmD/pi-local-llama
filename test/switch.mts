// Run: node test/switch.mts
// Uses Pi's real prompt preflight and model registry, with a fake extension API
// and agent loop, to verify restart recovery and manual-selection races.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const TMP = mkdtempSync(join(tmpdir(), "pi-llama-switch-"));
process.env.PI_CODING_AGENT_DIR = TMP;
delete process.env.PI_LOCAL_SERVERS;

let served = ["qwen:27b"];
let down = false;
let modelRequests = 0;
let holdRequest;
const server = createServer((req, res) => {
	res.setHeader("content-type", "application/json");
	if (down) { res.writeHead(503); res.end(); return; }
	if (req.url !== "/v1/models") { res.writeHead(404); res.end(); return; }
	modelRequests++;
	const respond = () => res.end(JSON.stringify({ data: served.map((id) => ({ id })) }));
	if (holdRequest?.number === modelRequests) {
		const { reached } = holdRequest;
		holdRequest = undefined;
		reached(respond);
	} else respond();
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const PORT = server.address().port;
const PROVIDER = `local-127-0-0-1-${PORT}`;
const config = { pollIntervalMs: 60000, timeoutMs: 2000, autoScopePattern: false, servers: [`http://127.0.0.1:${PORT}/v1`] };
const writeConfig = (extra = {}) => writeFileSync(join(TMP, "local-llama-servers.json"), JSON.stringify({ ...config, ...extra }));
writeConfig();

const { AgentSession, AuthStorage, ModelRegistry } = await import("@earendil-works/pi-coding-agent");
const factory = (await import(pathToFileURL(join(process.cwd(), "src/index.ts")).href)).default;
const registry = ModelRegistry.inMemory(AuthStorage.inMemory());
const events = new Map();
const commands = new Map();
const statuses = new Map();
let currentModel;
let idle = true;
let setModelCalls = 0;
let promptedModel;
const ctx = {
	cwd: TMP, mode: "tui", hasUI: true, isProjectTrusted: () => false, isIdle: () => idle,
	ui: { notify() {}, setStatus: (k, v) => { if (v === undefined) statuses.delete(k); else statuses.set(k, v); } },
	get model() { return currentModel; },
	modelRegistry: registry,
};
const pi = {
	on: (event, handler) => events.set(event, handler),
	registerProvider: (name, cfg) => {
		registry.registerProvider(name, cfg);
		// Pi refreshes a selected model's definition when it is still registered.
		if (currentModel) currentModel = registry.find(currentModel.provider, currentModel.id) ?? currentModel;
	},
	unregisterProvider: (name) => registry.unregisterProvider(name),
	registerCommand: (name, command) => commands.set(name, command),
	setModel: async (model) => {
		if (!registry.hasConfiguredAuth(model)) return false;
		currentModel = model;
		setModelCalls++;
		return true;
	},
};
const input = () => events.get("input")?.({ type: "input", text: "hi", source: "interactive" }, ctx);
// Execute Pi's own input -> auth validation -> before_agent_start ordering.
// Stop at the agent loop, so the test never sends an inference request.
const promptSession = {
	get model() { return currentModel; },
	get isStreaming() { return !idle; },
	_modelRegistry: registry,
	_extensionRunner: {
		hasHandlers: (event) => events.has(event),
		emitInput: async () => (await input()) ?? { action: "continue" },
		emitBeforeAgentStart: async () => events.get("before_agent_start")?.({ type: "before_agent_start", prompt: "hi" }, ctx),
	},
	_flushPendingBashMessages() {},
	_findLastAssistantMessage: () => undefined,
	_pendingNextTurnMessages: [],
	agent: { state: {} },
	_runAgentPrompt: async () => { promptedModel = currentModel; },
};
const prompt = () => AgentSession.prototype.prompt.call(promptSession, "hi", { expandPromptTemplates: false });
const rescan = () => commands.get("local-llama-rescan").handler("", ctx);
let failed = 0;
async function check(label, run) {
	try { await run(); console.log(`PASS  ${label}`); }
	catch (error) { failed++; console.error(`FAIL  ${label}`, error); }
}

try {
	await factory(pi);
	await events.get("session_start")({ reason: "startup" }, ctx);
	currentModel = registry.find(PROVIDER, "qwen:27b");
	assert.ok(currentModel);
	// A separately configured provider represents a user's manual selection.
	registry.registerProvider("manual", { api: "openai-completions", apiKey: "local", baseUrl: config.servers[0], models: [{ ...currentModel, id: "chosen" }] });
	const otherModel = registry.find("manual", "chosen");

	await check("served model stays selected", async () => {
		await prompt();
		assert.equal(setModelCalls, 0);
		assert.equal(promptedModel.id, "qwen:27b");
	});
	await check("missing model switches alphabetically before the prompt", async () => {
		served = ["mistral:7b", "gemma:9b", "zephyr:3b"];
		await prompt();
		assert.equal(promptedModel.id, "gemma:9b");
		assert.equal(setModelCalls, 1);
		assert.ok(registry.find(PROVIDER, "gemma:9b"));
		const status = statuses.get("local-llama-switch") ?? "";
		assert.ok(status.includes("qwen:27b") && status.includes("gemma:9b"));
		await prompt();
		assert.equal(setModelCalls, 1);
	});
	await check("non-local model skips discovery", async () => {
		currentModel = otherModel;
		const requests = modelRequests;
		await prompt();
		assert.equal(currentModel, otherModel);
		assert.equal(modelRequests, requests);
	});
	await check("empty or unreachable server does not switch", async () => {
		currentModel = registry.find(PROVIDER, "gemma:9b");
		const calls = setModelCalls;
		served = [];
		await prompt();
		down = true;
		try { await prompt(); } finally { down = false; }
		assert.equal(currentModel.id, "gemma:9b");
		assert.equal(setModelCalls, calls);
	});

	for (const recoveredId of ["replacement", "qwen:27b"]) {
		await check(`unregistered server recovers before auth validation (${recoveredId})`, async () => {
			served = ["qwen:27b"];
			await rescan();
			currentModel = registry.find(PROVIDER, "qwen:27b");
			down = true;
			try { await rescan(); } finally { down = false; }
			assert.equal(registry.find(PROVIDER, "qwen:27b"), undefined);
			assert.equal(registry.hasConfiguredAuth(currentModel), false);
			served = [recoveredId];
			const calls = setModelCalls;
			await prompt();
			assert.equal(promptedModel.id, recoveredId);
			assert.equal(registry.hasConfiguredAuth(currentModel), true);
			assert.equal(setModelCalls - calls, recoveredId === "qwen:27b" ? 0 : 1);
		});
	}

	for (const stage of ["probe", "refresh"]) {
		for (const sameProvider of [false, true]) {
			await check(`manual selection during ${stage} is preserved (${sameProvider ? "same" : "other"} provider)`, async () => {
				served = ["qwen:27b", "z-manual-choice"];
				await rescan();
				currentModel = registry.find(PROVIDER, "qwen:27b");
				const manual = sameProvider ? registry.find(PROVIDER, "z-manual-choice") : otherModel;
				served = ["replacement", "z-manual-choice"];
				const reached = new Promise((resolve) => {
					holdRequest = { number: modelRequests + (stage === "probe" ? 1 : 2), reached: resolve };
				});
				const calls = setModelCalls;
				const pending = prompt();
				let release;
				try {
					release = await Promise.race([reached, pending.then(() => { throw new Error("prompt completed before the held server check"); })]);
					currentModel = manual;
					release();
					release = undefined;
					await pending;
					assert.equal(currentModel.provider, manual.provider);
					assert.equal(promptedModel.id, manual.id);
					assert.equal(setModelCalls, calls);
				} finally {
					holdRequest = undefined;
					release?.();
					await pending.catch(() => {});
				}
			});
		}
	}

	await check("queued input does not probe or switch a streaming agent", async () => {
		currentModel = { ...currentModel, provider: PROVIDER, id: "old" };
		const requests = modelRequests;
		const calls = setModelCalls;
		idle = false;
		try { await input(); } finally { idle = true; }
		assert.equal(modelRequests, requests);
		assert.equal(setModelCalls, calls);
	});
	await check("autoSwitchModel false disables discovery and switching", async () => {
		writeConfig({ autoSwitchModel: false });
		await events.get("session_start")({ reason: "reload" }, ctx);
		const requests = modelRequests;
		const calls = setModelCalls;
		await prompt();
		assert.equal(currentModel.id, "old");
		assert.equal(modelRequests, requests);
		assert.equal(setModelCalls, calls);
	});
	await check("headless prompts still switch without UI calls", async () => {
		writeConfig();
		await events.get("session_start")({ reason: "reload" }, ctx);
		ctx.hasUI = false;
		const ui = ctx.ui;
		let uiCalls = 0;
		const noUI = () => { uiCalls++; throw new Error("no UI"); };
		ctx.ui = { notify: noUI, setStatus: noUI };
		try {
			served = ["headless"];
			await prompt();
			assert.equal(promptedModel.id, "headless");
			assert.equal(uiCalls, 0);
		} finally { ctx.ui = ui; ctx.hasUI = true; }
	});
} finally {
	await events.get("session_shutdown")?.({ reason: "quit" }, ctx);
	await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
	rmSync(TMP, { recursive: true, force: true });
}
console.log(failed ? `\n${failed} FAILED` : "\nALL PASSED");
process.exitCode = failed ? 1 : 0;
