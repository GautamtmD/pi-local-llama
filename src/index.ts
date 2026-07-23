/**
 * pi-local-llama
 *
 * Auto-detects OpenAI-compatible local LLM servers (Ollama, LM Studio, vLLM,
 * llama.cpp, SGLang, etc.) and registers their models with pi.
 *
 *   - Reads a list of server base URLs from config.
 *   - Polls each server's `<baseUrl>/models` endpoint.
 *   - When a server is reachable, registers a provider so its models appear in
 *     `/model` and `/scoped-models`.
 *   - When the model list OR a model's live context window changes, re-registers
 *     to keep it up to date.
 *   - When a server is killed / unreachable, unregisters its provider.
 *   - Detects each server's real running context size (not a fixed 128k) by
 *     probing llama-server's `/props` and `/slots` (see resolveContextWindow).
 *   - Optionally manages a scope pattern (see SCOPE_PATTERN) in `settings.json`
 *     (`enabledModels`) so detected models join Ctrl+P cycling.
 *
 * Every provider registered by this extension is named with the `local-`
 * prefix (e.g. `local-localhost-1234`), so the configured scope glob reliably
 * matches `<provider>/<modelId>` references like
 * `local-localhost-1234/llama3.1:8b`.
 *
 * Config (JSON), looked up in order and merged (later wins for duplicate URLs):
 *   1. `<agentDir>/local-llama-servers.json`   (global, always read)
 *   2. `PI_LOCAL_SERVERS` env var              (comma-separated base URLs)
 *   3. `<cwd>/.pi/local-llama-servers.json`    (project-local, trusted projects only)
 *
 * A bare `host:port` (no path) is normalized to `http://host:port/v1`.
 *
 * Commands:
 *   /local-llama-servers add|remove|list [url] [--project]  (server list)
 *   /local-llama-scope   enable|disable|status              (Ctrl+P pattern)
 *   /local-llama-rescan                                     (rescan now)
 *
 * Note on install/uninstall: pi has NO install/uninstall lifecycle hooks for
 * extensions. "Add the pattern on install" is approximated by a one-time,
 * idempotent add on the first `session_start` (see `autoScopePattern`).
 * Removal is NOT automatic — use `/local-llama-scope disable` before
 * `pi remove`, since the extension can no longer run after it is uninstalled.
 */

import { readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
	CONFIG_DIR_NAME,
	getAgentDir,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Every provider we register starts with this prefix. */
const PROVIDER_PREFIX = "local-";
/**
 * Glob pattern added to settings `enabledModels` for Ctrl+P cycling.
 * Must include the `/` because minimatch `*` does not cross it, so this matches
 * `${provider}/${id}` references like `local-localhost-1234/llama3.1:8b`.
 */
const SCOPE_PATTERN = "local-*/*";
/**
 * Fallback context window (tokens) used only when neither the server nor the
 * model payload reports one. llama-server reports the live value via /props,
 * /slots, or the model's `meta.n_ctx`, so this default is rarely hit there.
 */
const DEFAULT_CONTEXT_WINDOW = 128000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ServerCompat {
	supportsDeveloperRole?: boolean;
	supportsReasoningEffort?: boolean;
	maxTokensField?: "max_completion_tokens" | "max_tokens";
	[k: string]: unknown;
}

interface ServerConfig {
	baseUrl: string; // normalized, no trailing slash
	apiKey?: string; // sent as `Authorization: Bearer <apiKey>`; local servers usually ignore it
	name?: string; // display name
	provider?: string; // explicit provider id (will be `local-` prefixed)
	compat?: ServerCompat; // merged into each model's compat
	input?: ("text" | "image")[];
}

interface Config {
	servers: ServerConfig[];
	pollIntervalMs: number;
	timeoutMs: number;
	notify: boolean;
	status: boolean;
	discoverAtStartup: boolean;
	autoScopePattern: boolean; // add `scopePattern` to settings on first startup
	scopeSettings: "global" | "project"; // which settings.json to manage
	scopePattern: string; // the glob to add/remove (default SCOPE_PATTERN)
}

interface NormalizedServer extends ServerConfig {
	modelsUrl: string;
	providerName: string; // always starts with PROVIDER_PREFIX
	displayName: string;
}

interface RegisteredEntry {
	server: NormalizedServer;
	modelIdsKey: string; // sorted, newline-joined model ids; "" if none
	signature: string; // ids + contextWindow + maxTokens; change triggers re-register
}

interface RawModel {
	id?: string;
	name?: string;
	context_window?: number;
	context_length?: number;
	max_tokens?: number;
	[k: string]: unknown;
}

interface DiscoverOk {
	ok: true;
	models: RawModel[];
}
interface DiscoverFail {
	ok: false;
	error: string;
}
type DiscoverResult = DiscoverOk | DiscoverFail;

interface RefreshSummary {
	added: NormalizedServer[];
	changed: NormalizedServer[];
	removed: { server: NormalizedServer; reason: string }[];
	up: { server: NormalizedServer; count: number }[];
	down: { server: NormalizedServer; error: string }[];
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const DEFAULTS: Config = {
	servers: [],
	pollIntervalMs: 5000,
	timeoutMs: 3000,
	notify: true,
	status: true,
	discoverAtStartup: true,
	autoScopePattern: true,
	scopeSettings: "global",
	scopePattern: SCOPE_PATTERN,
};

function errMsg(e: unknown): string {
	if (e instanceof Error) return e.message;
	return String(e);
}

function num(v: unknown, fallback: number): number {
	if (typeof v === "number" && Number.isFinite(v) && v > 0) return Math.floor(v);
	return fallback;
}

/** Lowercase, `[a-z0-9-]` only; collapses other runs into a single dash. */
function sanitize(input: string): string {
	return input
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "");
}

/** Guarantee the provider name is valid and starts with the `local-` prefix. */
function toProviderName(explicit: string | undefined, host: string): string {
	const base = explicit ? sanitize(explicit) : sanitize(`local-${host}`);
	let name = base || "local-llama";
	if (!name.startsWith(PROVIDER_PREFIX)) name = PROVIDER_PREFIX + name;
	return name;
}

function normalizeBaseUrl(raw: string): string | undefined {
	const trimmed = raw.trim();
	if (!trimmed) return undefined;
	const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
	let u: URL;
	try {
		u = new URL(withScheme);
	} catch {
		return undefined;
	}
	// Bare host:port (no path) almost always means the OpenAI-compatible `/v1`
	// endpoint, so append it. Full paths are kept as-is.
	if (u.pathname === "" || u.pathname === "/") u.pathname = "/v1";
	return `${u.protocol}//${u.host}${u.pathname.replace(/\/+$/, "")}${u.search}`;
}

async function readJsonFile(path: string): Promise<unknown | null> {
	try {
		const text = await readFile(path, "utf8");
		return JSON.parse(text);
	} catch (e) {
		const code = (e as NodeJS.ErrnoException)?.code;
		if (code === "ENOENT") return null; // missing file is fine
		console.error(`[pi-local-llama] failed to read ${path}: ${errMsg(e)}`);
		return null;
	}
}

// ---------------------------------------------------------------------------
// Config loading & merging
// ---------------------------------------------------------------------------

function coerceServer(entry: unknown): ServerConfig | undefined {
	if (typeof entry === "string") {
		const baseUrl = normalizeBaseUrl(entry);
		return baseUrl ? { baseUrl } : undefined;
	}
	if (entry && typeof entry === "object") {
		const obj = entry as Record<string, unknown>;
		const baseUrl = normalizeBaseUrl(String(obj.baseUrl ?? obj.url ?? ""));
		if (!baseUrl) return undefined;
		const server: ServerConfig = { baseUrl };
		if (typeof obj.apiKey === "string") server.apiKey = obj.apiKey;
		if (typeof obj.name === "string") server.name = obj.name;
		if (typeof obj.provider === "string") server.provider = obj.provider;
		if (obj.compat && typeof obj.compat === "object") {
			server.compat = { ...(obj.compat as ServerCompat) };
		}
		if (Array.isArray(obj.input)) {
			const input = obj.input.filter((x): x is "text" | "image" => x === "text" || x === "image");
			if (input.length) server.input = input;
		}
		return server;
	}
	return undefined;
}

function coerceConfig(raw: unknown): Config {
	const cfg: Config = { ...DEFAULTS };
	if (!raw || typeof raw !== "object") return cfg;
	const obj = raw as Record<string, unknown>;

	if (typeof obj.pollIntervalMs === "number") cfg.pollIntervalMs = Math.max(1000, obj.pollIntervalMs);
	if (typeof obj.timeoutMs === "number") cfg.timeoutMs = Math.max(250, obj.timeoutMs);
	if (typeof obj.notify === "boolean") cfg.notify = obj.notify;
	if (typeof obj.status === "boolean") cfg.status = obj.status;
	if (typeof obj.discoverAtStartup === "boolean") cfg.discoverAtStartup = obj.discoverAtStartup;
	if (typeof obj.autoScopePattern === "boolean") cfg.autoScopePattern = obj.autoScopePattern;
	if (typeof obj.scopePattern === "string" && obj.scopePattern.trim()) cfg.scopePattern = obj.scopePattern.trim();
	if (obj.scopeSettings === "project") cfg.scopeSettings = "project";
	else if (obj.scopeSettings === "global") cfg.scopeSettings = "global";

	const serverList = Array.isArray(obj.servers) ? obj.servers : [];
	cfg.servers = serverList
		.map(coerceServer)
		.filter((s): s is ServerConfig => s !== undefined);

	return cfg;
}

/** Merge configs: servers accumulate (later `override` wins per baseUrl); scalars override. */
function mergeConfigs(base: Config, override: Config): Config {
	const byUrl = new Map<string, ServerConfig>();
	for (const s of base.servers) byUrl.set(s.baseUrl, s);
	for (const s of override.servers) byUrl.set(s.baseUrl, s);
	const useOverrideScalars = override.servers.length > 0;
	return {
		servers: [...byUrl.values()],
		pollIntervalMs: useOverrideScalars ? override.pollIntervalMs : base.pollIntervalMs,
		timeoutMs: useOverrideScalars ? override.timeoutMs : base.timeoutMs,
		notify: useOverrideScalars ? override.notify : base.notify,
		status: useOverrideScalars ? override.status : base.status,
		discoverAtStartup: base.discoverAtStartup && override.discoverAtStartup,
		autoScopePattern: useOverrideScalars ? override.autoScopePattern : base.autoScopePattern,
		scopeSettings: useOverrideScalars ? override.scopeSettings : base.scopeSettings,
		scopePattern: useOverrideScalars ? override.scopePattern : base.scopePattern,
	};
}

async function loadGlobalConfig(): Promise<Config> {
	const path = join(getAgentDir(), "local-llama-servers.json");
	return coerceConfig(await readJsonFile(path));
}

function loadEnvConfig(): Config {
	const env = process.env.PI_LOCAL_SERVERS;
	if (!env) return { ...DEFAULTS, servers: [] };
	const urls = env.split(",").map((s) => s.trim()).filter(Boolean);
	return { ...DEFAULTS, servers: urls.map(coerceServer).filter((s): s is ServerConfig => !!s) };
}

async function loadProjectConfig(cwd: string): Promise<Config> {
	const path = join(cwd, CONFIG_DIR_NAME, "local-llama-servers.json");
	return coerceConfig(await readJsonFile(path));
}

/** Recompute the effective config from global + env + (trusted) project sources. */
async function computeConfig(ctx: { cwd: string; isProjectTrusted(): boolean }): Promise<Config> {
	let cfg = mergeConfigs(await loadGlobalConfig(), loadEnvConfig());
	if (ctx.isProjectTrusted()) cfg = mergeConfigs(cfg, await loadProjectConfig(ctx.cwd));
	return cfg;
}

// ---------------------------------------------------------------------------
// local-llama-servers.json management (the polled server list)
// ---------------------------------------------------------------------------

function serversConfigPath(scope: "global" | "project", cwd: string): string {
	return scope === "global"
		? join(getAgentDir(), "local-llama-servers.json")
		: join(cwd, CONFIG_DIR_NAME, "local-llama-servers.json");
}

/** Extract/normalize the baseUrl from a stored server entry (string or object). */
function storedServerUrl(entry: unknown): string | undefined {
	if (typeof entry === "string") return normalizeBaseUrl(entry);
	if (entry && typeof entry === "object") {
		const obj = entry as Record<string, unknown>;
		return normalizeBaseUrl(String(obj.baseUrl ?? obj.url ?? ""));
	}
	return undefined;
}

async function readServersObject(path: string): Promise<Record<string, unknown> | null> {
	const raw = await readJsonFile(path);
	if (raw === null) return {}; // missing file -> empty
	if (typeof raw !== "object" || Array.isArray(raw)) {
		console.error(`[pi-local-llama] ${path} is not a JSON object; leaving it untouched`);
		return null;
	}
	return raw as Record<string, unknown>;
}

interface AddServerResult {
	changed: boolean;
	baseUrl?: string;
	reason: string;
}

async function addServer(path: string, input: string): Promise<AddServerResult> {
	const baseUrl = normalizeBaseUrl(input);
	if (!baseUrl) return { changed: false, reason: `invalid URL: "${input}"` };
	const cfg = await readServersObject(path);
	if (!cfg) return { changed: false, reason: `${path} is unreadable` };

	const servers = Array.isArray(cfg.servers) ? cfg.servers : [];
	if (servers.some((s) => storedServerUrl(s) === baseUrl)) {
		return { changed: false, baseUrl, reason: `already configured` };
	}
	cfg.servers = [...servers, baseUrl];
	await writeSettingsObject(path, cfg);
	return { changed: true, baseUrl, reason: "added" };
}

interface RemoveServerResult {
	changed: boolean;
	baseUrl?: string;
	reason: string;
}

async function removeServer(path: string, input: string): Promise<RemoveServerResult> {
	const baseUrl = normalizeBaseUrl(input);
	if (!baseUrl) return { changed: false, reason: `invalid URL: "${input}"` };
	const cfg = await readServersObject(path);
	if (!cfg) return { changed: false, reason: `${path} is unreadable` };

	const servers = Array.isArray(cfg.servers) ? cfg.servers : [];
	const next = servers.filter((s) => storedServerUrl(s) !== baseUrl);
	if (next.length === servers.length) {
		return { changed: false, baseUrl, reason: `not found` };
	}
	if (next.length === 0) delete cfg.servers;
	else cfg.servers = next;
	await writeSettingsObject(path, cfg);
	return { changed: true, baseUrl, reason: "removed" };
}

async function listServers(path: string): Promise<{ url: string; raw: unknown }[]> {
	const cfg = await readServersObject(path);
	const servers = Array.isArray(cfg?.servers) ? cfg.servers : [];
	return servers
		.map((s) => ({ raw: s, url: storedServerUrl(s) ?? String(s) }))
		.filter((s) => s.url);
}

// ---------------------------------------------------------------------------
// settings.json management (the `enabledModels` Ctrl+P pattern)
// ---------------------------------------------------------------------------

function settingsFilePath(scope: "global" | "project", cwd: string): string {
	return scope === "global"
		? join(getAgentDir(), "settings.json")
		: join(cwd, CONFIG_DIR_NAME, "settings.json");
}

async function readSettingsObject(path: string): Promise<Record<string, unknown> | null> {
	const raw = await readJsonFile(path);
	if (raw === null) return {}; // missing file -> treat as empty
	if (typeof raw !== "object" || Array.isArray(raw)) {
		console.error(`[pi-local-llama] ${path} is not a JSON object; leaving it untouched`);
		return null; // unreadable/invalid -> refuse to write
	}
	return raw as Record<string, unknown>;
}

/** Write atomically (tmp + rename) to reduce races with pi's own settings writes. */
async function writeSettingsObject(path: string, obj: Record<string, unknown>): Promise<void> {
	const text = `${JSON.stringify(obj, null, 2)}\n`;
	const tmp = `${path}.tmp`;
	await writeFile(tmp, text, "utf8");
	await rename(tmp, path);
}

type ScopeMode = "always" | "if-empty";

interface EnsureResult {
	changed: boolean;
	reason: string;
}

async function ensureScopePattern(path: string, pattern: string, mode: ScopeMode): Promise<EnsureResult> {
	const settings = await readSettingsObject(path);
	if (!settings) return { changed: false, reason: "settings unreadable" };

	const cur = settings.enabledModels;
	if (Array.isArray(cur) && cur.some((p) => typeof p === "string" && p === pattern)) {
		return { changed: false, reason: "already present" };
	}
	if (mode === "if-empty" && Array.isArray(cur) && cur.length > 0) {
		return { changed: false, reason: "user has existing enabledModels" };
	}

	const strings = (Array.isArray(cur) ? cur : []).filter((p): p is string => typeof p === "string");
	settings.enabledModels = [...strings, pattern];
	await writeSettingsObject(path, settings);
	return { changed: true, reason: "added" };
}

interface RemoveResult {
	changed: boolean;
	remaining: number;
}

async function removeScopePattern(path: string, pattern: string): Promise<RemoveResult> {
	const settings = await readSettingsObject(path);
	if (!settings) return { changed: false, remaining: 0 };

	const cur = settings.enabledModels;
	if (!Array.isArray(cur) || !cur.some((p) => typeof p === "string" && p === pattern)) {
		return { changed: false, remaining: Array.isArray(cur) ? cur.length : 0 };
	}

	const next = cur.filter((p) => typeof p === "string" && p !== pattern);
	if (next.length === 0) delete settings.enabledModels;
	else settings.enabledModels = next;
	await writeSettingsObject(path, settings);
	return { changed: true, remaining: next.length };
}

async function scopePatternStatus(path: string, pattern: string): Promise<{ present: boolean; enabledModels: string[] }> {
	const settings = await readSettingsObject(path);
	const cur = settings?.enabledModels;
	const arr = Array.isArray(cur) ? cur.filter((p): p is string => typeof p === "string") : [];
	return { present: arr.includes(pattern), enabledModels: arr };
}

// ---------------------------------------------------------------------------
// Server normalization
// ---------------------------------------------------------------------------

function normalizeServer(server: ServerConfig): NormalizedServer {
	let host = server.baseUrl;
	try {
		host = new URL(server.baseUrl).host;
	} catch {
		/* fall back to raw baseUrl */
	}

	return {
		...server,
		modelsUrl: `${server.baseUrl}/models`,
		providerName: toProviderName(server.provider, host),
		displayName: server.name ?? host,
	};
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

function timeoutSignal(ms: number): AbortSignal {
	if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
		return AbortSignal.timeout(ms);
	}
	const controller = new AbortController();
	setTimeout(() => controller.abort(), ms);
	return controller.signal;
}

async function fetchWithTimeout(
	url: string,
	init: RequestInit,
	timeoutMs: number,
): Promise<Response> {
	return fetch(url, { ...init, signal: timeoutSignal(timeoutMs) });
}

async function discoverModels(server: NormalizedServer, timeoutMs: number): Promise<DiscoverResult> {
	try {
		const headers: Record<string, string> = { accept: "application/json" };
		if (server.apiKey) headers.authorization = `Bearer ${server.apiKey}`;

		const res = await fetchWithTimeout(server.modelsUrl, { method: "GET", headers }, timeoutMs);
		if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };

		const json: unknown = await res.json().catch(() => null);
		let list: unknown[] = [];
		if (Array.isArray(json)) list = json;
		else if (json && typeof json === "object") {
			const obj = json as Record<string, unknown>;
			if (Array.isArray(obj.data)) list = obj.data;
			else if (Array.isArray(obj.models)) list = obj.models;
		}

		const models = list.filter(
			(m): m is RawModel =>
				!!m && typeof m === "object" && (typeof (m as RawModel).id === "string" || typeof (m as RawModel).name === "string"),
		);
		return { ok: true, models };
	} catch (e) {
		return { ok: false, error: errMsg(e) };
	}
}

/**
 * Best-effort GET that returns parsed JSON, or null on any failure (network
 * error, non-2xx, or unparseable body). Used for optional llama-server probes
 * where a missing endpoint is normal and must NOT surface as an error.
 */
async function getJson(url: string, timeoutMs: number, apiKey?: string): Promise<unknown | null> {
	try {
		const headers: Record<string, string> = { accept: "application/json" };
		if (apiKey) headers.authorization = `Bearer ${apiKey}`;
		const res = await fetchWithTimeout(url, { method: "GET", headers }, timeoutMs);
		if (!res.ok) return null;
		return await res.json().catch(() => null);
	} catch {
		return null;
	}
}

function readPositiveInt(v: unknown): number | undefined {
	return typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : undefined;
}

/** Pull the per-slot running context out of llama-server's /props payload. */
function ctxFromProps(props: unknown): number | undefined {
	if (!props || typeof props !== "object") return undefined;
	const o = props as Record<string, unknown>;
	const dgs = o.default_generation_settings;
	if (dgs && typeof dgs === "object") {
		const n = readPositiveInt((dgs as Record<string, unknown>).n_ctx);
		if (n) return n;
	}
	// Alternate / newer top-level fields.
	return readPositiveInt(o.n_ctx_per_slot) ?? readPositiveInt(o.n_ctx);
}

/** Pull the per-slot running context out of llama-server's /slots payload (max across slots). */
function ctxFromSlots(slots: unknown): number | undefined {
	if (!Array.isArray(slots)) return undefined;
	let max: number | undefined;
	for (const s of slots) {
		if (s && typeof s === "object") {
			const n = readPositiveInt((s as Record<string, unknown>).n_ctx);
			if (n !== undefined) max = max === undefined ? n : Math.max(max, n);
		}
	}
	return max;
}

/**
 * Discover the LIVE context window for a server — the value it was actually
 * launched with (llama-server's `-c`, divided across slots), reflecting real
 * VRAM / launch constraints rather than the model's architectural max.
 *
 * Probes, in order: llama-server `/props` (`default_generation_settings.n_ctx`,
 * or top-level `n_ctx` / `n_ctx_per_slot`), then `/slots` (per-slot `n_ctx`).
 * Returns undefined for servers that expose neither (e.g. plain Ollama, vLLM
 * without these endpoints) — callers then fall back to per-model fields.
 */
async function discoverServerContext(
	server: NormalizedServer,
	timeoutMs: number,
): Promise<number | undefined> {
	const fromProps = ctxFromProps(await getJson(`${server.baseUrl}/props`, timeoutMs, server.apiKey));
	if (fromProps) return fromProps;
	return ctxFromSlots(await getJson(`${server.baseUrl}/slots`, timeoutMs, server.apiKey));
}

/** Read llama-server's `meta.n_ctx` (running context) from a /v1/models entry. */
function metaNctx(meta: unknown): number {
	if (meta && typeof meta === "object") {
		return readPositiveInt((meta as Record<string, unknown>).n_ctx) ?? 0;
	}
	return 0;
}

/**
 * Resolve a model's context window, preferring the most authoritative LIVE
 * source: server-level running context (per-slot, from /props or /slots), then
 * the model payload's own fields (`meta.n_ctx`, `context_window`,
 * `context_length`), then a sane default.
 */
function resolveContextWindow(raw: RawModel, serverCtx: number | undefined): number {
	if (serverCtx && serverCtx > 0) return serverCtx;
	const fromModel =
		metaNctx(raw.meta) ||
		readPositiveInt(raw.context_window) ||
		readPositiveInt(raw.context_length) ||
		0;
	return fromModel > 0 ? fromModel : DEFAULT_CONTEXT_WINDOW;
}

function toModelDefinition(raw: RawModel, server: NormalizedServer, serverCtx: number | undefined) {
	const id = String(raw.id ?? raw.name ?? "").trim();
	const name = String(raw.name ?? id);
	const input = server.input ?? ["text"];
	return {
		id,
		name: name || id,
		reasoning: false,
		input,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: resolveContextWindow(raw, serverCtx),
		maxTokens: num(raw.max_tokens, 8192),
		compat: {
			// Safe defaults for OpenAI-compatible local servers (Ollama, vLLM,
			// SGLang, llama.cpp, LM Studio). See docs/models.md.
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			...(server.compat ?? {}),
		},
	};
}

function buildProviderConfig(
	server: NormalizedServer,
	models: RawModel[],
	serverCtx: number | undefined,
) {
	return {
		name: server.displayName,
		baseUrl: server.baseUrl,
		apiKey: server.apiKey ?? "local", // local servers ignore it, but pi needs *some* auth
		api: "openai-completions" as const,
		models: models.map((m) => toModelDefinition(m, server, serverCtx)),
	};
}

function modelIdsKey(models: RawModel[]): string {
	return models
		.map((m) => String(m.id ?? m.name ?? ""))
		.filter(Boolean)
		.sort()
		.join("\n");
}

/**
 * Stable signature of the provider config we would register: sorted
 * `id:contextWindow:maxTokens` per model. Compared across polls so we
 * re-register not just when the model id set changes, but also when a server's
 * live context window changes (e.g. llama-server relaunched with a different
 * `-c`, or a model's reported context changed). Without this, pi would keep a
 * stale contextWindow after a relaunch because the model ids are unchanged.
 */
function providerSignature(
	models: { id: string; contextWindow: number; maxTokens: number }[],
): string {
	return models.map((m) => `${m.id}:${m.contextWindow}:${m.maxTokens}`).sort().join("|");
}

// ---------------------------------------------------------------------------
// Reconcile: bring registered providers in line with current discovery
// ---------------------------------------------------------------------------

interface UiLike {
	notify?(...a: unknown[]): void;
	setStatus?(...a: unknown[]): void;
}
interface CtxLike {
	ui?: UiLike;
	hasUI?: boolean;
}

async function refresh(
	pi: ExtensionAPI,
	config: Config,
	registered: Map<string, RegisteredEntry>,
	ctx?: CtxLike,
): Promise<RefreshSummary> {
	const summary: RefreshSummary = {
		added: [],
		changed: [],
		removed: [],
		up: [],
		down: [],
	};

	const normalized = config.servers.map(normalizeServer);
	const results = await Promise.all(
		normalized.map(async (server) => {
			// Fetch the model list and the live server context (llama-server's
			// /props or /slots) in parallel. serverCtx is best-effort: it is
			// undefined for servers that don't expose those endpoints.
			const [result, serverCtx] = await Promise.all([
				discoverModels(server, config.timeoutMs),
				discoverServerContext(server, config.timeoutMs),
			]);
			return { server, result, serverCtx };
		}),
	);

	const seen = new Set<string>();

	for (const { server, result, serverCtx } of results) {
		seen.add(server.providerName);

		if (result.ok) {
			const providerConfig = buildProviderConfig(server, result.models, serverCtx);
			const signature = providerSignature(providerConfig.models);
			const idsKey = modelIdsKey(result.models);
			const prev = registered.get(server.providerName);
			if (!prev) {
				pi.registerProvider(server.providerName, providerConfig);
				registered.set(server.providerName, { server, modelIdsKey: idsKey, signature });
				summary.added.push(server);
			} else if (prev.signature !== signature) {
				// Model set OR per-model fields (context window, max tokens)
				// changed since last poll — re-register to keep pi in sync.
				// This also covers relaunching llama-server with a different -c.
				pi.registerProvider(server.providerName, providerConfig);
				registered.set(server.providerName, { server, modelIdsKey: idsKey, signature });
				summary.changed.push(server);
			}
			summary.up.push({ server, count: result.models.length });
		} else {
			const prev = registered.get(server.providerName);
			if (prev) {
				pi.unregisterProvider(server.providerName);
				registered.delete(server.providerName);
				summary.removed.push({ server, reason: result.error });
			}
			summary.down.push({ server, error: result.error });
		}
	}

	// Drop providers whose servers vanished from the config entirely.
	for (const [name, entry] of registered) {
		if (!seen.has(name)) {
			pi.unregisterProvider(name);
			registered.delete(name);
			summary.removed.push({ server: entry.server, reason: "removed from config" });
		}
	}

	emitNotifications(ctx, config, summary);
	emitStatus(ctx, config, summary, registered);

	return summary;
}

function emitNotifications(ctx: CtxLike | undefined, config: Config, summary: RefreshSummary): void {
	if (!config.notify || !ctx?.hasUI || !ctx.ui?.notify) return;

	const lines: string[] = [];
	if (summary.added.length) {
		lines.push(`+ up: ${summary.added.map((s) => `${s.displayName} (${s.providerName})`).join(", ")}`);
	}
	if (summary.changed.length) {
		lines.push(`~ updated: ${summary.changed.map((s) => s.displayName).join(", ")}`);
	}
	if (summary.removed.length) {
		lines.push(`- down: ${summary.removed.map((r) => `${r.server.displayName} [${r.reason}]`).join(", ")}`);
	}
	if (!lines.length) return;

	const message = `local-llama: ${lines.join("  |  ")}`;
	const type = summary.removed.length ? "warning" : "info";
	ctx.ui.notify(message, type);
}

function emitStatus(
	ctx: CtxLike | undefined,
	config: Config,
	summary: RefreshSummary,
	registered: Map<string, RegisteredEntry>,
): void {
	if (!config.status || !ctx?.hasUI || !ctx.ui?.setStatus) return;

	if (registered.size === 0 && summary.down.length === 0) {
		ctx.ui.setStatus("local-llama", undefined);
		return;
	}

	const totalModels = [...registered.values()].reduce(
		(acc, e) => acc + e.modelIdsKey.split("\n").filter(Boolean).length,
		0,
	);
	const parts: string[] = [];
	if (registered.size) parts.push(`${registered.size} up (${totalModels} models)`);
	if (summary.down.length) parts.push(`${summary.down.length} down`);
	ctx.ui.setStatus("local-llama", `🦙 ${parts.join(", ")}`);
}

function emitStartupSummary(
	ctx: CtxLike | undefined,
	config: Config,
	registered: Map<string, RegisteredEntry>,
	down: { server: NormalizedServer; error: string }[],
): void {
	if (!config.notify || !ctx?.hasUI || !ctx.ui?.notify) return;
	const entries = [...registered.values()];
	if (!entries.length && !down.length) return;

	const totalModels = entries.reduce(
		(acc, e) => acc + e.modelIdsKey.split("\n").filter(Boolean).length,
		0,
	);
	const bits: string[] = [];
	if (entries.length) bits.push(`${entries.length} server(s) up, ${totalModels} model(s)`);
	if (down.length) bits.push(`${down.length} down`);
	ctx.ui.notify(`local-llama: ${bits.join("; ")}`, down.length ? "warning" : "info");
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default async function localLlamaExtension(pi: ExtensionAPI): Promise<void> {
	const baseConfig = mergeConfigs(await loadGlobalConfig(), loadEnvConfig());

	// Mutable runtime config (updated at session_start when project config is merged in).
	const state: { config: Config } = { config: baseConfig };

	// Providers we currently have registered (process-global ModelRegistry).
	const registered = new Map<string, RegisteredEntry>();

	let timer: ReturnType<typeof setInterval> | undefined;

	// One-time initial discovery so models are available immediately at
	// startup (and for `pi --list-models`).
	if (baseConfig.discoverAtStartup && baseConfig.servers.length) {
		await refresh(pi, baseConfig, registered);
	}

	pi.on("session_start", async (event, ctx) => {
		// Effective config = global + env + (trusted) project.
		state.config = await computeConfig(ctx);
		const config = state.config;

		// Best-effort "on install" pattern management: add the scope pattern to
		// settings.json the first time we see an empty `enabledModels`. Only on
		// real startup to avoid surprising users mid-session, and only writes
		// when the pattern is actually missing (idempotent).
		if (event.reason === "startup" && config.autoScopePattern) {
			const allowProject = config.scopeSettings !== "project" || ctx.isProjectTrusted();
			if (allowProject) {
				const path = settingsFilePath(config.scopeSettings, ctx.cwd);
				try {
					const res = await ensureScopePattern(path, config.scopePattern, "if-empty");
					if (res.changed && ctx.hasUI) {
						ctx.ui.notify(
							`local-llama: added "${config.scopePattern}" to ${path} (Ctrl+P cycling). Restart pi for it to take effect.`,
							"info",
						);
					}
				} catch (e) {
					console.error(`[pi-local-llama] failed to update ${path}: ${errMsg(e)}`);
				}
			}
		}

		// Immediate reconcile + startup notice.
		const summary = await refresh(pi, config, registered, ctx);
		if (ctx.hasUI) emitStartupSummary(ctx, config, registered, summary.down);

		if (timer) clearInterval(timer);
		if (config.servers.length && config.pollIntervalMs > 0) {
			timer = setInterval(async () => {
				try {
					// Read state.config (not a captured snapshot) so servers added/removed
					// via /local-llama-servers are picked up by the next poll.
					await refresh(pi, state.config, registered, ctx);
				} catch (e) {
					console.error(`[pi-local-llama] poll error: ${errMsg(e)}`);
				}
			}, config.pollIntervalMs);
		}
	});

	pi.on("session_shutdown", () => {
		if (timer) {
			clearInterval(timer);
			timer = undefined;
		}
		// Intentionally do NOT unregister providers here: "model removed when
		// server is killed" is handled by reachability polling. We also do NOT
		// remove the scope pattern here — it must persist between sessions so it
		// is present at the next startup's scope resolution.
	});

	// Manual rescan command.
	pi.registerCommand("local-llama-rescan", {
		description: "Re-scan configured local LLM servers and sync their models.",
		handler: async (_args, ctx) => {
			const config = state.config;
			const summary = await refresh(pi, config, registered, ctx);
			if (!ctx.hasUI) return;
			if (!config.servers.length) {
				ctx.ui.notify("local-llama: no servers configured", "warning");
				return;
			}
			ctx.ui.notify(
				`local-llama: scan complete — ${summary.up.length} up, ${summary.down.length} down`,
				summary.down.length ? "warning" : "info",
			);
		},
	});

	// Manual scope-pattern management (the reliable add/remove path).
	pi.registerCommand("local-llama-scope", {
		description: "Manage the local-models Ctrl+P pattern in settings.json. Usage: /local-llama-scope enable|disable|status",
		handler: async (args, ctx) => {
			const config = state.config;
			const sub = (args ?? "").trim().toLowerCase() || "status";
			const path = settingsFilePath(config.scopeSettings, ctx.cwd);

			if (!ctx.hasUI) return;

			try {
				if (sub === "enable") {
					const res = await ensureScopePattern(path, config.scopePattern, "always");
					ctx.ui.notify(
						res.changed
							? `local-llama: added "${config.scopePattern}" to ${path}. Restart pi for Ctrl+P.`
							: `local-llama: "${config.scopePattern}" already in ${path}.`,
						"info",
					);
				} else if (sub === "disable") {
					const res = await removeScopePattern(path, config.scopePattern);
					ctx.ui.notify(
						res.changed
							? `local-llama: removed "${config.scopePattern}" from ${path} (${res.remaining} pattern(s) left).`
							: `local-llama: "${config.scopePattern}" was not present in ${path}.`,
						"info",
					);
				} else if (sub === "status") {
					const s = await scopePatternStatus(path, config.scopePattern);
					const list = s.enabledModels.length ? s.enabledModels.join(", ") : "(none)";
					ctx.ui.notify(
						`local-llama: pattern "${config.scopePattern}" ${s.present ? "IS" : "is NOT"} in ${path}. enabledModels: ${list}`,
						"info",
					);
				} else {
					ctx.ui.notify(
						`local-llama: unknown subcommand "${sub}". Use enable, disable, or status.`,
						"warning",
					);
				}
			} catch (e) {
				ctx.ui.notify(`local-llama: failed to update ${path}: ${errMsg(e)}`, "error");
			}
		},
	});

	// Manage the polled server list (persisted to local-llama-servers.json).
	pi.registerCommand("local-llama-servers", {
		description:
			"Manage polled local LLM servers. Usage: /local-llama-servers add|remove|list [url] [--project]",
		handler: async (args, ctx) => {
			if (!ctx.hasUI) return;

			const tokens = (args ?? "").trim().split(/\s+/).filter(Boolean);
			const projectIdx = tokens.findIndex((t) => t === "--project");
			const scope: "global" | "project" = projectIdx !== -1 ? "project" : "global";
			const rest = tokens.filter((_, i) => i !== projectIdx);
			const sub = (rest[0] ?? "list").toLowerCase();
			const urlInput = rest.slice(1).join(" ").trim();

			if (scope === "project" && !ctx.isProjectTrusted()) {
				ctx.ui.notify("local-llama: project not trusted; cannot write project config.", "warning");
				return;
			}
			const path = serversConfigPath(scope, ctx.cwd);

			try {
				if (sub === "add") {
					if (!urlInput) {
						ctx.ui.notify("local-llama: usage: /local-llama-servers add <url> [--project]", "warning");
						return;
					}
					const res = await addServer(path, urlInput);
					if (res.changed) {
						ctx.ui.notify(`local-llama: added ${res.baseUrl} to ${path}`, "info");
						state.config = await computeConfig(ctx);
						await refresh(pi, state.config, registered, ctx); // notifies up/down
					} else {
						ctx.ui.notify(`local-llama: ${res.baseUrl ?? urlInput} ${res.reason}.`, "info");
					}
				} else if (sub === "remove" || sub === "rm") {
					if (!urlInput) {
						ctx.ui.notify("local-llama: usage: /local-llama-servers remove <url> [--project]", "warning");
						return;
					}
					const res = await removeServer(path, urlInput);
					if (res.changed) {
						ctx.ui.notify(`local-llama: removed ${res.baseUrl} from ${path}`, "info");
						state.config = await computeConfig(ctx);
						await refresh(pi, state.config, registered, ctx); // unregisters if needed
					} else {
						ctx.ui.notify(`local-llama: ${res.baseUrl ?? urlInput} ${res.reason}.`, "info");
					}
				} else if (sub === "list" || sub === "ls") {
					const entries = await listServers(path);
					if (!entries.length) {
						ctx.ui.notify(`local-llama: no servers configured in ${path}.`, "info");
						return;
					}
					const lines = entries.map((e) => {
						const sc = coerceServer(e.raw) ?? { baseUrl: e.url };
						const up = registered.has(normalizeServer(sc).providerName);
						return `${up ? "up  " : "down"}  ${e.url}`;
					});
					ctx.ui.notify(`local-llama (${path}):
${lines.join("\n")}`, "info");
				} else {
					ctx.ui.notify(
						`local-llama: unknown subcommand "${sub}". Use add, remove, or list.`,
						"warning",
					);
				}
			} catch (e) {
				ctx.ui.notify(`local-llama: failed to update ${path}: ${errMsg(e)}`, "error");
			}
		},
	});
}
