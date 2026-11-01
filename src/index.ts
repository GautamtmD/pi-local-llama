/**
 * pi-local-llama
 *
 * Auto-detects OpenAI-compatible local LLM servers (Ollama, LM Studio, vLLM,
 * llama.cpp, SGLang, etc.) and registers their models with pi.
 *
 *   - Reads a list of server base URLs from config.
 *   - Polls each server's `<baseUrl>/models` endpoint, reporting 401/403 as
 *     authentication required or rejected rather than an unreachable server.
 *   - Stores keys entered with `/local-llama-servers key <url>` in a separate,
 *     agent-global `local-llama-credentials.json` (plaintext, private where
 *     supported); reuses them for discovery and inference. Explicit `apiKey`
 *     in server config takes precedence. Pi's input dialog may show the key.
 *     `/local-llama-servers key-remove <url>` removes the saved key.
 *   - Warns on missing or rejected keys (401/403) without repeating every poll.
 *   - When a server is reachable, registers a provider so its models appear in
 *     `/model` and `/scoped-models`.
 *   - When the model list, a model's live context window, OR the key changes,
 *     re-registers to keep the provider up to date.
 *   - When a server is killed / unreachable, unregisters its provider.
 *   - When a `localhost` URL fails or answers with something that isn't a
 *     model list (e.g. a static file server on `[::]` shadowing a model
 *     server on `127.0.0.1`), retries `127.0.0.1` and `[::1]` and registers
 *     the provider with the address that answered (see loopbackAlternates).
 *   - Detects each server's real running context size (not a fixed 128k) by
 *     probing llama-server's `/props` and `/slots` (see resolveContextWindow).
 *   - Per-server `reasoning`, `maxTokens` and `thinkingLevelMap` options mark
 *     models as thinking models (sending reasoning_effort) and set output caps.
 *   - Optionally manages a scope pattern (see SCOPE_PATTERN) in `settings.json`
 *     (`enabledModels`) so detected models join Ctrl+P cycling.
 *   - Before each idle prompt (`input`, before authentication), recovers
 *     configured local providers removed during a server restart. If the
 *     selected model is no longer served, switches to the server's first
 *     model alphabetically. If its server is unavailable, falls back to another
 *     configured server: one serving the same model id, else the first
 *     reachable one (config order) and its first model alphabetically. Shows a
 *     temporary footer status (`autoSwitchModel`, default on). Preserves manual
 *     model selections made while checking servers.
 *   - After an auto-switch, rewrites exact `enabledModels` entries for the old
 *     model (global + trusted project settings.json) and a project-level
 *     default pointing at it, so new pi processes (e.g. sub-agents) start on
 *     the new model. Pi itself saves the global default.
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
 *   /local-llama-servers key|key-remove <url>             (saved API keys)
 *   /local-llama-scope   enable|disable|status              (Ctrl+P pattern)
 *   /local-llama-rescan                                     (rescan now)
 *
 * Note on install/uninstall: pi has NO install/uninstall lifecycle hooks for
 * extensions. "Add the pattern on install" is approximated by a one-time,
 * idempotent add on the first `session_start` (see `autoScopePattern`).
 * Removal is NOT automatic — use `/local-llama-scope disable` before
 * `pi remove`, since the extension can no longer run after it is uninstalled.
 */

import { mkdirSync, readFileSync, renameSync, rmdirSync, statSync, writeFileSync } from "node:fs";
import { readFile, rename, unlink, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
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
/** Footer status key and lifetime for the temporary "model switched" message. */
const SWITCH_STATUS_KEY = "local-llama-switch";
const SWITCH_STATUS_MS = 10000;
/** `:<level>` suffixes pi accepts on `enabledModels` entries. */
const THINKING_SUFFIXES = ["off", "minimal", "low", "medium", "high", "xhigh"];
/**
 * Pi guards settings.json with proper-lockfile: a `<file>.lock` directory,
 * considered stale after 10s. Pi gives up after ~200ms, so hold it briefly.
 */
const SETTINGS_LOCK_STALE_MS = 10000;
const SETTINGS_LOCK_ATTEMPTS = 50;
const SETTINGS_LOCK_DELAY_MS = 20;
/** File names under `<agentDir>` (global) or `<cwd>/.pi` (project). */
const SERVERS_FILE = "local-llama-servers.json";
const SETTINGS_FILE = "settings.json";
const CREDENTIALS_FILE = "local-llama-credentials.json";
const STATUS_KEY = "local-llama";

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
	reasoning?: boolean; // model thinks; enables pi's thinking level (sent as reasoning_effort)
	maxTokens?: number; // output cap, overrides the model's `max_tokens` and the 8192 default
	thinkingLevelMap?: Record<string, string | null>; // pi level -> server value; null hides a level
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
	autoSwitchModel: boolean; // switch off a local model its server no longer serves
}

interface NormalizedServer extends ServerConfig {
	providerName: string; // always starts with PROVIDER_PREFIX
	displayName: string;
}

type AuthKind = "missing" | "rejected";

interface RegisteredEntry {
	server: NormalizedServer;
	modelIds: string[]; // sorted model ids; empty if none
	signature: string; // ids + contextWindow + maxTokens; change triggers re-register
	apiKey: string; // compare in memory so key replacement refreshes inference credentials
	baseUrl: string; // effective URL registered with pi (may be a loopback fallback)
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
	baseUrl: string; // URL that actually answered (configured, or a loopback fallback)
	fallbackFrom?: string; // why the configured URL was not used, when it wasn't
}
interface DiscoverFail {
	ok: false;
	error: string;
	auth?: AuthKind;
}
type DiscoverResult = DiscoverOk | DiscoverFail;

interface RefreshSummary {
	added: NormalizedServer[];
	changed: NormalizedServer[];
	removed: { server: NormalizedServer; reason: string }[];
	up: { server: NormalizedServer; count: number }[];
	down: { server: NormalizedServer; error: string }[];
	auth: { server: NormalizedServer; kind: AuthKind }[];
	rerouted: { server: NormalizedServer; baseUrl: string; reason: string }[];
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
	autoSwitchModel: true,
};

function errMsg(e: unknown): string {
	if (e instanceof Error) return e.message;
	return String(e);
}

function readPositiveInt(v: unknown): number | undefined {
	return typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : undefined;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
	return !!v && typeof v === "object" && !Array.isArray(v);
}

/** `<agentDir>/<file>` (global) or `<cwd>/.pi/<file>` (project). */
function configFilePath(scope: "global" | "project", cwd: string, file: string): string {
	return scope === "global" ? join(getAgentDir(), file) : join(cwd, CONFIG_DIR_NAME, file);
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

/**
 * Read a JSON object file we may rewrite: `{}` when missing, null when it is
 * unreadable or not an object (callers must then refuse to write it).
 */
async function readJsonObject(path: string): Promise<Record<string, unknown> | null> {
	const raw = await readJsonFile(path);
	if (raw === null) return {};
	if (!isPlainObject(raw)) {
		console.error(`[pi-local-llama] ${path} is not a JSON object; leaving it untouched`);
		return null;
	}
	return raw;
}

/**
 * Write atomically (tmp + rename) so a concurrent reader — including pi's own
 * settings handling — never sees a half-written file.
 */
async function writeJsonObject(path: string, obj: Record<string, unknown>): Promise<void> {
	const tmp = `${path}.tmp`;
	await writeFile(tmp, `${JSON.stringify(obj, null, 2)}\n`, "utf8");
	await rename(tmp, path);
}

// ---------------------------------------------------------------------------
// Config loading & merging
// ---------------------------------------------------------------------------

/** Extract/normalize the baseUrl from a server entry (string or object). */
function serverEntryUrl(entry: unknown): string | undefined {
	if (typeof entry === "string") return normalizeBaseUrl(entry);
	if (entry && typeof entry === "object") {
		const obj = entry as Record<string, unknown>;
		return normalizeBaseUrl(String(obj.baseUrl ?? obj.url ?? ""));
	}
	return undefined;
}

function coerceServer(entry: unknown): ServerConfig | undefined {
	const baseUrl = serverEntryUrl(entry);
	if (!baseUrl) return undefined;
	if (typeof entry === "string") return { baseUrl };

	const obj = entry as Record<string, unknown>;
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
	if (typeof obj.reasoning === "boolean") server.reasoning = obj.reasoning;
	const maxTokens = readPositiveInt(obj.maxTokens);
	if (maxTokens) server.maxTokens = maxTokens;
	if (obj.thinkingLevelMap && typeof obj.thinkingLevelMap === "object") {
		server.thinkingLevelMap = { ...(obj.thinkingLevelMap as Record<string, string | null>) };
	}
	return server;
}

function coerceServers(list: unknown[]): ServerConfig[] {
	return list.map(coerceServer).filter((s): s is ServerConfig => s !== undefined);
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
	if (typeof obj.autoSwitchModel === "boolean") cfg.autoSwitchModel = obj.autoSwitchModel;
	if (typeof obj.scopePattern === "string" && obj.scopePattern.trim()) cfg.scopePattern = obj.scopePattern.trim();
	if (obj.scopeSettings === "project") cfg.scopeSettings = "project";
	else if (obj.scopeSettings === "global") cfg.scopeSettings = "global";

	cfg.servers = coerceServers(Array.isArray(obj.servers) ? obj.servers : []);
	return cfg;
}

/**
 * Merge configs: servers accumulate (later `override` wins per baseUrl).
 * Scalars come from `override` only if it contributed a server; either side
 * can disable `discoverAtStartup`.
 */
function mergeConfigs(base: Config, override: Config): Config {
	const byUrl = new Map<string, ServerConfig>();
	for (const s of base.servers) byUrl.set(s.baseUrl, s);
	for (const s of override.servers) byUrl.set(s.baseUrl, s);
	const scalars = override.servers.length > 0 ? override : base;
	return {
		...scalars,
		servers: [...byUrl.values()],
		discoverAtStartup: base.discoverAtStartup && override.discoverAtStartup,
	};
}

/** Config file for the global (no cwd needed) or a project scope. */
async function loadConfigFile(scope: "global" | "project", cwd = ""): Promise<Config> {
	return coerceConfig(await readJsonFile(configFilePath(scope, cwd, SERVERS_FILE)));
}

function loadEnvConfig(): Config {
	const urls = (process.env.PI_LOCAL_SERVERS ?? "").split(",").map((s) => s.trim()).filter(Boolean);
	return { ...DEFAULTS, servers: coerceServers(urls) };
}

/** Recompute the effective config from global + env + (trusted) project sources. */
async function computeConfig(ctx: { cwd: string; isProjectTrusted(): boolean }): Promise<Config> {
	let cfg = mergeConfigs(await loadConfigFile("global"), loadEnvConfig());
	if (ctx.isProjectTrusted()) cfg = mergeConfigs(cfg, await loadConfigFile("project", ctx.cwd));
	return cfg;
}

// ---------------------------------------------------------------------------
// Saved API keys (agent-global, never stored in project config by the command)
// ---------------------------------------------------------------------------

function credentialsPath(): string {
	return configFilePath("global", "", CREDENTIALS_FILE);
}

/** Return null for a broken/unreadable store so commands never overwrite it. */
async function readCredentials(): Promise<Record<string, string> | null> {
	const path = credentialsPath();
	let text: string;
	try {
		text = await readFile(path, "utf8");
	} catch (e) {
		if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return {};
		console.error(`[pi-local-llama] failed to read ${path}: ${errMsg(e)}`);
		return null;
	}
	let raw: unknown;
	try {
		raw = JSON.parse(text);
	} catch {
		// SyntaxError messages can include fragments of the secret file.
		console.error(`[pi-local-llama] invalid JSON in ${path}; leaving it untouched`);
		return null;
	}
	if (!raw || typeof raw !== "object" || Array.isArray(raw) ||
		Object.values(raw).some((v) => typeof v !== "string")) {
		console.error(`[pi-local-llama] ${path} is not a URL-to-key JSON object; leaving it untouched`);
		return null;
	}
	return raw as Record<string, string>;
}

/** Write a private, uniquely named temp file then atomically replace the store. */
async function writeCredentials(keys: Record<string, string>): Promise<void> {
	const path = credentialsPath();
	const tmp = `${path}.${randomUUID()}.tmp`;
	try {
		await writeFile(tmp, `${JSON.stringify(keys, null, 2)}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
		await rename(tmp, path);
	} finally {
		await unlink(tmp).catch((e: NodeJS.ErrnoException) => {
			if (e.code !== "ENOENT") console.error(`[pi-local-llama] failed to remove ${tmp}: ${errMsg(e)}`);
		});
	}
}

async function updateCredential(baseUrl: string, key: string | undefined): Promise<void> {
	const keys = await readCredentials();
	if (!keys) throw new Error(`credentials file is unreadable: ${credentialsPath()}`);
	if (key === undefined) delete keys[baseUrl];
	else keys[baseUrl] = key;
	await writeCredentials(keys);
}

/** An explicit per-server key takes precedence over a saved key. */
function withCredential(server: ServerConfig, keys: Record<string, string>): ServerConfig {
	return { ...server, apiKey: server.apiKey?.trim() || keys[server.baseUrl] || undefined };
}

// ---------------------------------------------------------------------------
// local-llama-servers.json management (the polled server list)
// ---------------------------------------------------------------------------

interface ServerEditResult {
	changed: boolean;
	baseUrl?: string;
	reason: string;
}

/**
 * Read the server list at `path`, let `edit` return the new list (or a
 * reason it is unchanged), and write it back. An empty list drops the key.
 */
async function editServerList(
	path: string,
	input: string,
	edit: (servers: unknown[], baseUrl: string) => { next: unknown[]; reason: string } | string,
): Promise<ServerEditResult> {
	const baseUrl = normalizeBaseUrl(input);
	if (!baseUrl) return { changed: false, reason: `invalid URL: "${input}"` };
	const cfg = await readJsonObject(path);
	if (!cfg) return { changed: false, reason: `${path} is unreadable` };

	const result = edit(Array.isArray(cfg.servers) ? cfg.servers : [], baseUrl);
	if (typeof result === "string") return { changed: false, baseUrl, reason: result };
	if (result.next.length === 0) delete cfg.servers;
	else cfg.servers = result.next;
	await writeJsonObject(path, cfg);
	return { changed: true, baseUrl, reason: result.reason };
}

function addServer(path: string, input: string): Promise<ServerEditResult> {
	return editServerList(path, input, (servers, baseUrl) =>
		servers.some((s) => serverEntryUrl(s) === baseUrl)
			? "already configured"
			: { next: [...servers, baseUrl], reason: "added" });
}

function removeServer(path: string, input: string): Promise<ServerEditResult> {
	return editServerList(path, input, (servers, baseUrl) => {
		const next = servers.filter((s) => serverEntryUrl(s) !== baseUrl);
		return next.length === servers.length ? "not found" : { next, reason: "removed" };
	});
}

async function listServers(path: string): Promise<{ url: string; raw: unknown }[]> {
	const cfg = await readJsonObject(path);
	const servers = Array.isArray(cfg?.servers) ? cfg.servers : [];
	return servers
		.map((s) => ({ raw: s, url: serverEntryUrl(s) ?? String(s) }))
		.filter((s) => s.url);
}

// ---------------------------------------------------------------------------
// settings.json management (the `enabledModels` Ctrl+P pattern)
// ---------------------------------------------------------------------------

function settingsFilePath(scope: "global" | "project", cwd: string): string {
	return configFilePath(scope, cwd, SETTINGS_FILE);
}

/** The string entries of a settings object's `enabledModels`. */
function enabledModelStrings(settings: Record<string, unknown> | null): string[] {
	const cur = settings?.enabledModels;
	return Array.isArray(cur) ? cur.filter((p): p is string => typeof p === "string") : [];
}

type ScopeMode = "always" | "if-empty";

interface EnsureResult {
	changed: boolean;
	reason: string;
}

async function ensureScopePattern(path: string, pattern: string, mode: ScopeMode): Promise<EnsureResult> {
	const settings = await readJsonObject(path);
	if (!settings) return { changed: false, reason: "settings unreadable" };

	const cur = settings.enabledModels;
	const strings = enabledModelStrings(settings);
	if (strings.includes(pattern)) return { changed: false, reason: "already present" };
	if (mode === "if-empty" && Array.isArray(cur) && cur.length > 0) {
		return { changed: false, reason: "user has existing enabledModels" };
	}

	settings.enabledModels = [...strings, pattern];
	await writeJsonObject(path, settings);
	return { changed: true, reason: "added" };
}

interface RemoveResult {
	changed: boolean;
	remaining: number;
}

async function removeScopePattern(path: string, pattern: string): Promise<RemoveResult> {
	const settings = await readJsonObject(path);
	if (!settings) return { changed: false, remaining: 0 };

	const cur = settings.enabledModels;
	const strings = enabledModelStrings(settings);
	if (!strings.includes(pattern)) {
		return { changed: false, remaining: Array.isArray(cur) ? cur.length : 0 };
	}

	const next = strings.filter((p) => p !== pattern);
	if (next.length === 0) delete settings.enabledModels;
	else settings.enabledModels = next;
	await writeJsonObject(path, settings);
	return { changed: true, remaining: next.length };
}

async function scopePatternStatus(path: string, pattern: string): Promise<{ present: boolean; enabledModels: string[] }> {
	const enabledModels = enabledModelStrings(await readJsonObject(path));
	return { present: enabledModels.includes(pattern), enabledModels };
}

// ---------------------------------------------------------------------------
// settings.json sync after an auto-switch (scoped models + project default)
// ---------------------------------------------------------------------------

interface ModelRef {
	provider: string;
	id: string;
}

/**
 * Take pi's proper-lockfile-compatible `<path>.lock` directory around `fn`.
 * The locked section is synchronous on purpose: pi retries its own lock in a
 * blocking loop, so awaiting while holding it would starve pi's write in this
 * process, and a section this short cannot outlive the stale threshold.
 */
async function withSettingsLock<T>(path: string, fn: () => T): Promise<T> {
	const lock = `${path}.lock`;
	for (let attempt = 1; ; attempt++) {
		try {
			mkdirSync(lock);
		} catch (e) {
			if ((e as NodeJS.ErrnoException)?.code !== "EEXIST") throw e;
			let removed = false;
			try {
				if (Date.now() - statSync(lock).mtimeMs > SETTINGS_LOCK_STALE_MS) {
					rmdirSync(lock);
					removed = true;
				}
			} catch {
				/* lock released meanwhile, or a stale lock we cannot remove */
			}
			if (attempt >= SETTINGS_LOCK_ATTEMPTS) throw new Error(`settings lock is busy: ${lock}`);
			if (!removed) await new Promise((resolve) => setTimeout(resolve, SETTINGS_LOCK_DELAY_MS));
			continue;
		}
		try {
			return fn();
		} finally {
			try {
				rmdirSync(lock);
			} catch {
				/* already gone */
			}
		}
	}
}

/** Synchronous readJsonObject() for use while holding the settings lock. */
function readJsonObjectSync(path: string): Record<string, unknown> | null {
	let raw: unknown;
	try {
		raw = JSON.parse(readFileSync(path, "utf8"));
	} catch (e) {
		if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return {};
		console.error(`[pi-local-llama] failed to read ${path}: ${errMsg(e)}`);
		return null;
	}
	return isPlainObject(raw) ? raw : null;
}

/**
 * The `:<level>` suffix of an exact `enabledModels` entry for `ref`, or
 * undefined. `<id>:<level>` that is itself a served model id names that
 * model, as pi's exact-match-first resolution would.
 */
function scopeEntrySuffix(entry: string, ref: ModelRef, isModel: (ref: ModelRef) => boolean): string | undefined {
	const exact = `${ref.provider}/${ref.id}`.toLowerCase();
	const lower = entry.toLowerCase();
	if (lower === exact) return "";
	if (!lower.startsWith(`${exact}:`)) return undefined;
	const level = entry.slice(exact.length + 1);
	if (!THINKING_SUFFIXES.includes(level)) return undefined;
	return isModel({ provider: ref.provider, id: entry.slice(ref.provider.length + 1) }) ? undefined : `:${level}`;
}

/**
 * Point settings at the model the extension switched to, so a fresh `pi`
 * (e.g. a sub-agent) starts on it:
 *   - exact `enabledModels` entries for `from` gain an entry for `to` (same
 *     `:<level>` suffix). A same-server switch replaces the stale entry; a
 *     cross-server switch keeps it, since that server may come back. Globs
 *     are left alone — they already match whatever is registered.
 *   - `defaultProvider`/`defaultModel` fields pointing at `from` are
 *     rewritten. Pi merges the two independently and its setModel only updates
 *     the global file, so a project file setting either one overrides it.
 */
async function syncSwitchedModel(
	path: string,
	from: ModelRef,
	to: ModelRef,
	replace: boolean,
	isModel: (ref: ModelRef) => boolean,
): Promise<boolean> {
	const update = (settings: Record<string, unknown>): boolean => {
		let changed = false;
		const cur = settings.enabledModels;
		if (Array.isArray(cur)) {
			const suffixes = cur.map((e) => typeof e === "string" ? scopeEntrySuffix(e, from, isModel) : undefined);
			// Dedupe against entries that survive, not ones being replaced.
			const kept = cur.filter((_, i) => suffixes[i] === undefined || !replace);
			const next: unknown[] = [];
			const has = (s: string) => [...kept, ...next].some((p) => typeof p === "string" && p.toLowerCase() === s.toLowerCase());
			cur.forEach((entry, i) => {
				const suffix = suffixes[i];
				if (suffix === undefined) {
					next.push(entry);
					return;
				}
				const add = !has(`${to.provider}/${to.id}${suffix}`);
				if (add) next.push(`${to.provider}/${to.id}${suffix}`);
				if (!replace) next.push(entry);
				if (add || replace) changed = true;
			});
			if (changed) settings.enabledModels = next;
		}
		const hasProvider = typeof settings.defaultProvider === "string";
		const hasModel = typeof settings.defaultModel === "string";
		if ((hasProvider || hasModel) &&
			(!hasProvider || settings.defaultProvider === from.provider) &&
			(!hasModel || settings.defaultModel === from.id)) {
			if (hasProvider && from.provider !== to.provider) {
				settings.defaultProvider = to.provider;
				changed = true;
			}
			if (hasModel && from.id !== to.id) {
				settings.defaultModel = to.id;
				changed = true;
			}
		}
		return changed;
	};

	// Cheap unlocked check first: most settings files need no change.
	const preview = await readJsonObject(path);
	if (!preview || !update(preview)) return false;
	return withSettingsLock(path, () => {
		const settings = readJsonObjectSync(path);
		if (!settings || !update(settings)) return false;
		const tmp = `${path}.tmp`;
		writeFileSync(tmp, `${JSON.stringify(settings, null, 2)}\n`, "utf8");
		renameSync(tmp, path);
		return true;
	});
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
		providerName: toProviderName(server.provider, host),
		displayName: server.name ?? host,
	};
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/** GET a JSON endpoint with an optional Bearer key and a hard timeout. */
function fetchJson(url: string, apiKey: string | undefined, timeoutMs: number): Promise<Response> {
	const headers: Record<string, string> = { accept: "application/json" };
	if (apiKey) headers.authorization = `Bearer ${apiKey}`;
	return fetch(url, { method: "GET", headers, signal: AbortSignal.timeout(timeoutMs) });
}

async function discoverModelsAt(baseUrl: string, apiKey: string | undefined, timeoutMs: number): Promise<DiscoverResult> {
	try {
		const res = await fetchJson(`${baseUrl}/models`, apiKey, timeoutMs);
		if (!res.ok) return {
			ok: false,
			error: `HTTP ${res.status}`,
			auth: res.status === 401 || res.status === 403
				? (apiKey ? "rejected" : "missing") : undefined,
		};

		const json: unknown = await res.json().catch(() => null);
		let list: unknown[] | undefined;
		if (Array.isArray(json)) list = json;
		else if (json && typeof json === "object") {
			const obj = json as Record<string, unknown>;
			if (Array.isArray(obj.data)) list = obj.data;
			else if (Array.isArray(obj.models)) list = obj.models;
		}
		// A 2xx that isn't a model list (e.g. an HTML page from some other web
		// server on the same port) is not an OpenAI-compatible server.
		if (!list) return { ok: false, error: "response is not an OpenAI-compatible model list" };

		const models = list.filter(
			(m): m is RawModel =>
				!!m && typeof m === "object" && (typeof (m as RawModel).id === "string" || typeof (m as RawModel).name === "string"),
		);
		return { ok: true, models, baseUrl };
	} catch (e) {
		return { ok: false, error: errMsg(e) };
	}
}

/**
 * Explicit loopback addresses to retry when a `localhost` URL fails. Node (and
 * pi's HTTP client) try `::1` before `127.0.0.1`, so an unrelated process bound
 * to `[::]:<port>` (e.g. `python -m http.server`) silently shadows a model
 * server bound only to `127.0.0.1:<port>` — and vice versa.
 */
function loopbackAlternates(baseUrl: string): string[] {
	let url: URL;
	try {
		url = new URL(baseUrl);
	} catch {
		return [];
	}
	if (url.hostname.toLowerCase() !== "localhost") return [];
	return ["127.0.0.1", "[::1]"].map((host) => {
		const alt = new URL(url.href);
		alt.hostname = host;
		return alt.href.replace(/\/+$/, "");
	});
}

async function discoverModels(server: NormalizedServer, timeoutMs: number): Promise<DiscoverResult> {
	const primary = await discoverModelsAt(server.baseUrl, server.apiKey, timeoutMs);
	if (primary.ok) return primary;
	const alternates = loopbackAlternates(server.baseUrl);
	if (!alternates.length) return primary;
	const results = await Promise.all(alternates.map((url) => discoverModelsAt(url, server.apiKey, timeoutMs)));
	const found = results.find((r): r is DiscoverOk => r.ok);
	return found ? { ...found, fallbackFrom: primary.error } : primary;
}

/**
 * Best-effort GET that returns parsed JSON, or null on any failure (network
 * error, non-2xx, or unparseable body). Used for optional llama-server probes
 * where a missing endpoint is normal and must NOT surface as an error.
 */
async function getJson(url: string, timeoutMs: number, apiKey?: string): Promise<unknown | null> {
	try {
		const res = await fetchJson(url, apiKey, timeoutMs);
		if (!res.ok) return null;
		return await res.json().catch(() => null);
	} catch {
		return null;
	}
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
	baseUrl: string,
	apiKey: string | undefined,
	timeoutMs: number,
): Promise<number | undefined> {
	const fromProps = ctxFromProps(await getJson(`${baseUrl}/props`, timeoutMs, apiKey));
	if (fromProps) return fromProps;
	return ctxFromSlots(await getJson(`${baseUrl}/slots`, timeoutMs, apiKey));
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
	const reasoning = server.reasoning ?? false;
	return {
		id,
		name: name || id,
		reasoning,
		input,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: resolveContextWindow(raw, serverCtx),
		maxTokens: server.maxTokens ?? readPositiveInt(raw.max_tokens) ?? 8192,
		...(server.thinkingLevelMap ? { thinkingLevelMap: server.thinkingLevelMap } : {}),
		compat: {
			// Safe defaults for OpenAI-compatible local servers (Ollama, vLLM,
			// SGLang, llama.cpp, LM Studio). See docs/models.md. A server marked
			// `reasoning` gets OpenAI-style reasoning_effort unless compat says otherwise.
			supportsDeveloperRole: false,
			supportsReasoningEffort: reasoning,
			...(server.compat ?? {}),
		},
	};
}

function buildProviderConfig(
	server: NormalizedServer,
	models: RawModel[],
	serverCtx: number | undefined,
	baseUrl: string = server.baseUrl,
) {
	return {
		name: server.displayName,
		baseUrl,
		apiKey: server.apiKey ?? "local", // local servers ignore it, but pi needs *some* auth
		api: "openai-completions" as const,
		models: models.map((m) => toModelDefinition(m, server, serverCtx)),
	};
}

/** Sorted, non-empty model ids from a /models payload. */
function modelIdList(models: RawModel[]): string[] {
	return models
		.map((m) => String(m.id ?? m.name ?? ""))
		.filter(Boolean)
		.sort();
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

/** Short state labels, shared by the footer, notices and `/local-llama-servers list`. */
const AUTH_LABEL: Record<AuthKind, string> = { missing: "needs API key", rejected: "key rejected" };
/** Why a provider was unregistered after an auth failure. */
const AUTH_REMOVED_REASON: Record<AuthKind, string> = { missing: "API key required", rejected: "API key rejected" };

/** e.g. ["1 needs API key", "2 key rejected"], omitting zero counts. */
function authCountParts(auth: { kind: AuthKind }[]): string[] {
	const parts: string[] = [];
	for (const kind of ["missing", "rejected"] as const) {
		const count = auth.filter((a) => a.kind === kind).length;
		if (count) parts.push(`${count} ${AUTH_LABEL[kind]}`);
	}
	return parts;
}

function totalModelCount(registered: Map<string, RegisteredEntry>): number {
	let total = 0;
	for (const e of registered.values()) total += e.modelIds.length;
	return total;
}

async function refresh(
	pi: ExtensionAPI,
	config: Config,
	registered: Map<string, RegisteredEntry>,
	authStates: Map<string, AuthKind>,
	ctx?: CtxLike,
	keep?: string, // provider left registered even if down (the next poll removes it)
): Promise<RefreshSummary> {
	const summary: RefreshSummary = {
		added: [],
		changed: [],
		removed: [],
		up: [],
		down: [],
		auth: [],
		rerouted: [],
	};

	const keys = await readCredentials();
	const normalized = config.servers.map((s) => normalizeServer(withCredential(s, keys ?? {})));
	const results = await Promise.all(
		normalized.map(async (server) => {
			// Fetch the model list and the live server context (llama-server's
			// /props or /slots) in parallel. serverCtx is best-effort: it is
			// undefined for servers that don't expose those endpoints.
			let [result, serverCtx] = await Promise.all([
				discoverModels(server, config.timeoutMs),
				discoverServerContext(server.baseUrl, server.apiKey, config.timeoutMs),
			]);
			// The models answered on a loopback fallback: the context probe above
			// hit whatever is shadowing the configured URL, so ask the real server.
			if (result.ok && result.baseUrl !== server.baseUrl) {
				serverCtx = await discoverServerContext(result.baseUrl, server.apiKey, config.timeoutMs);
			}
			return { server, result, serverCtx };
		}),
	);

	const seen = new Set<string>();

	for (const { server, result, serverCtx } of results) {
		seen.add(server.providerName);

		if (result.ok) {
			const providerConfig = buildProviderConfig(server, result.models, serverCtx, result.baseUrl);
			const signature = providerSignature(providerConfig.models);
			const prev = registered.get(server.providerName);
			const entry: RegisteredEntry = {
				server,
				modelIds: modelIdList(result.models),
				signature,
				apiKey: providerConfig.apiKey,
				baseUrl: result.baseUrl,
			};
			if (!prev) {
				pi.registerProvider(server.providerName, providerConfig);
				registered.set(server.providerName, entry);
				summary.added.push(server);
			} else if (prev.signature !== signature || prev.apiKey !== providerConfig.apiKey || prev.baseUrl !== result.baseUrl) {
				// Model set OR per-model fields (context window, max tokens)
				// changed since last poll — re-register to keep pi in sync.
				// This also covers relaunching llama-server with a different -c,
				// and moving to/from a loopback fallback URL.
				pi.registerProvider(server.providerName, providerConfig);
				registered.set(server.providerName, entry);
				summary.changed.push(server);
			}
			if (result.fallbackFrom && prev?.baseUrl !== result.baseUrl) {
				summary.rerouted.push({ server, baseUrl: result.baseUrl, reason: result.fallbackFrom });
			}
			authStates.delete(server.providerName);
			summary.up.push({ server, count: result.models.length });
		} else {
			const prev = registered.get(server.providerName);
			if (prev && server.providerName !== keep) {
				pi.unregisterProvider(server.providerName);
				registered.delete(server.providerName);
				summary.removed.push({ server, reason: result.auth ? AUTH_REMOVED_REASON[result.auth] : result.error });
			}
			if (result.auth) {
				summary.auth.push({ server, kind: result.auth });
			} else {
				authStates.delete(server.providerName);
				summary.down.push({ server, error: result.error });
			}
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
	for (const name of authStates.keys()) if (!seen.has(name)) authStates.delete(name);

	emitNotifications(ctx, config, summary, authStates);
	emitStatus(ctx, config, summary, registered);

	return summary;
}

function emitNotifications(
	ctx: CtxLike | undefined,
	config: Config,
	summary: RefreshSummary,
	authStates: Map<string, AuthKind>,
): void {
	if (!config.notify || !ctx?.hasUI || !ctx.ui?.notify) return;

	const lines: string[] = [];
	if (summary.added.length) {
		lines.push(`+ up: ${summary.added.map((s) => `${s.displayName} (${s.providerName})`).join(", ")}`);
	}
	if (summary.changed.length) {
		lines.push(`~ updated: ${summary.changed.map((s) => s.displayName).join(", ")}`);
	}
	for (const { server, baseUrl, reason } of summary.rerouted) {
		lines.push(`${server.baseUrl} answered [${reason}] (another program on that port?); using ${baseUrl}`);
	}
	if (summary.removed.length) {
		lines.push(`- unavailable: ${summary.removed.map((r) => `${r.server.displayName} [${r.reason}]`).join(", ")}`);
	}
	const newAuth = summary.auth.filter(({ server, kind }) => authStates.get(server.providerName) !== kind);
	for (const { server, kind } of summary.auth) authStates.set(server.providerName, kind);
	for (const { server, kind } of newAuth) {
		lines.push(kind === "missing"
			? `${server.displayName} is reachable but requires an API key; use /local-llama-servers key ${server.baseUrl}`
			: `${server.displayName} rejected its API key; use /local-llama-servers key ${server.baseUrl}`);
	}
	if (!lines.length) return;

	const message = `local-llama: ${lines.join("  |  ")}`;
	const type = summary.removed.length || newAuth.length || summary.rerouted.length ? "warning" : "info";
	ctx.ui.notify(message, type);
}

function emitStatus(
	ctx: CtxLike | undefined,
	config: Config,
	summary: RefreshSummary,
	registered: Map<string, RegisteredEntry>,
): void {
	if (!config.status || !ctx?.hasUI || !ctx.ui?.setStatus) return;

	if (registered.size === 0 && summary.down.length === 0 && summary.auth.length === 0) {
		ctx.ui.setStatus(STATUS_KEY, undefined);
		return;
	}

	const parts: string[] = [];
	if (registered.size) parts.push(`${registered.size} up (${totalModelCount(registered)} models)`);
	if (summary.down.length) parts.push(`${summary.down.length} down`);
	parts.push(...authCountParts(summary.auth));
	ctx.ui.setStatus(STATUS_KEY, `🦙 ${parts.join(", ")}`);
}

function emitStartupSummary(
	ctx: CtxLike | undefined,
	config: Config,
	registered: Map<string, RegisteredEntry>,
	{ down, auth }: RefreshSummary,
): void {
	if (!config.notify || !ctx?.hasUI || !ctx.ui?.notify) return;
	if (!registered.size && !down.length && !auth.length) return;

	const bits: string[] = [];
	if (registered.size) bits.push(`${registered.size} server(s) up, ${totalModelCount(registered)} model(s)`);
	if (down.length) bits.push(`${down.length} down`);
	bits.push(...authCountParts(auth));
	const rerouted = [...registered.values()].filter((e) => e.baseUrl !== e.server.baseUrl);
	for (const e of rerouted) {
		bits.push(`${e.server.baseUrl} is answered by another program; using ${e.baseUrl}`);
	}
	ctx.ui.notify(`local-llama: ${bits.join("; ")}`, down.length || auth.length || rerouted.length ? "warning" : "info");
}

// ---------------------------------------------------------------------------
// Auto-switch target selection
// ---------------------------------------------------------------------------

interface SwitchTarget {
	entry: RegisteredEntry;
	id: string;
}

/**
 * Pick the model to switch to after reconciliation.
 *   - Own server live: its first model alphabetically, unless it still serves
 *     `current.id` (then nothing to do).
 *   - Own server down: another configured server serving the same id, else
 *     the first reachable one (config order) with its first model.
 */
function pickSwitchTarget(
	current: ModelRef,
	ownServerLive: boolean,
	servers: ServerConfig[],
	registered: Map<string, RegisteredEntry>,
): SwitchTarget | undefined {
	if (ownServerLive) {
		const own = registered.get(current.provider);
		if (!own?.modelIds.length || own.modelIds.includes(current.id)) return undefined;
		return { entry: own, id: own.modelIds[0] };
	}
	const others = servers
		.map((s) => registered.get(normalizeServer(s).providerName))
		.filter((e): e is RegisteredEntry => !!e && e.server.providerName !== current.provider && e.modelIds.length > 0);
	const same = others.find((e) => e.modelIds.includes(current.id));
	if (same) return { entry: same, id: current.id };
	return others.length ? { entry: others[0], id: others[0].modelIds[0] } : undefined;
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default async function localLlamaExtension(pi: ExtensionAPI): Promise<void> {
	const baseConfig = mergeConfigs(await loadConfigFile("global"), loadEnvConfig());

	// Mutable runtime config (updated at session_start when project config is merged in).
	const state: { config: Config } = { config: baseConfig };

	// Providers we currently have registered (process-global ModelRegistry).
	const registered = new Map<string, RegisteredEntry>();
	const authStates = new Map<string, AuthKind>();
	// A command can rescan while a poll is in flight; apply results in order.
	let pendingRefresh: Promise<void> = Promise.resolve();
	function runRefresh(config: Config, ctx?: CtxLike, keep?: string): Promise<RefreshSummary> {
		const run = pendingRefresh.then(() => refresh(pi, config, registered, authStates, ctx, keep));
		pendingRefresh = run.then(() => {}, () => {});
		return run;
	}

	let timer: ReturnType<typeof setInterval> | undefined;

	// One-time initial discovery so models are available immediately at
	// startup (and for `pi --list-models`).
	if (baseConfig.discoverAtStartup && baseConfig.servers.length) {
		await runRefresh(baseConfig);
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
		const summary = await runRefresh(config, ctx);
		emitStartupSummary(ctx, config, registered, summary);

		if (timer) clearInterval(timer);
		if (config.servers.length && config.pollIntervalMs > 0) {
			timer = setInterval(async () => {
				try {
					// Read state.config (not a captured snapshot) so servers added/removed
					// via /local-llama-servers are picked up by the next poll.
					await runRefresh(state.config, ctx);
				} catch (e) {
					console.error(`[pi-local-llama] poll error: ${errMsg(e)}`);
				}
			}, config.pollIntervalMs);
		}
	});

	// Input runs before pi validates auth (and compacts), so it can recover a
	// provider that polling removed during a restart. Do not change models for
	// input queued while an agent is already running.
	let switchStatusTimer: ReturnType<typeof setTimeout> | undefined;
	function clearSwitchStatusTimer(): void {
		if (switchStatusTimer) clearTimeout(switchStatusTimer);
		switchStatusTimer = undefined;
	}
	/** Show a temporary footer status that clears itself after SWITCH_STATUS_MS. */
	function flashSwitchStatus(ctx: { ui: { setStatus(key: string, text: string | undefined): void } }, text: string): void {
		ctx.ui.setStatus(SWITCH_STATUS_KEY, text);
		clearSwitchStatusTimer();
		switchStatusTimer = setTimeout(() => {
			switchStatusTimer = undefined;
			try {
				ctx.ui.setStatus(SWITCH_STATUS_KEY, undefined);
			} catch {
				/* session may have ended */
			}
		}, SWITCH_STATUS_MS);
	}

	pi.on("input", async (_event, ctx) => {
		try {
			if (!state.config.autoSwitchModel || !ctx.isIdle()) return;
			const current = ctx.model;
			if (!current) return;
			// The registered map forgets unreachable providers; config still tells
			// us which server owns the selected model after a restart.
			const configured = state.config.servers.find((s) => normalizeServer(s).providerName === current.provider);
			if (!configured) return;
			const stillSelected = () => {
				const selected = ctx.model;
				return ctx.isIdle() && selected?.provider === current.provider && selected.id === current.id;
			};
			const keys = (await readCredentials()) ?? {};
			const server = normalizeServer(withCredential(configured, keys));
			const hasModels = (r: DiscoverResult) => r.ok && modelIdList(r.models).length > 0;

			// Check the server live rather than trusting the last poll, which may
			// be up to pollIntervalMs old.
			const live = await discoverModels(server, state.config.timeoutMs);
			if (!stillSelected()) return;
			if (live.ok) {
				const liveIds = modelIdList(live.models);
				if (!liveIds.length) return;
				if (liveIds.includes(current.id) && registered.has(current.provider)) return;
			} else {
				// Only act once another configured server is confirmed live with
				// models; otherwise leave the selection and registry untouched.
				const others = state.config.servers
					.map((s) => normalizeServer(withCredential(s, keys)))
					.filter((s) => s.providerName !== current.provider);
				const probes = await Promise.all(others.map((s) => discoverModels(s, state.config.timeoutMs)));
				if (!stillSelected() || !probes.some(hasModels)) return;
			}

			// Restore registration/auth even if the server came back with the same
			// model. Reconciliation may discover a newer list than the live probe,
			// and registers any other server that came up since the last poll. When
			// falling back, keep the selected provider registered in case the
			// fallback vanishes before the switch; the next poll removes it.
			await runRefresh(state.config, ctx, live.ok ? undefined : current.provider);
			if (!stillSelected()) return; // preserve a manual selection made during I/O
			const target = pickSwitchTarget(current, live.ok, state.config.servers, registered);
			if (!target) return;
			const next = ctx.modelRegistry.find(target.entry.server.providerName, target.id);
			if (!next || !(await pi.setModel(next))) return;

			// Persist the switch for future pi processes (e.g. sub-agents): pi's
			// setModel saved the global default; fix scoped models and project default.
			const sameServer = next.provider === current.provider;
			const paths = [settingsFilePath("global", ctx.cwd)];
			if (ctx.isProjectTrusted()) paths.push(settingsFilePath("project", ctx.cwd));
			for (const path of paths) {
				try {
					await syncSwitchedModel(path, current, next, sameServer, (m) => !!ctx.modelRegistry.find(m.provider, m.id));
				} catch (e) {
					console.error(`[pi-local-llama] failed to update ${path}: ${errMsg(e)}`);
				}
			}

			if (!ctx.hasUI || !state.config.status) return;
			const via = target.entry.server.displayName;
			flashSwitchStatus(ctx, sameServer
				? `🦙 switched model: ${current.id} → ${next.id} (${via})`
				: `🦙 switched model: ${current.provider}/${current.id} → ${next.provider}/${next.id} (${via})`);
		} catch (e) {
			console.error(`[pi-local-llama] model auto-switch failed: ${errMsg(e)}`);
		}
	});

	pi.on("session_shutdown", () => {
		if (timer) {
			clearInterval(timer);
			timer = undefined;
		}
		clearSwitchStatusTimer();
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
			const summary = await runRefresh(config, ctx);
			if (!ctx.hasUI) return;
			if (!config.servers.length) {
				ctx.ui.notify("local-llama: no servers configured", "warning");
				return;
			}
			const missing = summary.auth.filter((a) => a.kind === "missing").length;
			ctx.ui.notify(
				`local-llama: scan complete — ${summary.up.length} up, ${summary.down.length} down, ${missing} needs API key, ${summary.auth.length - missing} key rejected`,
				summary.down.length || summary.auth.length ? "warning" : "info",
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
			"Manage local servers and keys. Usage: /local-llama-servers add|remove|list [url] [--project], key|key-remove <url>",
		handler: async (args, ctx) => {
			if (!ctx.hasUI) return;

			const tokens = (args ?? "").trim().split(/\s+/).filter(Boolean);
			const projectIdx = tokens.findIndex((t) => t === "--project");
			const scope: "global" | "project" = projectIdx !== -1 ? "project" : "global";
			const rest = tokens.filter((_, i) => i !== projectIdx);
			const sub = (rest[0] ?? "list").toLowerCase();
			const urlInput = rest.slice(1).join(" ").trim();

			if ((sub === "key" || sub === "key-remove") && projectIdx !== -1) {
				ctx.ui.notify("local-llama: API keys are saved globally; omit --project.", "warning");
				return;
			}
			if (scope === "project" && !ctx.isProjectTrusted()) {
				ctx.ui.notify("local-llama: project not trusted; cannot write project config.", "warning");
				return;
			}
			const path = configFilePath(scope, ctx.cwd, SERVERS_FILE);

			try {
				if (sub === "key" || sub === "key-remove") {
					const baseUrl = normalizeBaseUrl(urlInput);
					if (!baseUrl || !state.config.servers.some((s) => s.baseUrl === baseUrl)) {
						ctx.ui.notify("local-llama: specify a configured server URL (add it first if needed).", "warning");
						return;
					}
					if (sub === "key") {
						// Pi's built-in input dialog is not guaranteed to mask the key.
						ctx.ui.notify("local-llama: key entry may be visible while typing; avoid screen sharing.", "warning");
						const input = await ctx.ui.input(`API key for ${baseUrl}`, "Paste API key (not saved on cancel)");
						if (input === undefined) return;
						const key = input.trim();
						if (!key || /[\r\n]/.test(input)) {
							ctx.ui.notify("local-llama: API key must be non-empty and single-line.", "warning");
							return;
						}
						await updateCredential(baseUrl, key);
						ctx.ui.notify(`local-llama: saved API key for ${baseUrl} in ${credentialsPath()}.`, "info");
					} else {
						await updateCredential(baseUrl, undefined);
						ctx.ui.notify(`local-llama: removed saved API key for ${baseUrl}.`, "info");
					}
					if (state.config.servers.find((s) => s.baseUrl === baseUrl)?.apiKey) {
						ctx.ui.notify(`local-llama: ${baseUrl} has an explicit apiKey in server config, which takes precedence.`, "warning");
					}
					await runRefresh(state.config, ctx);
				} else if (sub === "add") {
					if (!urlInput) {
						ctx.ui.notify("local-llama: usage: /local-llama-servers add <url> [--project]", "warning");
						return;
					}
					const res = await addServer(path, urlInput);
					if (res.changed) {
						ctx.ui.notify(`local-llama: added ${res.baseUrl} to ${path}`, "info");
						state.config = await computeConfig(ctx);
						await runRefresh(state.config, ctx); // notifies up/down
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
						await runRefresh(state.config, ctx); // unregisters if needed
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
						const provider = normalizeServer(coerceServer(e.raw) ?? { baseUrl: e.url }).providerName;
						const auth = authStates.get(provider);
						const label = registered.has(provider) ? "up" : auth ? AUTH_LABEL[auth] : "down";
						return `${label}  ${e.url}`;
					});
					ctx.ui.notify(`local-llama (${path}):\n${lines.join("\n")}`, "info");
				} else {
					ctx.ui.notify(
						`local-llama: unknown subcommand "${sub}". Use add, remove, list, key, or key-remove.`,
						"warning",
					);
				}
			} catch (e) {
				ctx.ui.notify(`local-llama: failed to update ${path}: ${errMsg(e)}`, "error");
			}
		},
	});
}
