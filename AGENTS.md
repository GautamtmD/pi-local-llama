# AGENTS.md — pi-local-llama

A guide for coding agents (and humans) working on this extension. Read this
first. The README is user-facing; this file is for **debugging and improving**
the code.

## TL;DR

`pi-local-llama` is a [Pi](https://pi.dev) extension with **one source file**
(`src/index.ts`, ~900 lines, zero runtime dependencies) that:

1. Reads a list of OpenAI-compatible local LLM server base URLs from config.
2. Polls each server's `<baseUrl>/models` endpoint.
3. Calls `pi.registerProvider(...)` for reachable servers and
   `pi.unregisterProvider(...)` when they go down, change models, **or change
   live context size**.
4. Optionally manages a `local-*/*` glob in `settings.json → enabledModels`
   so discovered models join the **Ctrl+P** cycling list.
5. Detects each server's **real running context** (not a fixed 128k) by
   probing llama-server's `/props` and `/slots` — see [Context window
   detection](#context-window-detection) below.

It targets **llama.cpp's `llama-server`** but works with any OpenAI-compatible
server (Ollama, vLLM, LM Studio, SGLang).

## Repository layout

```
src/index.ts                 # THE extension. Everything lives here.
test/
  mock-server.mjs            # standalone background mock /v1/models server
  sanity.mts                 # end-to-end: startup discovery, kill server, rescan
  scope.mts                  # provider-name prefixing + settings.json scope mgmt
  servers.mts                # /local-llama-servers add|list|remove against a mock
local-servers.example.json   # documented example config
package.json                 # declares the extension via "pi.extensions": ["./src/index.ts"]
tsconfig.json                # noEmit; used only for editor/type checking
```

There is **no build step** and **no `dist/`**. Pi loads `src/index.ts`
directly via jiti. `npm install` is only needed for `tsc --noEmit` type-checking.

## Running / testing / type-checking

```bash
# Quick test in a real pi process (loads src/index.ts directly)
pi -e ./src/index.ts

# Install as a package (recommended for normal use)
pi install .

# Type-check only (requires npm install)
npx tsc --noEmit

# Run the integration tests (plain node — they fake the pi API)
node test/sanity.mts     # also proves a live poll cycle + provider unregister
node test/scope.mts      # uses an isolated temp PI_CODING_AGENT_DIR
node test/servers.mts    # uses an isolated temp PI_CODING_AGENT_DIR + mock server
```

**Note:** `test/sanity.mts` does NOT isolate config, so it will also pick up
your real global `~/.pi/agent/local-llama-servers.json` (you may see extra
providers like `local-localhost-8000` in its output — that's expected, it
proves the global-merge path works). `scope.mts` and `servers.mts` DO isolate
by setting `PI_CODING_AGENT_DIR` to a temp dir.

## The pi extension API surface actually used

The extension imports `{ CONFIG_DIR_NAME, getAgentDir, type ExtensionAPI }`
from `@earendil-works/pi-coding-agent`. It uses a deliberately small slice of
the API. The canonical type definitions live in
`node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts`
— **read that file** before changing how the extension talks to pi.

Calls used:
- `pi.registerProvider(name, config)` — registers/refreshes a provider. Queued
  during initial load, applied immediately thereafter (safe from callbacks).
  `config` shape: `{ name, baseUrl, apiKey, api: "openai-completions", models[] }`.
- `pi.unregisterProvider(name)` — removes the provider and all its models.
- `pi.registerCommand(name, { description, handler })` — `handler(args: string, ctx)`.
- `pi.on("session_start", (event, ctx) => ...)` — `event.reason` is
  `"startup" | "reload" | "new" | "resume" | "fork"`. Used to gate the
  first-startup scope-pattern add and to (re)start polling.
- `pi.on("session_shutdown", ...)` — clears the poll interval. **Does NOT
  unregister providers** (intentional — see invariants below).

`ctx` (the `ExtensionContext` / `ExtensionCommandContext`) fields used:
- `ctx.cwd`, `ctx.hasUI`, `ctx.isProjectTrusted()`, `ctx.mode`.
- `ctx.ui.notify(message, "info" | "warning" | "error")`.
- `ctx.ui.setStatus(key, text | undefined)` — footer status line.

**Always guard UI calls with `if (!ctx.hasUI) return;` / `ctx?.hasUI`** — the
extension runs in `print`/`json`/`rpc` modes too (e.g. `pi --list-models`),
where there is no UI. The code already does this; keep doing it.

## Mental model: the data flow

```
Config sources (merged)            Discovery (per server, in parallel)     Reconcile (refresh)
─────────────────────────          ─────────────────────────────────────  ────────────────────
1. global: <agentDir>/             • GET <baseUrl>/models  → RawModel[]      compare each server vs the
   local-llama-servers.json        • GET <baseUrl>/props   → n_ctx?            `registered` map, keyed by
2. PI_LOCAL_SERVERS env            • GET <baseUrl>/slots   → n_ctx?            `providerSignature` (ids +
3. project: <cwd>/.pi/               (props & slots are best-effort;          contextWindow + maxTokens):
   local-llama-servers.json           undefined when absent)                • not seen & ok       → registerProvider (added)
(trusted projects only)            (timeout = timeoutMs each)               • signature changed   → registerProvider (changed)
                                                                            • same & ok           → no-op
                                                                            • !ok (was seen)      → unregisterProvider
                                                                            • in map, not in cfg  → unregisterProvider
```

Three things to internalize:

1. **`computeConfig()` is re-run inside `/local-llama-servers` handlers and
   on each poll** (the poll reads `state.config`, not a snapshot), so servers
   added/removed via commands are picked up without a restart.
2. **Change detection is by `providerSignature`** — sorted
   `id:contextWindow:maxTokens` per model. `refresh()` re-registers whenever
   this signature changes, so relaunching llama-server with a different `-c`
   (which changes contextWindow) is picked up on the next poll without a
   `/reload`. `modelIdsKey` is still kept on `RegisteredEntry`, but only for
   counting models in the status footer. Note: changing ONLY `name`/`compat`
   mid-session will NOT re-register — by design.
3. **`refresh()` is the single source of truth** for register/unregister and
   for emitting notifications + footer status. Startup discovery, polling,
   `/local-llama-rescan`, and post-add/remove all funnel through it.

## Key invariants (do not break these)

- **Every provider name starts with `local-`.** `toProviderName()` enforces
  this (see `PROVIDER_PREFIX`). This is what makes the single glob
  `local-*/*` match all of them. The glob MUST contain the `/` because
  minimatch `*` does not cross `/`.
- **Pi has NO install/uninstall hooks for extensions.** "Add on install" is
  approximated by a one-time, idempotent write on the first `session_start`
  with `reason === "startup"` (the `autoScopePattern` / `if-empty` path).
  Uninstall cannot self-clean — that's why `/local-llama-scope disable`
  exists and is documented as the removal path.
- **`session_shutdown` does NOT unregister providers.** Provider removal is
  driven purely by reachability polling, so switching sessions doesn't yank a
  model mid-use. If you add cleanup on shutdown, you will break this.
- **`session_shutdown` does NOT remove the scope pattern either.** It must
  persist between sessions so it's present at the next startup's scope
  resolution.
- **settings.json writes are atomic** (tmp + rename) in `writeSettingsObject()`
  to avoid corrupting pi's own settings file on concurrent writes. Keep this.
- **The extension never throws out of an event/command handler** in a way that
  would crash pi — discovery errors become `{ ok: false, error }` and are
  reported via `notify`. Preserve this resilience.
- **`apiKey` defaults to `"local"`** in `buildProviderConfig()`. Local servers
  ignore it, but pi requires *some* auth before a model is selectable.

## Two distinct config files — don't confuse them

| File | Edited by | Purpose |
|------|-----------|---------|
| `local-llama-servers.json` | `/local-llama-servers add\|remove\|list` | The **polled server list** (+ poll/timeout flags) |
| `settings.json` (`enabledModels`) | `/local-llama-scope enable\|disable\|status` | The **Ctrl+P cycling glob** (`local-*/*`) |

They configure different things and are intentionally separate commands.

## Config merge semantics (`mergeConfigs`)

- Servers **accumulate** across sources, keyed by normalized `baseUrl`;
  later sources override earlier ones for the same URL (deep object merge per
  server is NOT done — later entry replaces entirely, so include all fields).
- **Scalars** (pollIntervalMs, timeoutMs, notify, status, scopeSettings,
  scopePattern, autoScopePattern) are taken from the **override** ONLY if the
  override contributed ≥1 server; otherwise the base's scalars win.
  (`useOverrideScalars = override.servers.length > 0`.)
- `discoverAtStartup` is the boolean AND of both (either side can disable it).
- Minimums are clamped in `coerceConfig`: `pollIntervalMs >= 1000`,
  `timeoutMs >= 250`.

If you see "the env var ignored my `timeoutMs`", it's because the env-only
config is built from `DEFAULTS` and merged — check the merge rule above.

## URL normalization (`normalizeBaseUrl`)

- Adds `http://` if no scheme.
- **A bare `host:port` (no path) is normalized to `http://host:port/v1`** —
  the conventional OpenAI-compatible path. A URL that already has a path is
  kept verbatim (trailing slashes stripped). This is why
  `/local-llama-servers add localhost:1234` works.
- Invalid URLs return `undefined` and are silently dropped during coercion.

## Provider/model derivation

- Provider name: `local-<sanitized host:port>`, OR `local-<sanitized explicit
  provider>` if `provider` is set. `sanitize()` lowercases and collapses
  non-`[a-z0-9]` runs into `-`. Examples: `local-localhost-1234`,
  `local-ollama`, `local-my-gpu`.
- Models come from the `/models` payload. Accepts `{ data: [...] }` (OpenAI),
  a bare `[...]` array, or `{ models: [...] }`. Each entry needs an `id` or
  `name` string. `max_tokens` → maxTokens (default 8192). contextWindow is
  resolved live per server — see [Context window detection](#context-window-detection).
- `compat` always starts from safe local defaults
  (`supportsDeveloperRole: false`, `supportsReasoningEffort: false`) and is
  overridden by the server-level `compat` from config. For servers that reject
  `max_completion_tokens`, the user sets `maxTokensField: "max_tokens"`.
  See `docs/models.md` in the pi package for the full compat table.

## Context window detection

`resolveContextWindow(raw, serverCtx)` picks a model's contextWindow,
preferring the most authoritative **live** source — the value the server was
actually launched with, not the model's architectural max:

1. **`serverCtx`** — discovered once per server by `discoverServerContext()`,
   which probes llama-server in order (`getJson` returns null on any failure, so
   a missing endpoint is normal and silent):
   - `GET <baseUrl>/props` → `ctxFromProps`:
     `default_generation_settings.n_ctx`, else top-level `n_ctx` / `n_ctx_per_slot`.
   - else `GET <baseUrl>/slots` → `ctxFromSlots`: max per-slot `n_ctx`.
2. The model's own fields from `/v1/models`: `meta.n_ctx` (via `metaNctx`),
   then `context_window`, then `context_length`.
3. `DEFAULT_CONTEXT_WINDOW` (128000) — last resort.

Ground truth from a real llama-server launched `-c 100096`: `/props`, `/slots`,
and `meta.n_ctx` **all** report `100096`; `meta.n_ctx_train` is `262144`
(architectural training max — deliberately **not** used). For plain Ollama (no
`/props`/`/slots`, and `/v1/models` carries no context) the chain falls through
to the default; that's acceptable because Ollama sets context per-request via
`num_ctx`, so a single server-wide number isn't meaningful there.

Because contextWindow is part of `providerSignature`, **relaunching
llama-server with a different `-c` re-registers within one poll cycle**.

## The Ctrl+P snapshot-vs-live gotcha (important for bug reports)

Pi's Ctrl+P handler resolves the `models` glob **once at startup**:
- **Server up at startup** → non-empty "session scope" **snapshot**. Model is
  in Ctrl+P immediately, but later changes need `/scoped-models` re-open or a
  restart. With a non-empty scope, Ctrl+P cycles **only** scoped models, so
  users should include their other providers in the pattern too.
- **Server down at startup, comes up later** → empty scope → Ctrl+P falls
  back to **live** `getAvailable()` on every keypress → model appears the
  moment it's registered, but cycling rotates through ALL available models.

Either way the model is ALWAYS immediately selectable via `/model` (Ctrl+L).
If a user reports "model not in Ctrl+P but is in /model", this is almost
certainly the cause, not a bug in this extension.

## Where to look when debugging

| Symptom | First place to look |
|---------|---------------------|
| Models don't appear in `/model` | `apiKey` missing? (defaults to `"local"`). Hit `<baseUrl>/models` in a browser — must return `{ data: [{ id }] }`. `discoverAtStartup` timing. |
| Provider registered but wrong name | `toProviderName()` / `sanitize()` / `PROVIDER_PREFIX` |
| Provider not re-registered after a change | `providerSignature()` (ids + contextWindow + maxTokens). Changing only `name`/`compat` will NOT re-register — by design. |
| Wrong / 128k context window shown | `discoverServerContext()` / `resolveContextWindow()`. Hit `<baseUrl>/props` and `<baseUrl>/slots` directly; confirm `meta.n_ctx`. `meta.n_ctx_train` is intentionally ignored. |
| Server up but treated as down | `discoverModels()` error path; `timeoutMs`; `fetchWithTimeout`/`AbortSignal.timeout` |
| Polling stopped / not running | `session_start` handler; `timer`/`clearInterval`; `config.servers.length` and `pollIntervalMs > 0` gate |
| Scope pattern not added | `autoScopePattern`, `scopeSettings`, `isProjectTrusted()`, the `if-empty` mode in `ensureScopePattern()` |
| settings.json corrupted / race | `writeSettingsObject()` (tmp+rename); pi writing concurrently |
| Wrong servers loaded | config merge order (global → env → project); a stale `local-llama-servers.json` |
| Status footer wrong | `emitStatus()` / `emitNotifications()` / `emitStartupSummary()` |
| Command not found / wrong parsing | the `local-llama-servers` arg tokenizing (`--project` strip) in the handler |

## Conventions for changes

- **Keep everything in `src/index.ts`.** It's intentionally one file. Add a
  new section header comment in the existing style rather than splitting files.
- **Zero runtime deps.** Use only `node:*` builtins + the pi API. New
  dependencies should be rejected unless absolutely necessary.
- **Mirror the resilience pattern:** every external call (fs, fetch) is
  wrapped so it returns a structured result or logs via `console.error` with
  a `[pi-local-llama]` prefix, never throws uncaught.
- **When you change behavior, update both the README and the JSDoc header at
  the top of `src/index.ts`** — they intentionally describe the same thing.
- **Add/adjust a test under `test/`** for new logic. The existing tests fake
  the pi API with a tiny object (`on`, `registerProvider`,
  `unregisterProvider`, `registerCommand`) — copy that pattern.
- **No transpilation artifacts** — `.gitignore` excludes `*.js`/`dist/`. Don't
  commit built files.

## Environment variables

- `PI_LOCAL_SERVERS` — comma-separated base URLs (config source #2).
- `PI_CODING_AGENT_DIR` — overrides the global config dir (used by the tests
  to isolate state). `getAgentDir()` reads this.

## Quick reference: extension entry point shape

```ts
export default async function localLlamaExtension(pi: ExtensionAPI): Promise<void> {
  // 1. load global + env config (process-wide)
  // 2. optional blocking discoverAtStartup → refresh()
  // 3. pi.on("session_start") → merge project config, maybe write scope
  //    pattern, refresh(), start setInterval(poll → refresh(state.config))
  // 4. pi.on("session_shutdown") → clearInterval (providers left registered)
  // 5. registerCommand("local-llama-rescan" | "local-llama-scope" |
  //                    "local-llama-servers")
}
```

When in doubt, read `refresh()` — it's the heart of the extension.
