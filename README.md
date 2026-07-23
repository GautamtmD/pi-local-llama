# pi-local-llama

A [Pi](https://pi.dev) extension that **auto-detects OpenAI-compatible local LLM servers** (Ollama, LM Studio, vLLM, llama.cpp server, SGLang, etc.) and **registers their models with Pi** — adding them when a server comes up, and removing them when it's killed or unreachable.

## What it does

- Reads a list of server base URLs from config.
- Polls each server's `<baseUrl>/models` endpoint (OpenAI-compatible).
- When a server is reachable, registers a provider so its models appear in
  `/model` and `/scoped-models`.
- When the available model list **or a model's live context window** changes,
  re-registers to keep it in sync.
- When a server is killed / unreachable, unregisters its provider so the
  models disappear.
- Detects each server's **real running context size** (not a fixed 128k) by
  probing llama-server's `/props` and `/slots` endpoints — see
  [Context window detection](#context-window-detection).
- Shows a footer status line and notifies on state transitions.

## About "scoped models" and the Ctrl+P pattern

Pi's **Ctrl+P cycling list** (the one you edit with `/scoped-models`) is **not
mutable from an extension** — there is no public API for it, and pi has **no
install/uninstall lifecycle hooks** for extensions. The supported way to make
discovered models available is `pi.registerProvider()` / `pi.unregisterProvider()`.
Registered models then show up in:

- `/model` (Ctrl+L) — selectable as the active model.
- `/scoped-models` — toggle them **on** to add them to Ctrl+P cycling.

### Naming guarantee

Every provider this extension registers is named with the **`local-`** prefix,
e.g. `local-localhost-1234` (or `local-<your-explicit-name>` if you set
`"provider": "..."`). That means the single glob `local-*/*` reliably matches
all of them (e.g. `local-localhost-1234/llama3.1:8b`). The glob must include
the `/` because minimatch `*` does not cross `/`.

### Auto-managed pattern (best-effort "on install")

With `autoScopePattern: true` (the default), the extension adds `local-*/*` to
the `enabledModels` array in `settings.json` the **first time pi starts with the
extension present and `enabledModels` is empty/absent**. It only writes when the
pattern is missing (idempotent), and it leaves your `enabledModels` untouched if
you already have your own patterns. Which file it manages is set by
`scopeSettings` (`"global"` → `~/.pi/agent/settings.json`, or `"project"` →
`.pi/settings.json`).

This is the closest thing to "add on install" that pi allows: there's no install
hook, so it runs on first `session_start` instead. The change takes effect on
the **next** pi startup (scope patterns are resolved once at startup).

### Removal / uninstall (important)

Because there is **no uninstall hook**, `pi remove` does **not** clean up the
pattern from `settings.json`. Remove it yourself, either by editing the file or
via the command **before/after** uninstalling:

```
/local-llama-scope disable
```

The full command set:

- `/local-llama-scope enable` — force-add the pattern (even if you have other
  `enabledModels`).
- `/local-llama-scope disable` — remove the pattern.
- `/local-llama-scope status` — show whether the pattern is present and list
  current `enabledModels`.

If you'd rather not have the extension touch your settings at all, set
`"autoScopePattern": false` and manage the pattern manually (or rely on the
live fallback below).

## Ctrl+P cycling: startup vs. mid-session

Pi's Ctrl+P handler has two modes, chosen by whether the `models` pattern
resolved to anything at **startup**:

- **Server up at startup** → the pattern resolves to a non-empty "session scope",
  which is a **snapshot**. The model is in Ctrl+P immediately, but later changes
  (a new server appearing, or a server's model list changing) won't show up in
  Ctrl+P until you re-open `/scoped-models` (which re-resolves) or restart Pi.
  Also, with a non-empty scope Ctrl+P cycles **only** scoped models — include
  your other providers in the pattern too, e.g.
  `"models": ["local-*/*", "claude-*/*", "gpt-*/*"]`.
- **Server down at startup, comes up mid-session** → the pattern matches nothing
  at startup, the scope stays empty, and Ctrl+P falls back to **live**
  `getAvailable()` on every keypress. The model then appears in Ctrl+P the
  moment the extension registers it (no `/reload` needed) — but cycling will
  then rotate through *all* available models, not just local ones.

In both cases the model is always immediately selectable via `/model` (Ctrl+L).

## Install

### Option A — as a Pi package (recommended)

From this directory:

```bash
pi install .
```

Or once published to npm/git:

```bash
pi install npm:pi-local-llama
pi install git:github.com/<you>/pi-local-llama
```

### Option B — copy into your extensions dir

```bash
cp -r . ~/.pi/agent/extensions/pi-local-llama
```

### Option C — quick test

```bash
pi -e ./src/index.ts
```

> No build step or `npm install` is required to *run* the extension (Pi loads
> TypeScript via jiti, and this extension has no runtime dependencies). Run
> `npm install` only if you want editor type-checking / `npx tsc --noEmit`.

## Configuration

The config file is JSON, looked up in this order and **merged** (later sources
override earlier ones for duplicate base URLs):

1. `<agentDir>/local-llama-servers.json` — global, always read.
   (`<agentDir>` is `~/.pi/agent`, or `$PI_CODING_AGENT_DIR` if set.)
2. `PI_LOCAL_SERVERS` env var — comma-separated base URLs.
3. `<cwd>/.pi/local-llama-servers.json` — project-local, **trusted projects only**.

Copy [`local-servers.example.json`](./local-servers.example.json) to get started:

```bash
cp local-servers.example.json ~/.pi/agent/local-llama-servers.json
```

### Schema

```jsonc
{
  "pollIntervalMs": 5000,      // min 1000; how often to re-check servers
  "timeoutMs": 3000,           // min 250; per-request timeout (also caps startup wait)
  "notify": true,              // notify on up/down transitions
  "status": true,              // show a footer status line
  "discoverAtStartup": true,   // do one blocking discovery pass before startup
  "autoScopePattern": true,    // add `scopePattern` to settings.json on first startup (if enabledModels empty)
  "scopeSettings": "global",   // "global" (~/.pi/agent/settings.json) or "project" (.pi/settings.json)
  "scopePattern": "local-*/*", // glob added/removed from settings `enabledModels` for Ctrl+P
  "servers": [
    "http://localhost:1234/v1",            // shorthand: just a base URL
    {
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",                  // sent as Authorization: Bearer; local servers usually ignore it
      "name": "ollama",                    // display name
      "provider": "ollama",            // provider id; always forced to start with `local-` -> `local-ollama`
      "input": ["text", "image"],          // default input types per model
      "compat": {                          // OpenAI-compat flags (see docs/models.md)
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      }
    }
  ]
}
```

Each server becomes its own provider. Provider names are **always forced to
start with `local-`** (so `local-*/*` matches them): `local-<host>-<port>` by
default, or `local-<provider>` if you set `provider`. Models are addressable as
`<provider>/<modelId>`, e.g. `local-ollama/llama3.1:8b`.

### `compat` notes for local servers

Many OpenAI-compatible local servers don't fully implement the latest OpenAI
API. Safe defaults (`supportsDeveloperRole: false`, `supportsReasoningEffort:
false`) are applied automatically. For servers that reject
`max_completion_tokens`, add `"maxTokensField": "max_tokens"`.

## Context window detection

Pi needs a model's context window so it knows how much it can fit in one
request. A fixed default (128k) is wrong for local servers: llama-server's
actual context is whatever you launched it with (`-c`), and that varies with
VRAM, model size, and what else is using the GPU. So the extension detects the
**live** value per server, in this precedence:

1. **llama-server `/props`** → `default_generation_settings.n_ctx` (the
   per-slot running context — exactly what one request can use). Falls back to
   top-level `n_ctx` / `n_ctx_per_slot` on newer builds.
2. **llama-server `/slots`** → the max per-slot `n_ctx`.
3. **The model's own fields** from `/v1/models`: `meta.n_ctx`, then
   `context_window`, then `context_length`.
4. A 128k default — only if nothing above reported a value.

For plain llama-server, step 1 or 2 always wins, so the reported size matches
your `-c` flag (divided across `-np` parallel slots). The model's
`meta.n_ctx_train` (the architectural/training max) is deliberately **not**
used — that's the ceiling the model was trained on, not what this server can do
right now.

If you relaunch the server with a different `-c` while pi is running, the next
poll detects the change and re-registers with the new size (no `/reload`
needed).

Servers that don't expose `/props` or `/slots` (e.g. plain Ollama, some vLLM
builds) fall through to per-model fields or the default. (Ollama in particular
sets context per request via `num_ctx`, so a single server-wide number isn't
meaningful there.)

## How it works / lifecycle

- **Startup:** an `async` extension factory does a one-time, parallel discovery
  pass so models are available immediately — including to `pi --list-models`.
  (Set `"discoverAtStartup": false` to skip the startup wait and rely on
  background polling instead.)
- **Per session (`session_start`):** merges project-local config, reconciles
  state once, then starts a polling `setInterval`.
- **Polling:** on each tick it registers providers for newly-reachable servers
  (or when their model list **or live context window** changed) and unregisters
  providers for servers that became unreachable.
- **Shutdown (`session_shutdown`):** clears the interval. Providers are *left
  registered* — "model removed when server is killed" is handled by
  reachability polling, not by session end, so switching sessions won't yank
  models mid-use.

## Commands

- `/local-llama-servers add <url> [--project]` — add a server to poll and
  rescan immediately. A bare `host:port` is normalized to `http://host:port/v1`
  (the conventional OpenAI-compatible path); give a full path to override.
  Writes to the global config by default; `--project` writes to
  `.pi/local-llama-servers.json` (trusted projects only).
- `/local-llama-servers remove <url> [--project]` — remove a server (unregisters
  its provider on the next scan).
- `/local-llama-servers list [--project]` — list configured servers with up/down
  status.
- `/local-llama-rescan` — immediately re-scan all configured servers and sync.
- `/local-llama-scope enable|disable|status` — manage the `local-*/*` Ctrl+P
  pattern in `settings.json` (the reliable add/remove path; see "Removal /
  uninstall" above).

> `/local-llama-servers` edits `local-llama-servers.json` (the server list);
> `/local-llama-scope` edits `settings.json` (`enabledModels`). They're separate
> because they configure different things.

## Environment variables

- `PI_LOCAL_SERVERS` — comma-separated base URLs, e.g.
  `PI_LOCAL_SERVERS=http://localhost:1234/v1,http://localhost:11434/v1 pi`
- `PI_CODING_AGENT_DIR` — override the global config directory.

## Troubleshooting

- **Models don't appear in `/model`:** local servers ignore the API key, but Pi
  needs *some* auth before a model is selectable. The extension defaults
  `apiKey` to `"local"`; you generally don't need to set it.
- **Server is up but models are empty:** open `<baseUrl>/models` in a browser to
  confirm it returns `{ "data": [{ "id": "..." }, ...] }`.
- **Startup feels slow:** if all your servers are usually down, set
  `"discoverAtStartup": false` (or lower `"timeoutMs"`).
