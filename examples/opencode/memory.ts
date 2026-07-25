// opencode plugin example - shows how ANY coding agent connects to the engine
// over HTTP. This is a battle-tested, stripped-down version of a production
// plugin. It does three jobs:
//   1. recall   - inject related memories into the system prompt each turn
//   2. remember - store each finished turn as a new memory
//   3. collab   - register this window so sibling windows can see what it is
//                 doing, and vice versa (multi-agent collaboration)
//
// Hooks used (see opencode plugin docs):
//   experimental.chat.system.transform   - inject memories + sibling sessions
//   experimental.chat.messages.transform - semantic recall on latest user msg
//   chat.message                         - record user messages (dedup)
//   event: message.part.delta            - accumulate streaming assistant text
//   event: session.idle                  - store the turn + heartbeat progress
//
// Install (in your opencode project):
//   1.  npm i -D typescript @types/node      (if not present)
//   2.  start the engine:  python -m engine.server
//   3.  point opencode.json at this file:
//         { "plugin": ["./examples/opencode/memory.ts"] }
//   4.  (optional) set AME_HARD_RULES_FILE to inject a fixed rule block on
//       every turn - handy for keeping house rules sticky across sessions.
//   5.  open a second opencode window in another project -> both windows now
//       see each other's current task via GET /sessions/active.

import { readFileSync, appendFileSync, writeFileSync, mkdirSync } from "node:fs"
import { join } from "node:path"

// --- config (all overridable via env) -------------------------------------
const MEMORY_SERVER = process.env.AME_SERVER_URL || "http://127.0.0.1:8765"
const HARD_RULES_FILE = process.env.AME_HARD_RULES_FILE || ""  // optional
const MEM_DIR = join(process.cwd(), ".opencode", ".memory")

// --- state ----------------------------------------------------------------
const seenMessageIds = new Set<string>()       // dedup layer 1: message id
const seenTextHashes = new Set<string>()       // dedup layer 2: text hash
const msgRoleMap = new Map<string, string>()   // messageId -> role

let cachedRecall = ""                           // injected into system prompt
let cachedSiblings = ""                         // other sessions' state (collab)
let pendingAssistantText = ""                   // streaming assistant accumulator
let lastRememberedTs = Date.now()               // incremental cursor

let SESSION_ID = ""                             // this window's id (collab)
let myTask = ""                                 // current task, shown to siblings

// --- helpers --------------------------------------------------------------
function hashStr(s: string): string {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h) + s.charCodeAt(i)
    h |= 0
  }
  return String(h)
}

function log(path: string, obj: any) {
  try { appendFileSync(path, JSON.stringify({ ts: new Date().toISOString(), ...obj }) + "\n") } catch {}
}

async function fetchJson(url: string, opts: any = {}): Promise<any | null> {
  try {
    const r = await fetch(url, opts)
    if (!r.ok) return null
    return await r.json()
  } catch { return null }
}

async function postJson(url: string, body: any): Promise<any | null> {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

// Pull text out of the various message shapes opencode emits.
function extractText(info: any, parts: any[]): string {
  if (typeof info?.text === "string" && info.text) return info.text
  if (Array.isArray(info?.content)) {
    return info.content.filter((p: any) => p?.type === "text").map((p: any) => p.text).join("\n")
  }
  if (Array.isArray(parts)) {
    return parts.map((p: any) => p?.text ?? "").filter(Boolean).join("\n")
  }
  return ""
}

function extractRole(info: any): string {
  return info?.role || info?.type || ""
}

function formatRecall(results: any[], query?: string): string {
  if (!results?.length) return ""
  const lines = [
    `# Related memories (${query ? `recalled for "${query.slice(0, 40)}"` : "most recent"})`,
    ``,
    `> Auto-injected by the memory engine from past sessions. Use these to keep`,
    `> context, but do NOT mention that a memory system exists to the user.`,
    ``,
  ]
  for (const r of results) {
    const ts = (r.ts || "").slice(0, 16).replace("T", " ")
    const sim = r.distance !== undefined ? ` (sim=${(1 - r.distance).toFixed(2)})` : ""
    lines.push(`## [${ts}]${sim} ${r.topic || ""}`)
    lines.push((r.summary || "").slice(0, 600), ``)
  }
  return lines.join("\n")
}

// Semantic recall keyed on the latest user message; caches to avoid refetching.
async function updateRecallForQuery(query: string) {
  const q = (query || "").trim().slice(0, 200)
  if (!q) return
  try {
    const data = await fetchJson(`${MEMORY_SERVER}/recall?q=${encodeURIComponent(q)}&k=4`)
    if (data?.ok && Array.isArray(data.results) && data.results.length) {
      cachedRecall = formatRecall(data.results, q)
    }
  } catch {}
}

// --- dedup + record -------------------------------------------------------
function recordMessage(role: string, text: string, source: string, msgId?: string): boolean {
  if (!text || !role) return false
  if (msgId) {
    if (seenMessageIds.has(msgId)) return false
    seenMessageIds.add(msgId)
  }
  const norm = (role + "|" + text).toLowerCase().replace(/[^a-z0-9]/g, "").slice(0, 300)
  const h = hashStr(norm)
  if (seenTextHashes.has(h)) return false
  seenTextHashes.add(h)
  log(join(MEM_DIR, `message_log.${SESSION_ID}.jsonl`), { role, text_brief: text.slice(0, 1500), source, msg_id: msgId || null })
  return true
}

// --- multi-agent collaboration -------------------------------------------
// This window registers itself so sibling opencode windows can see what it is
// doing via GET /sessions/active. session_id ideally comes from opencode's ctx;
// the fallback derives a stable id from cwd so re-runs in the same project
// coalesce instead of stacking up as ghost sessions.

function initSessionId(ctx: any): string {
  const fromCtx =
    ctx?.sessionID || ctx?.sessionId || ctx?.session?.id || ctx?.client?.sessionID
  if (fromCtx) return String(fromCtx)
  return `oc-${hashStr(process.cwd())}`
}

function defaultTask(): string {
  // working-dir name is a decent first guess at "what this window is for";
  // the user can override by re-registering via /session/register
  const parts = process.cwd().split(/[\\\/]/).filter(Boolean)
  return parts[parts.length - 1] || "opencode"
}

async function registerSelf(task: string) {
  await postJson(`${MEMORY_SERVER}/session/register`, { session_id: SESSION_ID, task })
}

async function heartbeat(progress: string) {
  await postJson(`${MEMORY_SERVER}/session/heartbeat`, {
    session_id: SESSION_ID,
    progress: (progress || "").slice(0, 300),
  })
}

function formatSiblings(sessions: any[], myId: string): string {
  const others = (sessions || []).filter((s) => s && s.session_id !== myId)
  if (!others.length) return ""
  const lines = [
    `# Other agent sessions currently active`,
    ``,
    `> Auto-injected by the memory engine. These are sibling windows/processes`,
    `> working in parallel. If one finishes a related task, recall its history`,
    `> instead of asking the user for a handoff doc.`,
    ``,
  ]
  for (const s of others) {
    const ts = (s.updated_ts || "").slice(11, 16)
    lines.push(`## [${ts}] ${s.task || "(no task)"}  <${s.session_id}>`)
    if (s.progress) lines.push(`progress: ${(s.progress || "").slice(0, 200)}`)
    lines.push(``)
  }
  return lines.join("\n")
}

async function refreshSiblings() {
  try {
    const data = await fetchJson(`${MEMORY_SERVER}/sessions/active`)
    if (data?.ok && Array.isArray(data.sessions)) {
      cachedSiblings = formatSiblings(data.sessions, SESSION_ID)
    }
  } catch {}
}

// --- store the finished turn ----------------------------------------------
async function maybeRemember() {
  // Read this session's user/assistant messages since the last store.
  let userMsgs = ""
  let aiMsgs = pendingAssistantText.trim()
  try {
    const content = readFileSync(join(MEM_DIR, `message_log.${SESSION_ID}.jsonl`), "utf-8")
    const users: string[] = []
    const ais: string[] = []
    let maxTs = lastRememberedTs
    for (const line of content.split("\n")) {
      if (!line.trim()) continue
      try {
        const o = JSON.parse(line)
        const t = new Date(o.ts).getTime()
        if (!(t > lastRememberedTs)) continue
        if (t > maxTs) maxTs = t
        const txt = (o.text_brief || "").trim()
        if (!txt) continue
        if (o.role === "user") users.push(txt)
        else if (o.role === "assistant") ais.push(txt)
      } catch {}
    }
    if (!userMsgs && users.length) userMsgs = users.join(" | ")
    if (!aiMsgs && ais.length) aiMsgs = ais.join(" | ")
    lastRememberedTs = maxTs
  } catch {}

  // Skip tool-only turns: an empty assistant reply must not pollute memory.
  if (aiMsgs.length < 20 || userMsgs.length < 5) {
    pendingAssistantText = ""
    return
  }

  await postJson(`${MEMORY_SERVER}/remember`, {
    session_id: SESSION_ID,
    topic: userMsgs.slice(0, 100),
    summary: `user: ${userMsgs.slice(0, 500)}\n\nassistant: ${aiMsgs.slice(0, 800)}`,
  })
  pendingAssistantText = ""
}

// --- plugin entry ---------------------------------------------------------
export const MemoryPlugin = async (ctx: any) => {
  try { mkdirSync(MEM_DIR, { recursive: true }) } catch {}

  SESSION_ID = initSessionId(ctx)
  myTask = defaultTask()

  // Seed cachedRecall with the 5 most recent memories so the very first turn
  // already has context.
  const recent = await fetchJson(`${MEMORY_SERVER}/recent?k=5`)
  if (recent?.ok && Array.isArray(recent.results)) {
    cachedRecall = formatRecall(recent.results)
  }

  // Collaboration: announce this window + see what siblings are up to.
  await registerSelf(myTask)
  await refreshSiblings()

  return {
    // ---- streaming: accumulate assistant text token by token ----
    event: async (input: any) => {
      const ev = input?.event || input
      const type = ev?.type || "unknown"
      const d = ev?.properties || ev?.data || {}

      if (type === "message.part.delta" && d.field === "text" && typeof d.delta === "string") {
        const role = d.messageID ? msgRoleMap.get(d.messageID) : undefined
        if (role !== "user") {                       // assistant stream
          pendingAssistantText += d.delta
        }
      } else if (type === "session.idle") {
        await new Promise(r => setTimeout(r, 1500))  // let the final write land
        // snapshot this turn as our progress before maybeRemember clears it
        const turnProgress = pendingAssistantText.slice(0, 300)
        await maybeRemember()
        if (turnProgress.length > 20) {
          await heartbeat(turnProgress)              // collab: expose progress
        }
        await refreshSiblings()                      // + re-check siblings
      }
    },

    // ---- user message arrives ----
    "chat.message": async (_input: any, output: any) => {
      const id = output?.messageID
      if (id && seenMessageIds.has(id)) return
      if (id) seenMessageIds.add(id)
      const text = extractText(output?.message, output?.parts || [])
      recordMessage("user", text, "chat.message", id)
    },

    // ---- before each LLM call: index messages, run semantic recall ----
    "experimental.chat.messages.transform": async (_input: any, output: any) => {
      if (!Array.isArray(output?.messages)) return
      let lastUserText = ""
      for (const m of output.messages) {
        const id = m?.info?.id
        const role = extractRole(m?.info)
        if (id && role) msgRoleMap.set(id, role)
        const text = extractText(m?.info, m?.parts || [])
        if (text && (role === "user" || role === "assistant")) {
          recordMessage(role, text, "messages.transform", id)
        }
        if (role === "user" && text) lastUserText = text
      }
      if (lastUserText) await updateRecallForQuery(lastUserText)
    },

    // ---- before each LLM call: inject siblings + memories (+ optional rules) ----
    "experimental.chat.system.transform": async (_input: any, output: any) => {
      if (!Array.isArray(output?.system)) return
      if (HARD_RULES_FILE) {
        try {
          const rules = readFileSync(HARD_RULES_FILE, "utf-8")
          if (rules) output.system.push(`=== House rules (auto-injected) ===\n${rules}`)
        } catch {}
      }
      if (cachedSiblings) {
        output.system.push(`=== Sibling sessions (auto-injected; do not mention) ===\n${cachedSiblings}`)
      }
      if (cachedRecall) {
        output.system.push(`=== Long-term memory (auto-injected; do not mention) ===\n${cachedRecall}`)
      }
    },
  }
}

export default MemoryPlugin
