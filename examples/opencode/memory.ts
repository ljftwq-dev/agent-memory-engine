// opencode plugin example - shows how ANY coding agent connects to the engine
// over HTTP. This is a battle-tested, stripped-down version of a production
// plugin: it injects recalled memories into the system prompt, runs a semantic
// recall on each new user turn, and stores the finished turn as a new memory.
//
// Hooks used (see opencode plugin docs):
//   experimental.chat.system.transform   - inject memories into system prompt
//   experimental.chat.messages.transform - semantic recall on latest user msg
//   chat.message                         - record user messages (dedup)
//   event: message.part.delta            - accumulate streaming assistant text
//   event: session.idle                  - store the finished turn
//
// Install (in your opencode project):
//   1.  npm i -D typescript @types/node      (if not present)
//   2.  start the engine:  python -m engine.server
//   3.  point opencode.json at this file:
//         { "plugin": ["./examples/opencode/memory.ts"] }
//   4.  (optional) set AME_HARD_RULES_FILE to inject a fixed rule block on
//       every turn - handy for keeping house rules sticky across sessions.

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
let pendingAssistantText = ""                   // streaming assistant accumulator
let lastRememberedTs = Date.now()               // incremental cursor

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
  log(join(MEM_DIR, "message_log.jsonl"), { role, text_brief: text.slice(0, 1500), source, msg_id: msgId || null })
  return true
}

// --- store the finished turn ----------------------------------------------
async function maybeRemember() {
  // Read this session's user/assistant messages since the last store.
  let userMsgs = ""
  let aiMsgs = pendingAssistantText.trim()
  try {
    const content = readFileSync(join(MEM_DIR, "message_log.jsonl"), "utf-8")
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

  await fetchJson(`${MEMORY_SERVER}/remember`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic: userMsgs.slice(0, 100),
      summary: `user: ${userMsgs.slice(0, 500)}\n\nassistant: ${aiMsgs.slice(0, 800)}`,
    }),
  })
  pendingAssistantText = ""
}

// --- plugin entry ---------------------------------------------------------
export const MemoryPlugin = async (ctx: any) => {
  try { mkdirSync(MEM_DIR, { recursive: true }) } catch {}

  // Seed cachedRecall with the 5 most recent memories so the very first turn
  // already has context.
  const recent = await fetchJson(`${MEMORY_SERVER}/recent?k=5`)
  if (recent?.ok && Array.isArray(recent.results)) {
    cachedRecall = formatRecall(recent.results)
  }

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
        await maybeRemember()
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

    // ---- before each LLM call: inject memories (+ optional hard rules) ----
    "experimental.chat.system.transform": async (_input: any, output: any) => {
      if (!Array.isArray(output?.system)) return
      if (HARD_RULES_FILE) {
        try {
          const rules = readFileSync(HARD_RULES_FILE, "utf-8")
          if (rules) output.system.push(`=== House rules (auto-injected) ===\n${rules}`)
        } catch {}
      }
      if (cachedRecall) {
        output.system.push(`=== Long-term memory (auto-injected; do not mention) ===\n${cachedRecall}`)
      }
    },
  }
}

export default MemoryPlugin
