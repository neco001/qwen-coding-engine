# The Lachman Protocol: Qwen Engineering Engine

**Stop building apps by trial and error. Start shipping them by design.**

The Lachman Protocol is not just another "MCP server" for your collection. It is a high-performance engineering engine that turns your AI assistant (Claude, Antigravity, Cursor) into a production-ready software factory. 

By offloading heavy architectural planning and raw coding to specialized Qwen models, you stop the "two steps forward, one step back" dance and start delivering finished applications.

### 🛑 This is NOT for you if:
- You just want an AI to chat with or write your emails.
- You enjoy manually copy-pasting code because you don't trust agents.
- You have an unlimited budget to blow $50/day on "lazy" models that truncate your code with `// ... implementation here`.

### 💚 This IS for you if:
- You are a **"Vibecoder"** (building complex apps primarily via AI chat) and you're sick of the "Fix one feature, break two others" cycle.
- You're a **Senior Developer** who wants to delegate the "dirty work"—auditing logs, writing boilerplate, and complex refactoring—to an agent that won't get tired or impatient.
- You want the power of **Qwen 3.5 Plus** and **Qwen 2.5 Coder 32B** at a **fraction of the cost** of GPT-4o or Claude 3.5 Opus.

---

## ❓ The Problem: The "Lazy AI" Ceiling
Current flagship assistants are great, but they have major flaws when tasked with building real software:
1. **Context Amnesia:** They forget your core requirements 10 messages into a debug session.
2. **The Placeholder Trap:** They get lazy and give you snippets instead of functional files.
3. **Hallucination Cascades:** One small error leads to a chain of patches that eventually breaks the entire architecture.

**The Lachman Protocol solves this by hiring Qwen as your "Project Architect & Senior SRE".**

---

## 🏛️ The Core: The Lachman Protocol (LP)
When you initiate a project, the engine doesn't just "guess." It enters a multi-stage **Self-Healing Loop**:

1. **Discovery:** Qwen hires a virtual "Expert Squad" tailored to your specific goal (e.g., Security Auditor, Backend Engineer, UX Strategist).
2. **Architecting:** These roles debate and produce a **Detailed Project Blueprint**. 
3. **Self-Verification:** A separate "Verifier" model audits the blueprint. If it finds a flaw, the engine triggers a self-correction loop (up to 3 times) to fix the design *before* any code is written.

### The result? 
You get a surgical Technical Roadmap. Your primary assistant (Claude/Antigravity) acts as the **Commander**, while the Qwen Engine handles the **Heavy Logistics**.

---

## 🔥 Scenario: From Idea to Reality

### Phase 1: Planning without Hallucination
Instead of saying: *"Build me a CRM"*, you tell your assistant:
> "Plan a CRM with FastAPI and Postgres. **Call `lp_architect`** to generate the blueprint."

**Result:** You get a structured Roadmap + Security Audit + Risk Assessment.

### Phase 2: Full-Scale Implementation / Refactoring
Don't let your main assistant guess the syntax or "hallucinate" the logic. 
> "Take Step 1 of the blueprint and **call `qwen_coder`** to implement the models and database connection. Ensure the logic is complete."

**You can also use it for precise atomic tasks:**
> "In file `auth.py`, **call `qwen_coder`** to refactor the login function to use JWT instead of sessions. Do not use placeholders."

**Result:** You get 100% complete, working Python code. No truncated files, no "implement here" comments.

> "Here are my logs and current file. **Call `qwen_audit`** to find the root cause and fix it."

**Result:** A Senior SRE analysis that finds the memory leak or the null pointer in seconds.

---

## 🛠️ Performance & Strategy

### 🧠 We don't need Ralph
There is a popular method called **The Ralph Loop** (fresh context for every iteration). While interesting for naive agents, the Qwen Engineering Engine is designed differently. 

Because we use **The Lachman Protocol** (Spec -> Code -> Audit), we rely on **State & Blueprint Persistence** rather than a fresh start. We can tell Ralph to stay in Springfield—we have an Architect in the basement.

---

## 🛠️ The Arsenal (Included Tools)

| Tool | Role |
| :--- | :--- |
| `lp_architect` | **The Architect**: Multi-expert planner with a self-healing loop for blueprints. |
| `qwen_coder` | **The Senior Dev**: Writes long-form, functional code (Qwen 3.5 Plus). |
| `qwen_coder_25` | **The Specialist**: Tackles complex logic using Qwen 2.5 Coder 32B. |
| `qwen_audit` | **The SRE**: Audits logs and code to find and fix critical bugs. |
| `qwen_read_file` | **The Context Loader**: Securely reads files from your workspace for context. |
| `qwen_list_files` | **The Explorer**: Scans your project structure. |

---

## 📦 Installation & Setup

### 1. Local Development Setup (Quick Start)
Since the package is in development, install it in editable mode:

```bash
git clone <this-repo-url>
cd qwen-mcp
uv pip install -e .
```

### 2. Configure Environment
Create a `.env` file or set the following variables:
```bash
export DASHSCOPE_API_KEY=your_key_here
# Optional: for local mode
# export OLLAMA_BASE_URL=http://localhost:11434/v1
```

### 3. Google Antigravity / Claude Desktop
Add this to your MCP configuration (set `cwd` to the project directory):
```json
{
  "mcpServers": {
    "qwen-mcp": {
      "command": "uv",
      "args": ["run", "qwen-mcp"],
      "env": {
        "DASHSCOPE_API_KEY": "your_api_key_here",
        "LP_MAX_RETRIES": "3"
      }
    }
  }
}
```

---
**License: MIT**
**Build apps, not just conversations.**
