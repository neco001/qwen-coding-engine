# 🔧 The Repair Protocol (The Audit Triad)

When you encounter a bug, a regression, or a complex logic error, don't just patch it. Use the **Repair Protocol** to identify the root cause and update the architecture.

## 🛠️ Operational Steps:

1.  **Audit (The Analyst)**:
    -   Call `qwen_audit(logs_or_code, current_context)`.
    -   Focus on **Root Cause Analysis (RCA)**. 
    -   *Goal*: Understand exactly WHY it broke, not just WHERE.

2.  **Strategize (The Architect)**:
    -   Feed the Audit Verdict into the existing Project Blueprint.
    -   Ask the Architect to update the **Roadmap** and **Risk Assessment**.
    -   *Goal*: Ensure the fix doesn't violate the core architecture.

3.  **Implement (The coder & TDD)**:
    -   Follow the mandatory **[TDD Shackle](./TDD.md)**.
    -   Write a failing test that reproduces the bug.
    -   Implement the fix until the test is GREEN.

---

**Mantra: A bug is an architect's failure. Fix the plan, then the code.**
