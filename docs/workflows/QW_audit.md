---
description: Launches the auditor model for deep reasoning in search of runtime bugs.
---

# Workflow: Error Audit and SRE (Qwen Audit)

Principle: **When code fails in non-obvious ways or the architecture throws unexplained exceptions, before attempting a random fix – assign the task to the Auditor (QwQ).**

1. **Evidentiary Package (Logs and Context):**
  - Copy the console dump reporting the error and the specifically isolated problematic stack trace (especially runtime errors) into the `content` variable.
  - Describe the block's expected behavior at the application logic level in the `context` variable, and also include the content of the code being checked (function).
2. **Audit Assignment:**
  - Call the `qwen_audit` tool. Due to the deep overhead of the analytical process (chain-of-thought), the model will thoroughly check for defects.
3. **Conclusions and Fixes:**
  - Follow the recommendations of the Qwen Audit report.
  - If necessary after implementing fixes using tools (`QW_coder`), report the solution for review and ensure the bug is gone by re-running tests.
