---
description: Tasks the Qwen agent to generate code or perform refactoring using the TDD Shackle protocol.
---

# Workflow: Code Generation (Qwen Coder)

Principle: **If logic is more than 10 lines long, delegate it to Coder. Every implementation MUST be bound by the [TDD Shackle](../TDD.md).**

1. **Assess Task Complexity:**
  - **Standard Tasks**: For typical logic, boilerplate, or quick fixes, use `qwen_coder`.
  - **High-Logic / Specialist**: For complex algorithms or major refactoring, use `qwen_coder_25` (Coder-Next).

2. **The TDD Protocol (Mandatory):**
  - **Phase 1: RED (The Test)**: Before writing logic, call `qwen_coder` to write an **asymmetrically simple test**. Run it. **It must FAIL.** 
  - **Phase 2: GREEN (The Code)**: Once the test fails, use `qwen_coder` or `qwen_coder_25` to implement the logic that satisfies the test. Provide the failing test as context.
  - **Phase 3: REFACTOR (The Audit)**: After the test turns GREEN, call `qwen_audit` to clean up the implementation without breaking the test.

3. **Repository Change Process (Surgical Precision):**
  - Personally verify the code against the requirements.
  - Apply changes using partial editing (`replace_file_content` or `multi_replace_file_content`) to avoid full overwrites.
  - **No Placeholders**: Never accept code with `// ... implementation here`. Ensure complete, functional blocks.

4. **Final Verification:**
  - Run the final test suite. 
  - Ensure `npx tsc --noEmit` (or relevant compiler check) passes.
  - **Mantra**: No RED, no GREEN. No GREEN, no commit.
