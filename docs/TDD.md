# ⛓️ The TDD Shackle: Quality-First Implementation

This guide defines the mandatory **Test-Driven Development (TDD)** workflow when using the Qwen Engineering Engine. Following this protocol prevents "Hallucination Cascades" and ensures that every line of code serves a verified purpose.

## 🏆 The Core Protocol

Never call `qwen_coder` to implement logic without first having a failing test.

### 🔴 Phase 1: RED (The Test)
1.  **Objective**: Define what "success" looks like.
2.  **Action**: Call `qwen_coder` to write an **asymmetrically simple test**.
    -   *Example Prompt*: "Write a pytest benchmark for a function `calculate_roi` that takes (cost, revenue) and returns percentage. It should handle zero cost by raising ValueError."
3.  **Verification**: Run the test in your terminal. **It must fail.** If it passes, your test is either redundant or testing the wrong thing.

### 🟢 Phase 2: GREEN (The Code)
1.  **Objective**: Pass the test as quickly as possible.
2.  **Action**: Call `qwen_coder` (or `qwen_coder_25` for complex logic) providing the failing test as context.
    -   *Example Prompt*: "Implement the `calculate_roi` function to satisfy this test: [paste test code]."
3.  **Verification**: Run the test again. It must be **GREEN**.

### 🔵 Phase 3: REFACTOR (The Audit)
1.  **Objective**: Clean up the "dirty" implementation without breaking the test.
2.  **Action**: Call `qwen_audit` to review the implementation.
    -   *Focus*: Edge cases, performance, and "Gold Plating" (over-engineering).
3.  **Action**: Apply fixes and ensure the test stays **GREEN**.

---

## 🤖 Roles in the TDD Process

| Role | Tool | Responsibility |
| :--- | :--- | :--- |
| **Architect** | `qwen_architect` | Defines the requirements that need testing. |
| **Coder** | `qwen_coder` | Writes both the **Test** (RED) and the **Implementation** (GREEN). |
| **Auditor** | `qwen_audit` | Verifies the logic and identifies hidden bugs (REFACTOR). |

## Why use the Coder for tests?
Writing tests is "cheap" in terms of tokens but "expensive" in terms of logic. By letting Qwen write the test first:
1.  You verify that the model understands the requirement.
2.  You establish a ground truth before any complex logic is written.
3.  You prevent the model from "cheating" by writing code that only looks correct but doesn't work.

**Mantra: No RED, no GREEN. No GREEN, no commit.**
