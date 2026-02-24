---
description: Advanced administrative procedures and resource management (models, billing, environment token monitoring).
---

# ⚙️ Workflow: Qwen Engine Administration (Admin & Intelligence)

Principle: **Mandatory ROI verification and budget monitoring are key to success and protecting the account from unexpected costs.**

1. **Financial Quota Control:**
   - Report DuckDB usage at least once per day. Call the system tool `qwen_usage_report`, which aggregates tokens by dates and projects. Provide the user with a quick overview of this report.
2. **Intelligence Environment Configuration (Models):**
   - If certainty of assigning the correct model is required - check available LLMs using `qwen_list_available_models`.
   - If you identify a new or desired model, set it using `qwen_set_model(role, model_id)`. Roles are: `strategist` (for `architect` and `audit`), `coder` (for writing `qwen_coder_25`), and `scout` (for file tree mapping, context searching).
3. **Model Library Refresh:**
   - To force a passive refresh to search for SOTA models from the API, use: `qwen_refresh_models`.
