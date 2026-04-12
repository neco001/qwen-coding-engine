#!/usr/bin/env python3
"""
Pre-commit hook for Anti-Degradation System.

Validates code changes against functional snapshots to prevent regressions.
Must complete within 3 seconds latency requirement.

Exit codes:
    0 = pass (no regression detected)
    1 = block (regression detected in production mode)
    2 = error (execution error)

Environment variables:
    ANTI_DEGRADATION_SHADOW_MODE=true  - Warnings only, no blocking
    ANTI_DEGRADATION_PRODUCTION=true   - Blocking mode (default)
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.git_diff_parser import GitDiffParser
from src.graph.snapshot import FunctionalSnapshotGenerator
from src.qwen_mcp.anti_degradation_config import get_config


class PreCommitHook:
    """Pre-commit hook orchestrator for anti-degradation checks."""

    AUDIT_LOG_PATH = Path(".anti_degradation/audit_history.jsonl")
    SNAPSHOT_DIR = Path(".anti_degradation/snapshots")
    LATENCY_LIMIT_SECONDS = 3.0

    def __init__(self, repo_path: str = "."):
        self.repo_path: str = repo_path
        self.start_time: float = 0.0
        self.shadow_mode: bool = False
        self.production_mode: bool = False
        self.git_parser: Optional[GitDiffParser] = None
        self.snapshot_generator: Optional[FunctionalSnapshotGenerator] = None
        self.results: Dict[str, Any] = {}

    def _load_environment_config(self) -> None:
        """Load configuration from environment variables."""
        self.shadow_mode = os.environ.get(
            "ANTI_DEGRADATION_SHADOW_MODE", "false"
        ).lower() == "true"
        self.production_mode = os.environ.get(
            "ANTI_DEGRADATION_PRODUCTION", "false"
        ).lower() == "true"

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def _log_audit(self, result: Dict[str, Any]) -> None:
        """Append audit result to JSONL log file."""
        try:
            with open(self.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        except IOError as e:
            self._output_error(f"Failed to write audit log: {e}")

    def _output_summary(self, passed: bool, regressions: list, latency: float) -> None:
        """Print human-readable summary to stdout."""
        status = "✅ PASS" if passed else "❌ BLOCK"
        mode = "SHADOW" if self.shadow_mode else "PRODUCTION"

        print(f"\n{'='*60}")
        print(f"Anti-Degradation Pre-commit Hook [{mode} Mode]")
        print(f"{'='*60}")
        print(f"Status: {status}")
        print(f"Latency: {latency:.3f}s (limit: {self.LATENCY_LIMIT_SECONDS}s)")
        print(f"Files analyzed: {self.results.get('files_analyzed', 0)}")
        print(f"Regressions detected: {len(regressions)}")

        if regressions:
            print(f"\n⚠️  Regression Details:")
            for reg in regressions[:5]:  # Limit output
                print(f"   - {reg.get('file', 'unknown')}: {reg.get('issue', 'unknown')}")
            if len(regressions) > 5:
                print(f"   ... and {len(regressions) - 5} more")

        if self.shadow_mode and regressions:
            print(f"\n⚠️  SHADOW MODE: Would block, but allowing commit")
        elif not self.shadow_mode and regressions:
            print(f"\n🚫  PRODUCTION MODE: Blocking commit due to regression")

        print(f"{'='*60}\n")

    def _output_json(self, result: Dict[str, Any]) -> None:
        """Output structured JSON to stderr for tooling integration."""
        print(json.dumps(result), file=sys.stderr, flush=True)

    def _output_error(self, message: str) -> None:
        """Output error message to both stdout and stderr."""
        print(f"ERROR: {message}", file=sys.stdout)
        print(json.dumps({"error": message, "exit_code": 2}), file=sys.stderr, flush=True)

    async def _run_analysis(self) -> Dict[str, Any]:
        """Run the core analysis asynchronously."""
        self.git_parser = GitDiffParser()
        config = get_config()
        self.snapshot_generator = FunctionalSnapshotGenerator(
            shadow_mode=self.shadow_mode,
            storage_dir=config.snapshots.storage_dir
        )

        # Get staged diff (fast operation)
        # Run get_staged_diff synchronously (not async)
        diff_result = self.git_parser.get_staged_diff()

        if not diff_result or not diff_result.files:
            return {
                "passed": True,
                "regressions": [],
                "files_analyzed": 0,
                "message": "No staged changes detected"
            }

        files_analyzed = len(diff_result.files)

        # Analyze change impact (synchronous method)
        impact_analysis = self.git_parser.analyze_change_impact(diff_result)

        # Load baseline snapshot
        baseline = self.snapshot_generator.load_snapshot(
            Path(self.repo_path or "."), "baseline"
        )

        # Capture current snapshot
        current = await asyncio.wait_for(
            self.snapshot_generator.capture_snapshot(Path(self.repo_path or ".")),
            timeout=1.0
        )

        regressions = []
        if baseline:
            # Compare snapshots
            diff = await asyncio.wait_for(
                self.snapshot_generator.compare_snapshots(baseline, current),
                timeout=0.5
            )
            # Detect regression
            regressions = await asyncio.wait_for(
                self.snapshot_generator.detect_regression(diff),
                timeout=0.5
            )
        else:
            # No baseline - create one
            self.snapshot_generator.save_snapshot(
                current, Path(self.repo_path or "."), "baseline"
            )

        return {
            "passed": len(regressions) == 0,
            "regressions": regressions,
            "files_analyzed": files_analyzed,
            "impact_score": impact_analysis.get("risk_score", 0),
            "message": "Baseline created" if not baseline else ""
        }

    async def execute(self) -> int:
        """Execute the pre-commit hook with latency enforcement."""
        self.start_time = time.time()
        self._load_environment_config()
        self._ensure_directories()

        try:
            # Run analysis with timeout
            self.results = await asyncio.wait_for(
                self._run_analysis(),
                timeout=self.LATENCY_LIMIT_SECONDS - 0.5  # Buffer for logging
            )

            latency = time.time() - self.start_time

            # Check latency constraint
            if latency > self.LATENCY_LIMIT_SECONDS:
                self._output_error(
                    f"Latency exceeded: {latency:.3f}s > {self.LATENCY_LIMIT_SECONDS}s"
                )
                return 2

            # Prepare audit record
            audit_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "latency_seconds": latency,
                "shadow_mode": self.shadow_mode,
                "production_mode": self.production_mode,
                "passed": self.results["passed"],
                "regressions_count": len(self.results["regressions"]),
                "files_analyzed": self.results["files_analyzed"],
                "exit_code": 0
            }

            # Determine exit code
            if self.results["passed"]:
                exit_code = 0
            elif self.shadow_mode:
                exit_code = 0  # Shadow mode never blocks
                audit_record["shadow_mode_override"] = True
            else:
                exit_code = 1  # Block in production mode

            audit_record["exit_code"] = exit_code

            # Log and output results
            self._log_audit(audit_record)
            self._output_summary(
                passed=(exit_code == 0),
                regressions=self.results["regressions"],
                latency=latency
            )
            self._output_json(audit_record)

            return exit_code

        except asyncio.TimeoutError:
            self._output_error("Analysis timed out (>3s)")
            return 2

        except Exception as e:
            self._output_error(f"Unexpected error: {str(e)}")
            return 2


async def main() -> int:
    """Main entry point."""
    hook = PreCommitHook()
    return await hook.execute()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)