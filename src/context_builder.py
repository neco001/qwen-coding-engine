"""
Parallel Context Builder - Async worker pool with token-based chunking and checkpointing.

This module provides:
- Async worker pool with configurable concurrency (max 4 workers MCP-safe)
- Token-based file chunking (estimate before process)
- Checkpoint state machine with atomic JSON writes
- Progressive result streaming
- Timeout handling and resume capability

Architecture:
1. Scan files and estimate tokens
2. Chunk files by token threshold
3. Process chunks in parallel worker pool
4. Stream results progressively
5. Save checkpoint after each chunk
6. Resume from checkpoint on timeout/crash
"""

import asyncio
import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ChunkState:
    """State of a single chunk during processing."""
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str = ""
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    result: Optional[Dict] = None
    processed_at: Optional[str] = None


@dataclass
class CheckpointState:
    """Checkpoint state for resume capability."""
    total_files: int = 0
    processed_files: List[str] = field(default_factory=list)
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    total_tokens: int = 0
    progress: float = 0.0
    started_at: Optional[str] = None
    last_updated: Optional[str] = None
    chunk_states: List[Dict] = field(default_factory=list)


class ParallelContextBuilder:
    """
    Parallel context builder with worker pool, chunking, and checkpointing.
    
    Features:
    - Async worker pool with max 4 concurrent workers (MCP-safe)
    - Token-based file chunking (estimate before process)
    - Checkpoint state machine with atomic JSON writes
    - Progressive result streaming
    - Timeout handling and resume capability
    
    Example:
        builder = ParallelContextBuilder(max_workers=4, chunk_size_tokens=1000)
        results = await builder.build_context(
            ["file1.py", "file2.py"],
            stream_callback=lambda r: print(f"Progress: {r}")
        )
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        chunk_size_tokens: int = 1000,
        checkpoint_file: str = ".context_builder_checkpoint.json",
        timeout_seconds: Optional[float] = None,
        analysis_handler: Optional[Callable[[Dict], Coroutine[Any, Any, Dict]]] = None,
    ):
        """
        Initialize the ParallelContextBuilder.
        
        Args:
            max_workers: Maximum concurrent workers (default 4 for MCP safety)
            chunk_size_tokens: Target tokens per chunk (default 1000)
            checkpoint_file: Path to checkpoint file for resume
            timeout_seconds: Optional timeout for entire operation
            analysis_handler: Optional async callable to analyze chunk content
        """
        self.max_workers = max_workers
        self.chunk_size_tokens = chunk_size_tokens
        self.checkpoint_file = checkpoint_file
        self.timeout_seconds = timeout_seconds
        self.analysis_handler = analysis_handler
        
        self.worker_pool: asyncio.Semaphore = asyncio.Semaphore(max_workers)
        self._checkpoint_lock = asyncio.Lock()
        self._stream_callback: Optional[Callable] = None
        self._state = CheckpointState()
    
    async def estimate_tokens(self, file_paths: List[Path]) -> int:
        """
        Estimate total tokens for a list of files.
        
        Uses heuristic: ~4 characters per token (conservative estimate).
        
        Args:
            file_paths: List of file paths to estimate
            
        Returns:
            Estimated total tokens
        """
        total_tokens = 0
        
        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                # Conservative estimate: 4 chars per token
                estimated = len(content) // 4
                total_tokens += estimated
            except Exception as e:
                logger.warning(f"Failed to estimate tokens for {file_path}: {e}")
        
        return total_tokens
    
    async def _chunk_file_by_tokens(self, file_path: Path) -> List[Dict]:
        """
        Chunk a file by token threshold.
        
        Args:
            file_path: Path to file to chunk
            
        Returns:
            List of chunk dicts with content, path, start_line, end_line
        """
        chunks = []
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            
            # Estimate tokens per line
            total_chars = len(content)
            total_tokens = max(total_chars // 4, 1)  # At least 1 token
            tokens_per_line = total_tokens / max(len(lines), 1)
            
            # Calculate lines per chunk - ensure we get multiple chunks for large files
            lines_per_chunk = max(
                int(self.chunk_size_tokens / max(tokens_per_line, 0.25)),
                1  # Allow single line chunks for very large token counts
            )
            
            # Create chunks
            chunk_id = 0
            for start_idx in range(0, len(lines), lines_per_chunk):
                end_idx = min(start_idx + lines_per_chunk, len(lines))
                chunk_lines = lines[start_idx:end_idx]
                chunk_content = "\n".join(chunk_lines)
                
                chunks.append({
                    "chunk_id": f"{file_path.name}_{chunk_id}",
                    "path": str(file_path),
                    "content": chunk_content,
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                    "estimated_tokens": max(len(chunk_content) // 4, 1),
                })
                chunk_id += 1
            
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            # Return single chunk with error marker
            chunks.append({
                "chunk_id": f"{file_path.name}_error",
                "path": str(file_path),
                "content": f"[Error reading file: {e}]",
                "start_line": 0,
                "end_line": 0,
                "estimated_tokens": 10,
            })
        
        return chunks
    
    async def _process_chunk(self, chunk: Dict) -> Dict:
        """
        Process a single chunk. Uses analysis_handler if provided.
        
        Args:
            chunk: Chunk dict with content, path, etc.
            
        Returns:
            Processed result dict
        """
        # Use analysis_handler if provided for LLM-based analysis
        if self.analysis_handler:
            try:
                analysis_result = await self.analysis_handler(chunk)
                return {
                    "chunk_id": chunk["chunk_id"],
                    "path": chunk["path"],
                    "lines": f"{chunk['start_line']}-{chunk['end_line']}",
                    "tokens": chunk["estimated_tokens"],
                    "analysis": analysis_result,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                logger.error(f"Analysis handler failed for {chunk['chunk_id']}: {e}")
                return {
                    "chunk_id": chunk["chunk_id"],
                    "path": chunk["path"],
                    "lines": f"{chunk['start_line']}-{chunk['end_line']}",
                    "tokens": chunk["estimated_tokens"],
                    "analysis": f"[Analysis failed: {e}]",
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
        
        # Default: just return chunk metadata
        return {
            "chunk_id": chunk["chunk_id"],
            "path": chunk["path"],
            "lines": f"{chunk['start_line']}-{chunk['end_line']}",
            "tokens": chunk["estimated_tokens"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    
    async def _process_chunks_in_parallel(self, chunks: List[Dict]) -> List[Any]:
        """
        Process chunks in parallel with worker pool limit.
        
        Args:
            chunks: List of chunk dicts to process
            
        Returns:
            List of processed results
        """
        async def process_with_semaphore(chunk):
            async with self.worker_pool:
                try:
                    result = await self._process_chunk(chunk)
                    return result
                except Exception as e:
                    logger.error(f"Chunk {chunk['chunk_id']} failed: {e}")
                    return e
        
        tasks = [process_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _read_checkpoint(self) -> Dict:
        """
        Read checkpoint state from file.
        
        Returns:
            Checkpoint state dict or empty dict if not found
        """
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read checkpoint: {e}")
        
        return {}
    
    async def _write_checkpoint(self, state: Dict) -> None:
        """
        Write checkpoint state atomically using temp + rename pattern.
        
        Args:
            state: State dict to save
        """
        async with self._checkpoint_lock:
            try:
                # Add timestamp
                state["last_updated"] = datetime.now(timezone.utc).isoformat()
                
                # Write to temp file first
                fd, temp_path = tempfile.mkstemp(
                    suffix=".tmp",
                    dir=os.path.dirname(self.checkpoint_file) or "."
                )
                
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2, ensure_ascii=False)
                    
                    # Atomic rename
                    os.replace(temp_path, self.checkpoint_file)
                    
                except Exception:
                    # Clean up temp file on failure
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
                    
            except Exception as e:
                logger.error(f"Failed to write checkpoint: {e}")
    
    async def build_context(
        self,
        file_paths: List[str],
        stream_callback: Optional[Callable[[Dict], Coroutine]] = None,
    ) -> List[Dict]:
        """
        Build context from files with parallel processing and checkpointing.
        
        Args:
            file_paths: List of file paths to process
            stream_callback: Optional async callback for progressive results
            
        Returns:
            List of processed results
        """
        self._stream_callback = stream_callback
        results = []
        
        # Convert to Path objects
        paths = [Path(p) for p in file_paths]
        
        # Initialize state
        self._state = CheckpointState(
            total_files=len(paths),
            total_tokens=await self.estimate_tokens(paths),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        
        # Check for existing checkpoint
        existing = await self._read_checkpoint()
        if existing and existing.get("processed_files"):
            logger.info(f"Resuming from checkpoint: {existing.get('completed_chunks', 0)} chunks done")
            self._state.completed_chunks = existing.get("completed_chunks", 0)
            self._state.processed_files = existing.get("processed_files", [])
        
        # Chunk all files
        all_chunks = []
        for path in paths:
            if str(path) in self._state.processed_files:
                logger.info(f"Skipping already processed: {path}")
                continue
            
            file_chunks = await self._chunk_file_by_tokens(path)
            all_chunks.extend(file_chunks)
        
        self._state.total_chunks = len(all_chunks)
        
        # Process chunks with timeout
        try:
            if self.timeout_seconds:
                results = await asyncio.wait_for(
                    self._process_all_chunks(all_chunks, paths),
                    timeout=self.timeout_seconds
                )
            else:
                results = await self._process_all_chunks(all_chunks, paths)
                
        except asyncio.TimeoutError:
            logger.error(f"Processing timed out after {self.timeout_seconds}s")
            # Save checkpoint before re-raising
            await self._save_final_checkpoint()
            raise
        
        # Save final checkpoint
        await self._save_final_checkpoint()
        
        return results
    
    async def _process_all_chunks(self, chunks: List[Dict], paths: List[Path]) -> List[Dict]:
        """Process all chunks and stream results."""
        results = []
        
        for i, chunk in enumerate(chunks):
            try:
                result = await self._process_chunk(chunk)
                results.append(result)
                
                # Update state
                self._state.completed_chunks += 1
                self._state.progress = self._state.completed_chunks / max(self._state.total_chunks, 1)
                
                # Add to processed files if this is the last chunk of a file
                chunk_file = chunk["path"]
                if not any(c["path"] == chunk_file and c["chunk_id"] != chunk["chunk_id"] 
                          for c in chunks[i+1:]):
                    if chunk_file not in self._state.processed_files:
                        self._state.processed_files.append(chunk_file)
                
                # Stream result if callback provided
                if self._stream_callback:
                    await self._stream_callback({
                        "type": "chunk_completed",
                        "chunk_id": chunk["chunk_id"],
                        "progress": self._state.progress,
                        "result": result,
                    })
                
                # Save checkpoint after EVERY chunk for crash recovery
                await self._save_checkpoint_now()
                    
            except Exception as e:
                logger.error(f"Chunk {chunk['chunk_id']} failed: {e}")
                self._state.failed_chunks += 1
                results.append({"error": str(e), "chunk_id": chunk["chunk_id"]})
        
        return results
    
    async def _save_final_checkpoint(self) -> None:
        """Save final checkpoint state."""
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        await self._write_checkpoint(asdict(self._state))
        logger.info(f"Checkpoint saved: {self._state.completed_chunks}/{self._state.total_chunks} chunks")
    
    async def _save_checkpoint_now(self) -> None:
        """Save checkpoint immediately (for crash recovery)."""
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        await self._write_checkpoint(asdict(self._state))
