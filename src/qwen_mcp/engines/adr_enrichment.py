"""Phase 6: Async Enrichment Pipeline for ADR decisions.

Non-blocking MCP integration with queue-based processing,
LRU caching, and graceful degradation.
"""
import asyncio
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation.
    
    Uses OrderedDict to maintain access order and automatically
    evict oldest entries when max_size is exceeded.
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache (default: 100)
        """
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, marking it as recently used.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found
        """
        if key not in self._cache:
            return None
        
        # Move to end (mark as recently used)
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache, evicting oldest if necessary.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            # Add new item
            self._cache[key] = value
            
            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()


class ADREnrichmentPipeline:
    """Async pipeline for ADR decision enrichment.
    
    Features:
    - Queue-based processing for non-blocking operations
    - LRU caching to avoid redundant enrichment
    - Graceful degradation when MCP client unavailable
    - Background worker for automatic processing
    
    Usage:
        pipeline = ADREnrichmentPipeline()
        await pipeline.enqueue_decision("decision-id")
        await pipeline.start_background_worker()
    """
    
    def __init__(self, mcp_client: Optional[Any] = None):
        """Initialize enrichment pipeline.
        
        Args:
            mcp_client: Optional MCP client for remote enrichment.
                       If None, uses local enrichment only.
        """
        self.mcp_client = mcp_client
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cache: LRUCache = LRUCache(max_size=100)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def enqueue_decision(self, decision_id: str) -> None:
        """Queue a decision for async enrichment.
        
        Args:
            decision_id: UUID of the decision to enrich
        """
        await self.queue.put(decision_id)
        logger.debug(f"Enqueued decision for enrichment: {decision_id}")
    
    async def process_queue_once(self) -> int:
        """Process a single item from the queue.
        
        Returns:
            Number of items processed (0 or 1)
        """
        if self.queue.empty():
            return 0
        
        decision_id = await self.queue.get()
        
        try:
            # Check cache first
            if decision_id in self.cache:
                logger.debug(f"Cache hit for decision: {decision_id}")
                return 1
            
            # Perform enrichment
            if self.mcp_client:
                await self._enrich_from_mcp(decision_id)
            else:
                await self._enrich_local(decision_id)
            
            # Cache the result
            self.cache.set(decision_id, {"enriched_at": datetime.now().isoformat()})
            logger.info(f"Enriched decision: {decision_id}")
            
            return 1
            
        except Exception as e:
            logger.warning(f"Enrichment failed for {decision_id}: {e}")
            # Still mark as done to prevent infinite loop
            return 1
            
        finally:
            self.queue.task_done()
    
    async def process_queue(self) -> None:
        """Background worker for processing enrichment queue.
        
        Runs continuously until stopped. Use start_background_worker()
        to run this as a background task.
        """
        self._running = True
        
        while self._running:
            try:
                # Wait for item with timeout
                try:
                    decision_id = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                try:
                    # Check cache first
                    if decision_id in self.cache:
                        logger.debug(f"Cache hit for decision: {decision_id}")
                        continue
                    
                    # Perform enrichment
                    if self.mcp_client:
                        await self._enrich_from_mcp(decision_id)
                    else:
                        await self._enrich_local(decision_id)
                    
                    # Cache the result
                    self.cache.set(decision_id, {"enriched_at": datetime.now().isoformat()})
                    logger.info(f"Enriched decision: {decision_id}")
                    
                except Exception as e:
                    logger.warning(f"Enrichment failed for {decision_id}: {e}")
                    # Continue processing other items
                    
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info("Enrichment pipeline worker stopped")
                break
            except Exception as e:
                logger.error(f"Unexpected error in enrichment worker: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _enrich_from_mcp(self, decision_id: str) -> None:
        """Enrich decision using MCP client (remote enrichment).
        
        Args:
            decision_id: UUID of the decision to enrich
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not available")
        
        # TODO: Implement MCP-based enrichment
        # This would call MCP tools to analyze the decision
        # and add metadata like code links, metrics, etc.
        logger.debug(f"MCP enrichment for {decision_id} (placeholder)")
    
    async def _enrich_local(self, decision_id: str) -> None:
        """Enrich decision using local analysis (fallback).
        
        Args:
            decision_id: UUID of the decision to enrich
        """
        # Local enrichment logic - can use:
        # - Tree-sitter for code analysis
        # - Static analysis for dependency tracking
        # - Pattern matching for ADR detection
        
        logger.debug(f"Local enrichment for {decision_id} (placeholder)")
    
    async def start_background_worker(self) -> asyncio.Task:
        """Start the background enrichment worker.
        
        Returns:
            The asyncio.Task running the worker
        """
        if self._worker_task and not self._worker_task.done():
            logger.warning("Background worker already running")
            return self._worker_task
        
        self._worker_task = asyncio.create_task(self.process_queue())
        logger.info("Started ADR enrichment background worker")
        return self._worker_task
    
    async def stop_background_worker(self) -> None:
        """Stop the background enrichment worker gracefully."""
        self._running = False
        
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        self._worker_task = None
        logger.info("Stopped ADR enrichment background worker")
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued items to be processed.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if queue was emptied, False if timeout occurred
        """
        try:
            await asyncio.wait_for(
                self.queue.join(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def get_queue_size(self) -> int:
        """Return number of items waiting in queue."""
        return self.queue.qsize()
    
    def get_cache_size(self) -> int:
        """Return number of items in cache."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary with queue size, cache size, and worker status
        """
        return {
            "queue_size": self.queue.qsize(),
            "cache_size": len(self.cache),
            "cache_max_size": self.cache.max_size,
            "worker_running": self._running,
            "worker_task_done": self._worker_task.done() if self._worker_task else True
        }
