import asyncio
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest
from pytest_asyncio import fixture

from src.context_builder import ParallelContextBuilder


@fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = []
    for i in range(10):
        file_path = temp_dir / f"file_{i}.txt"
        content = f"Content of file {i} " * 50  # ~250 tokens
        file_path.write_text(content)
        files.append(file_path)
    return files


class TestParallelContextBuilder:
    
    def test_initialization(self):
        """Test initialization with default parameters."""
        builder = ParallelContextBuilder(
            max_workers=4,
            chunk_size_tokens=1000,
            checkpoint_file="test_checkpoint.json"
        )
        
        assert builder.max_workers == 4
        assert builder.chunk_size_tokens == 1000
        assert builder.checkpoint_file == "test_checkpoint.json"
        assert builder.worker_pool._value == 4  # Semaphore initial value
    
    @pytest.mark.asyncio
    async def test_estimate_tokens(self, sample_files):
        """Test token estimation functionality."""
        builder = ParallelContextBuilder(max_workers=4, chunk_size_tokens=1000)
        
        # Estimate tokens for sample files
        total_tokens = await builder.estimate_tokens(sample_files)
        
        assert isinstance(total_tokens, int)
        assert total_tokens > 0
        
        # Each file has ~250 tokens, 10 files = ~2500 tokens
        assert total_tokens >= 2000  # Allow some variance in estimation
    
    @pytest.mark.asyncio
    async def test_chunk_file_by_tokens(self, sample_files):
        """Test file chunking based on token count."""
        builder = ParallelContextBuilder(max_workers=4, chunk_size_tokens=300)
        
        chunks = await builder._chunk_file_by_tokens(sample_files[0])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert 'content' in chunk
            assert 'path' in chunk
            assert 'start_line' in chunk
            assert 'end_line' in chunk
    
    @pytest.mark.asyncio
    async def test_worker_pool_concurrency_limit(self, sample_files):
        """Test that worker pool respects concurrency limit."""
        builder = ParallelContextBuilder(max_workers=2, chunk_size_tokens=1000)
        
        # Mock processing function to track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0
        
        async def mock_process_chunk(chunk):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            concurrent_calls -= 1
            return {"processed": True, "chunk": chunk["path"]}
        
        # Replace the actual processing method
        builder._process_chunk = mock_process_chunk
        
        # Process chunks
        chunks = []
        for file_path in sample_files[:3]:  # Use 3 files to ensure multiple workers
            file_chunks = await builder._chunk_file_by_tokens(file_path)
            chunks.extend(file_chunks)
        
        results = await builder._process_chunks_in_parallel(chunks)
        
        # Verify concurrency was limited
        assert max_concurrent <= 2
        assert len(results) == len(chunks)
    
    @pytest.mark.asyncio
    async def test_checkpoint_state_machine_atomic_writes(self, temp_dir):
        """Test checkpoint state machine with atomic JSON writes."""
        checkpoint_file = temp_dir / "checkpoint.json"
        builder = ParallelContextBuilder(
            max_workers=4,
            chunk_size_tokens=1000,
            checkpoint_file=str(checkpoint_file)
        )
        
        # Test initial state
        initial_state = await builder._read_checkpoint()
        assert initial_state == {}
        
        # Test writing checkpoint
        test_state = {
            "processed_files": ["file1.txt", "file2.txt"],
            "total_tokens": 500,
            "progress": 0.5,
            "completed_chunks": 10
        }
        
        await builder._write_checkpoint(test_state)
        
        # Verify checkpoint was written atomically
        read_state = await builder._read_checkpoint()
        assert read_state == test_state
        
        # Verify temporary file cleanup
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0
    
    @pytest.mark.asyncio
    async def test_progressive_result_streaming(self, sample_files):
        """Test progressive result streaming during processing."""
        builder = ParallelContextBuilder(max_workers=2, chunk_size_tokens=1000)
        
        # Track streamed results
        streamed_results = []
        
        async def mock_stream_callback(result):
            streamed_results.append(result)
        
        # Mock processing to simulate streaming
        original_process = builder._process_chunk
        
        async def mock_process_chunk(chunk):
            await asyncio.sleep(0.01)  # Small delay to allow streaming
            return {"chunk_path": str(chunk["path"]), "size": len(chunk["content"])}
        
        builder._process_chunk = mock_process_chunk
        
        # Process with streaming callback
        results = await builder.build_context(
            [str(f) for f in sample_files[:3]],
            stream_callback=mock_stream_callback
        )
        
        # Verify results were streamed progressively
        assert len(streamed_results) > 0
        assert len(results) == len(streamed_results)
    
    @pytest.mark.asyncio
    async def test_timeout_handling_and_resume(self, sample_files, temp_dir):
        """Test timeout handling and resume capability."""
        checkpoint_file = temp_dir / "timeout_checkpoint.json"
        builder = ParallelContextBuilder(
            max_workers=2,
            chunk_size_tokens=1000,
            checkpoint_file=str(checkpoint_file),
            timeout_seconds=0.1
        )
        
        # Create a slow processing function to trigger timeout
        processed_chunks = []
        
        async def slow_process_chunk(chunk):
            if len(processed_chunks) < 2:  # Let first 2 complete normally
                await asyncio.sleep(0.01)
                processed_chunks.append(chunk["path"])
                return {"processed": True, "chunk": chunk["path"]}
            else:
                # This will cause timeout
                await asyncio.sleep(0.5)
                processed_chunks.append(chunk["path"])
                return {"processed": True, "chunk": chunk["path"]}
        
        builder._process_chunk = slow_process_chunk
        
        # Process files - should handle timeout gracefully
        try:
            results = await builder.build_context([str(f) for f in sample_files[:5]])
        except asyncio.TimeoutError:
            pass  # Expected behavior
        
        # Verify checkpoint contains partial progress
        checkpoint = await builder._read_checkpoint()
        assert "processed_files" in checkpoint
        assert "progress" in checkpoint
    
    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, sample_files, temp_dir):
        """Test resuming processing from checkpoint."""
        checkpoint_file = temp_dir / "resume_checkpoint.json"
        builder = ParallelContextBuilder(
            max_workers=2,
            chunk_size_tokens=800,
            checkpoint_file=str(checkpoint_file)
        )
        
        # Create initial checkpoint with partial progress
        initial_checkpoint = {
            "processed_files": [str(sample_files[0])],
            "completed_chunks": 1,
            "total_chunks": 10,
            "progress": 0.1
        }
        await builder._write_checkpoint(initial_checkpoint)
        
        # Mock processing to verify resume behavior
        processed_after_resume = []
        
        async def mock_process_chunk(chunk):
            processed_after_resume.append(chunk["path"])
            return {"processed": True, "chunk": chunk["path"]}
        
        builder._process_chunk = mock_process_chunk
        
        # Build context - should resume from checkpoint
        results = await builder.build_context([str(f) for f in sample_files[:3]])
        
        # Verify processing resumed correctly
        checkpoint_after = await builder._read_checkpoint()
        assert checkpoint_after["completed_chunks"] > 1  # Progress was made
        assert len(processed_after_resume) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_worker_pool(self, sample_files):
        """Test error handling when workers encounter exceptions."""
        builder = ParallelContextBuilder(max_workers=2, chunk_size_tokens=1000)
        
        # Mock processing to raise exception for some chunks
        call_count = 0
        
        async def error_prone_process_chunk(chunk):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every third call fails
                raise ValueError(f"Processing failed for {chunk['path']}")
            return {"processed": True, "chunk": chunk["path"]}
        
        builder._process_chunk = error_prone_process_chunk
        
        # Should handle errors gracefully and continue processing
        results = await builder.build_context([str(f) for f in sample_files[:6]])
        
        # Verify some results were successful despite errors
        assert len([r for r in results if not isinstance(r, Exception)]) > 0
    
    @pytest.mark.asyncio
    async def test_empty_file_list_handling(self):
        """Test handling of empty file lists."""
        builder = ParallelContextBuilder(max_workers=4, chunk_size_tokens=1000)
        
        results = await builder.build_context([])
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_single_file_processing(self, sample_files):
        """Test processing of single file."""
        builder = ParallelContextBuilder(max_workers=4, chunk_size_tokens=1000)
        
        results = await builder.build_context([str(sample_files[0])])
        
        assert len(results) >= 1
        assert all(isinstance(r, dict) for r in results)
    
    @pytest.mark.asyncio
    async def test_large_file_chunking(self, temp_dir):
        """Test chunking of very large files."""
        large_file = temp_dir / "large_file.txt"
        # Create multi-line content (each line ~20 chars = ~5 tokens)
        # 1000 lines * 5 tokens = 5000 tokens total, should create ~10 chunks at 500 tokens each
        large_content = "\n".join([f"Line {i}: Large file content here" for i in range(1000)])
        large_file.write_text(large_content)
        
        builder = ParallelContextBuilder(max_workers=4, chunk_size_tokens=500)
        
        chunks = await builder._chunk_file_by_tokens(large_file)
        
        assert len(chunks) > 1  # Should be chunked into multiple parts
        for chunk in chunks:
            assert len(chunk["content"]) > 0
            assert chunk["path"] == str(large_file)
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery_on_crash(self, sample_files, temp_dir):
        """Test checkpoint recovery when process crashes."""
        checkpoint_file = temp_dir / "crash_checkpoint.json"
        builder = ParallelContextBuilder(
            max_workers=2,
            chunk_size_tokens=1000,
            checkpoint_file=str(checkpoint_file)
        )
        
        # Simulate crash by stopping after partial completion
        processed_count = 0
        
        async def crash_after_partial(chunk):
            nonlocal processed_count
            processed_count += 1
            # Crash BEFORE processing 3rd chunk (after 2 completed)
            if processed_count == 3:
                raise KeyboardInterrupt("Simulated crash")
            return {"processed": True, "chunk": chunk["path"]}
        
        builder._process_chunk = crash_after_partial
        
        # First run - should crash
        with pytest.raises(KeyboardInterrupt):
            await builder.build_context([str(f) for f in sample_files[:5]])
        
        # Verify checkpoint was saved before crash
        # Note: checkpoint is saved AFTER each chunk completes, so we expect 2 (not 3)
        checkpoint = await builder._read_checkpoint()
        assert "completed_chunks" in checkpoint
        assert checkpoint["completed_chunks"] >= 2  # At least 2 chunks completed before crash
    
    @pytest.mark.asyncio
    async def test_concurrent_access_to_checkpoint(self, temp_dir):
        """Test concurrent access to checkpoint file."""
        checkpoint_file = temp_dir / "concurrent_checkpoint.json"
        builder = ParallelContextBuilder(
            max_workers=4,
            chunk_size_tokens=1000,
            checkpoint_file=str(checkpoint_file)
        )
        
        # Simulate concurrent checkpoint updates
        async def update_checkpoint(i):
            state = {"task": i, "timestamp": asyncio.get_event_loop().time()}
            await builder._write_checkpoint(state)
            await asyncio.sleep(0.01)  # Small delay
            return await builder._read_checkpoint()
        
        tasks = [update_checkpoint(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify no corruption occurred
        assert len(results) == 10
        # All results should be valid dictionaries
        assert all(isinstance(r, dict) for r in results)
