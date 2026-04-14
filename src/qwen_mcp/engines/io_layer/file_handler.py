import os
import tempfile
from pathlib import Path
from typing import Optional


class FileHandler:
    """Handles file I/O operations with atomic write guarantees."""

    @staticmethod
    def write_atomic(file_path: Path, content: str) -> None:
        """
        Writes content to a file atomically.

        Uses a temporary file and os.replace to ensure that the target file
        is either fully written or not changed at all, preventing partial writes.

        Args:
            file_path: The target path to write to.
            content: The string content to write.
        """
        target_path = Path(file_path).resolve()
        parent_dir = target_path.parent
        
        # Ensure parent directory exists
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary file in the same directory to ensure atomic replace works
        fd, temp_path = tempfile.mkstemp(dir=str(parent_dir))
        
        try:
            # Write content to temp file
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomically replace the target file with the temp file
            os.replace(temp_path, str(target_path))
        except Exception:
            # Clean up temp file if something goes wrong before replace
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    @staticmethod
    def read_text(file_path: Path) -> Optional[str]:
        """
        Reads text content from a file.

        Args:
            file_path: The path to the file to read.

        Returns:
            The file content as a string, or None if the file does not exist.
        """
        target_path = Path(file_path)
        
        if not target_path.exists():
            return None
            
        return target_path.read_text(encoding='utf-8')
