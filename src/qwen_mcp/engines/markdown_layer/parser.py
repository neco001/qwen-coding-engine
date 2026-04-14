import re
from typing import List


class MarkdownParser:
    """Parser for markdown content extraction."""

    def extract_section(self, content: str, header_name: str) -> str:
        """
        Extract content under a specific markdown header.

        Args:
            content: The markdown content to parse
            header_name: The header name to search for (case-sensitive)

        Returns:
            The content under the header, or empty string if not found
        """
        if not content:
            return ""

        lines = content.split('\n')
        section_lines = []
        in_section = False

        for line in lines:
            if line.startswith('#'):
                header_match = re.match(r'^(#+)\s+(.+?)\s*$', line)
                if header_match:
                    found_header = header_match.group(2).strip()
                    if found_header == header_name:
                        in_section = True
                        continue
                    elif in_section:
                        break

            if in_section:
                section_lines.append(line)

        return '\n'.join(section_lines).strip()

    def extract_tasks(self, content: str) -> List[str]:
        """
        Extract task items from markdown content.

        Args:
            content: The markdown content to parse

        Returns:
            List of task strings (without the checkbox markers)
        """
        if not content:
            return []

        task_pattern = r'^\s*-\s+\[[ x]\]\s+(.+)$'
        tasks = []

        for line in content.split('\n'):
            match = re.match(task_pattern, line)
            if match:
                tasks.append(match.group(1).strip())

        return tasks
