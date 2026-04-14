import re


class SectionManager:
    """Manages sections in markdown content."""

    _COMPLETED_HEADER_PATTERN = re.compile(r'(## Completed\n?)', re.IGNORECASE)
    _MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')
    _CHANGELOG_PATTERN = re.compile(r'(#+ CHANGELOG\n+)', re.IGNORECASE)

    def archive_task(self, content: str, task_id: str) -> str:
        task_pattern = rf'- \[ \] (.+?) \[id:{re.escape(task_id)}\]'
        match = re.search(task_pattern, content)
        
        if not match:
            raise ValueError(f"Task with id {task_id} not found")
        
        task_text = match.group(1)
        old_task = match.group(0)
        new_task = f"- [x] {task_text} [id:{task_id}]"
        
        content = re.sub(rf'{re.escape(old_task)}\n?', '', content, count=1)
        
        completed_match = self._COMPLETED_HEADER_PATTERN.search(content)
        
        if completed_match:
            insert_pos = completed_match.end()
            content = content[:insert_pos] + new_task + '\n' + content[insert_pos:]
        else:
            content = content.rstrip() + '\n\n## Completed\n' + new_task + '\n'
        
        content = self._MULTI_NEWLINE_PATTERN.sub('\n\n', content)
        
        return content

    def append_to_changelog(self, content: str, entry: str) -> str:
        changelog_match = self._CHANGELOG_PATTERN.search(content)
        
        if changelog_match:
            insert_pos = changelog_match.end()
            content = content[:insert_pos] + entry + '\n\n' + content[insert_pos:]
        else:
            if content:
                content = content.rstrip() + '\n\n# CHANGELOG\n\n' + entry + '\n'
            else:
                content = '# CHANGELOG\n\n' + entry + '\n'
                
        content = self._MULTI_NEWLINE_PATTERN.sub('\n\n', content)
        return content
