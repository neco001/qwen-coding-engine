from qwen_mcp.engines.io_layer.path_resolver import PathResolver
from qwen_mcp.engines.io_layer.file_handler import FileHandler
from qwen_mcp.engines.markdown_layer.sections import SectionManager


class DecisionLogOrchestrator:
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver

    def archive_task(self, decision_id: str) -> bool:
        content = FileHandler.read_text(self.path_resolver.backlog_path)

        if content is None:
            return False

        try:
            section_manager = SectionManager()
            updated_content = section_manager.archive_task(content, decision_id)
            FileHandler.write_atomic(self.path_resolver.backlog_path, updated_content)
            return True
        except ValueError:
            return False

    def append_to_changelog(self, entry: str) -> None:
        content = FileHandler.read_text(self.path_resolver.changelog_path)

        # Allow creating new changelog if missing
        if content is None:
            content = ""

        section_manager = SectionManager()
        updated_content = section_manager.append_to_changelog(content, entry)
        FileHandler.write_atomic(self.path_resolver.changelog_path, updated_content)
