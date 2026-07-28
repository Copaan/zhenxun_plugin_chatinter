"""Fixed tools for the superuser Agent runtime."""

from ...llm_compat import ToolExecutable
from .artifact_tools import ArtifactReadTool
from .file_tools import (
    ApplyPatchTool,
    ListDirTool,
    ReadFileTool,
    ReplaceInFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from .plan_tools import PlanTool
from .shell_tools import ShellCommandTool

SUPERUSER_CORE_TOOL_NAMES = (
    "read_file",
    "list_dir",
    "search_files",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "shell_command",
    "artifact_read",
    "plan",
)


def build_superuser_tools() -> dict[str, ToolExecutable]:
    tools = (
        ReadFileTool(),
        ListDirTool(),
        SearchFilesTool(),
        WriteFileTool(),
        ReplaceInFileTool(),
        ApplyPatchTool(),
        ShellCommandTool(),
        ArtifactReadTool(),
        PlanTool(),
    )
    return {tool.name: tool for tool in tools}


__all__ = ["SUPERUSER_CORE_TOOL_NAMES", "build_superuser_tools"]
