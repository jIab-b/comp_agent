from omegaconf import DictConfig
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    FileSearchTool,
    ListDirectoryTool,
)
from langchain_community.tools.shell import ShellTool

def load_tools(cfg: DictConfig):
    tools = []
    
    for tool_name in cfg.agent.tools:
        if tool_name == "web_search":
            tools.append(DuckDuckGoSearchRun())
        elif tool_name == "shell":
            tools.append(ShellTool())
        elif tool_name == "file_read":
            tools.append(ReadFileTool())
        elif tool_name == "file_write":
            tools.append(WriteFileTool())
        elif tool_name == "file_search":
            tools.append(FileSearchTool())
        elif tool_name == "list_directory":
            tools.append(ListDirectoryTool())
            
    return tools