from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction

class WebTerminalLogger(BaseCallbackHandler):
    """
    Custom callback handler that intercepts the agent's internal thoughts, 
    actions, and tool executions to stream them to the frontend web terminal 
    in real-time.
    """
    def __init__(self):
        self.logs: List[str] = []

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """
        Triggered when the Orchestrator or Specialist begins a reasoning chain.
        Silenced to prevent repetitive log spam in the UI.
        """
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """
        Triggered precisely when the agent decides to invoke a specific tool.
        Captures the tool name and the exact parameters being injected.
        """
        self.logs.append(f"Agent invoking tool: [{action.tool}]")
        self.logs.append(f"Injected parameters: {action.tool_input}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """
        Triggered when the invoked Python function (tool) finishes its execution 
        and returns the mathematical or structural results to the agent.
        """
        self.logs.append("System observation: Tool execution completed successfully.")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """
        Triggered when the agent concludes its reasoning chain and formulates 
        the final response.
        Silenced to prevent repetitive log spam in the UI.
        """
        pass