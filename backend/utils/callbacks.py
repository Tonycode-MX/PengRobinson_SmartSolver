from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction

class WebTerminalLogger(BaseCallbackHandler):
    """
    Callback personalizado que intercepta los pensamientos y acciones del agente
    para enviarlos a la terminal de la interfaz web en tiempo real.
    """
    def __init__(self):
        self.logs: List[str] = []

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Se ejecuta cuando el Orquestador o Especialista empiezan a pensar."""
        self.logs.append("Inicializando cadena de razonamiento cognitivo...")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Se ejecuta justo cuando el agente decide usar una herramienta."""
        self.logs.append(f"Agente invoca herramienta: [{action.tool}]")
        self.logs.append(f"Parámetros inyectados: {action.tool_input}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Se ejecuta cuando tu código de Python (las herramientas) terminan su trabajo."""
        self.logs.append("Observación del sistema: Ejecución de herramienta completada con éxito.")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Se ejecuta cuando el agente tiene su respuesta final."""
        self.logs.append("Cadena de razonamiento concluida. Preparando payload de respuesta.")