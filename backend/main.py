from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

from agents.orchestrator import orchestrator_executor
from utils.callbacks import WebTerminalLogger

# 1. Inicializamos la Aplicación Web
app = FastAPI(title="PR-SmartSolver API")

# 2. Configuración CORS (VITAL para que tu archivo HTML pueda hablar con el servidor Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se restringe, pero para desarrollo local es perfecto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Definimos qué forma tiene la petición que llega del Frontend
class QueryRequest(BaseModel):
    prompt: str

# 4. El Endpoint que recibe la llamada desde JavaScript
@app.post("/api/v1/solve")
async def solve_thermo_endpoint(request: QueryRequest):
    """
    Recibe la petición del frontend, ejecuta los agentes, recoge los logs y la gráfica,
    y devuelve la estructura JSON maestra.
    """
    logger = WebTerminalLogger()
    
    try:
        # Ejecutamos el orquestador inyectándole el espía para los logs
        response = orchestrator_executor.invoke(
            {"input": request.prompt},
            config={"callbacks": [logger]}
        )
        
        agent_text = response['output']
        
        # Recogemos el JSON de Plotly
        plotly_dict = {}
        if os.path.exists("latest_plot.json"):
            with open("latest_plot.json", "r") as f:
                plotly_dict = json.load(f)
            os.remove("latest_plot.json")
            
        return {
            "status": "success",
            "agent_text_response": agent_text,
            "terminal_logs": logger.logs,
            "plotly_data": plotly_dict
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "agent_text_response": "Lo siento, hubo un fallo interno en la simulación.",
            "terminal_logs": logger.logs + [f"!! ERROR CRÍTICO: {str(e)}"],
            "plotly_data": {}
        }

# Para correr el servidor localmente
if __name__ == "__main__":
    import uvicorn
    print("🤖 Iniciando Servidor PR-SmartSolver en el puerto 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)






'''
from agents.orchestrator import orchestrator_executor

# =====================================================================
# EXECUTION LOOP (CHATBOT MODE)
# =====================================================================
if __name__ == "__main__":
    print("🤖 PR-SmartSolver Agent System Online")
    print("Escribe tu problema de termodinámica o 'salir' para finalizar.\n")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Cerrando el simulador. ¡Hasta luego!")
            break
            
        print("\n[Orquestador analizando enunciado...]")
        try:
            # El Orquestador toma el control y decide a qué Especialista llamar
            response = orchestrator_executor.invoke({"input": user_input})
            print(f"\nAgent:\n{response['output']}\n")
        except Exception as e:
            print(f"\n❌ Error durante la ejecución: {e}\n")
'''