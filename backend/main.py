import os
import tempfile
import sys
import webbrowser
import threading
from fastapi.responses import FileResponse

# 1. BLINDAJE DE LA CONSOLA (Para modo --windowed)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Usamos try/except porque os.devnull a veces no soporta 'reconfigure'
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# Importaciones de tu proyecto
from agents.orchestrator import orchestrator_executor
from utils.callbacks import WebTerminalLogger

# 2. FUNCIÓN PARA ENCONTRAR ARCHIVOS EN EL .EXE
def obtener_ruta_recurso(ruta_relativa):
    """
    Obtiene la ruta absoluta al recurso. 
    Funciona tanto en desarrollo como cuando está empaquetado en el .exe.
    """
    try:
        # PyInstaller crea una carpeta temporal y guarda la ruta en sys._MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Si no estamos en el .exe, usamos la ruta normal
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, ruta_relativa)


# Inicializamos la Aplicación Web
app = FastAPI(title="PR-SmartSolver API")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. ENDPOINT PARA MOSTRAR LA INTERFAZ GRÁFICA
@app.get("/")
async def servir_interfaz():
    """
    Este endpoint sirve tu archivo index.html cuando el usuario abre la URL base.
    """
    # Buscamos el HTML en la raíz, ya que en tu comando usaste:
    # --add-data "...frontend\index.html;."
    ruta_html = obtener_ruta_recurso("index.html")
    
    if os.path.exists(ruta_html):
        return FileResponse(ruta_html)
    return {"error": f"No se encontró el archivo HTML en: {ruta_html}"}


class QueryRequest(BaseModel):
    prompt: str

@app.post("/api/v1/solve")
async def solve_thermo_endpoint(request: QueryRequest):
    """
    Recibe la petición del frontend, ejecuta los agentes, recoge los logs y la gráfica,
    y devuelve la estructura JSON maestra.
    """
    logger = WebTerminalLogger()
    
    try:
        response = orchestrator_executor.invoke(
            {"input": request.prompt},
            config={"callbacks": [logger]}
        )
        
        agent_text = response['output']
        
        plotly_dict = {}
        ruta_segura = os.path.join(tempfile.gettempdir(), "latest_plot.json")
        
        if os.path.exists(ruta_segura):
            with open(ruta_segura, "r", encoding="utf-8") as f:
                plotly_dict = json.load(f)
            os.remove(ruta_segura)
            
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


# 4. FUNCIÓN PARA ABRIR EL NAVEGADOR
def abrir_navegador():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    import uvicorn
    
    # Programamos que el navegador se abra 1.5 segundos DESPUÉS de ejecutar este bloque.
    # Esto le da tiempo a uvicorn para encender el servidor.
    threading.Timer(1.5, abrir_navegador).start()
    
    print("🤖 Iniciando Servidor PR-SmartSolver en el puerto 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
