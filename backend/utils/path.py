import os
import sys

def obtener_ruta(ruta_relativa):
    """Obtiene la ruta absoluta al recurso, compatible con desarrollo y PyInstaller"""
    try:
        # PyInstaller guarda la ruta temporal en _MEIPASS
        ruta_base = sys._MEIPASS
    except Exception:
        ruta_base = os.path.abspath(".")

    return os.path.join(ruta_base, ruta_relativa)