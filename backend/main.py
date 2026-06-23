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