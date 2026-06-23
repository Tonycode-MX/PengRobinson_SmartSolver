from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from agents.llm_setup import llm
from agents.thermo_tools import thermo_specialist_tool

# =====================================================================
# ORCHESTRATOR AGENT CONFIGURATION (MAIN BRAIN)
# =====================================================================
orchestrator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Orchestrator Agent for an advanced Physics Simulator.
    Your objective is to read the user's problem, extract key information, and delegate the task to the correct Specialist.
    
    EXTRACTION RULES:
    1. Identify the chemical fluids and translate them to standard English (e.g., "agua" -> "Water").
    2. Identify molar fractions.
    3. Extract all exact state variables and units provided by the user.
    4. Identify the requested process type.
    
    ROUTING RULES:
    - For real gases, EOS, macroscopic processes -> Call 'Thermodynamics_Specialist'.
    - For molecular interactions, Lennard-Jones, particles -> Call 'Statistical_Specialist'.
    
    Format the extracted data clearly and pass it to the appropriate tool. Do not solve the physics problem yourself."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

orchestrator_tools = [thermo_specialist_tool] # Aquí agregarás el de Monte Carlo después

orchestrator_agent = create_tool_calling_agent(llm, orchestrator_tools, orchestrator_prompt)
orchestrator_executor = AgentExecutor(agent=orchestrator_agent, tools=orchestrator_tools, verbose=True)