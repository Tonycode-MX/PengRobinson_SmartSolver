# PengRobinson_SmartSolver

## Repository Structure

```text
PengRobinson_SmartSolver/
├── backend/                 # Backend logic, calculations, and AI integration
│   ├── agents/              # AI agents and LLM configuration
│   │   ├── __init__.py
│   │   ├── llm_setup.py
│   │   ├── orchestrator.py  
│   │   └── thermo_specialist.py
│   ├── grapher/             # Plotting and visualization modules
│   │   └── processes_plot.py
│   ├── thermo_core/         # Core thermodynamic equations and processes
│   │   ├── eos.py
│   │   └── processes.py
│   ├── utils/               # Helper functions and utilities
│   │   ├── callbacks.py
│   │   └── conversions.py
│   └── main.py              # Main backend entry point
├── environment/             # Conda environment configuration files
├── frontend/                # User interface files
│   └── index.html           # Main entry point for the web UI
├── .env                     # Environment variables (ignored by Git)
├── .gitignore               # Specifies files to exclude from version control
├── requirements.txt         # Python project dependencies
└── README.md                # Project documentation and usage instructions
```


### Virtual Environment Setup

Inside the `environment/` directory, you will find the `requirements.txt` file. This file contains the exported dependencies necessary to replicate the environment and run the code in this repository.

To set up the environment, we recommend using Python's built-in `venv` module. Open your terminal, navigate to the `PengRobinson_SmartSolver/` main directory, and follow these steps:

**1. Create the virtual environment**
Create a new virtual environment named `PR_SS_env` by running:

```bash
python -m venv PR_SS_env
```

**2. Activate the environment**

On Windows:

```bash
PR_SS_env\Scripts\activate
```

On macOS and Linux:

```bash
source PR_SS_env/bin/activate
```

**3. Install the exported dependencies**
Once the environment is activated (you should see (PR_SS_env) at the beginning of your terminal prompt), install the required packages from the exported file:

```bash
pip install -r environment/requirements.txt
```

### Deactivating the Environment
When you are done working on the solver or need to switch environments, you can simply deactivate it by running:

```bash
deactivate
```




## Running the Application

To run the program, you need to start the backend process first and then launch the frontend interface. Follow these steps:

### 1. Navigate to the Root Directory
Open your terminal (ensuring your Conda or virtual environment is already activated) and navigate to the main project folder:

```bash
cd PengRobinson_SmartSolver/
```

### 2. Run the Backend Script
Execute the main Python script located in the `backend/` directory:

```bash
python backend/main.py
```
*Note: Keep the terminal open and wait for the validation messages confirming that the backend is running properly.*

### 3. Launch the Frontend
Once the backend is successfully running, you need to open the user interface. Navigate to the `frontend/` directory in your file explorer and open the `index.html` file in your preferred web browser.

Path to the file:
```bash
PengRobinson_SmartSolver/frontend/index.html
```