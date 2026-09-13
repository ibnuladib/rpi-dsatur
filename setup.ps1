# PowerShell script to set up the Graph Theory Research project environment
# Run this script from the project root directory.

# 1. Create directory structure
$dirs = @(
    "data_gen",
    "baselines",
    "labels",
    "model",
    "inference",
    "experiments",
    "tests"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Host "Created directory: $d"
    }
}

# 2. Create a Python virtual environment
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Virtual environment '.venv' created."
} else {
    Write-Host "Virtual environment '.venv' already exists."
}

# 3. Install required packages (activate venv for the session)
$activateScript = ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Activated virtual environment."
    pip install --upgrade pip
    pip install torch==2.2.0 torch-geometric==2.5.0 networkx==3.2.1 numpy==1.26.4 scipy==1.13.0 matplotlib==3.8.3
    Write-Host "Installed core scientific packages."
    # Freeze exact versions
    pip freeze > requirements.txt
    Write-Host "Generated requirements.txt."
} else {
    Write-Error "Activation script not found. Ensure Python is installed and you have permission to create a virtual environment."
}

# 4. Create placeholder __init__.py files in each module directory
foreach ($d in $dirs) {
    $initPath = Join-Path $d "__init__.py"
    if (-not (Test-Path $initPath)) {
        Set-Content -Path $initPath -Value "# $d package"
        Write-Host "Created $initPath"
    }
}

Write-Host "Setup complete. To start using the environment, run '.\.venv\Scripts\Activate.ps1' in your PowerShell session."
