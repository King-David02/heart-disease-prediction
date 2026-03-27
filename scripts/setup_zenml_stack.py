import subprocess
import sys
from src.config import settings

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Failed: {cmd}")
        sys.exit(1)

def setup():
    print("Setting up ZenML stack...")

    run("uv run zenml init")
    
    run(
    f"uv run zenml experiment-tracker register mlflow_tracker "
    f"--flavor=mlflow --tracking_uri={settings.mlflow_tracking_uri} "
    f"--tracking_username={settings.dagshub_username} "
    f"--tracking_token={settings.dagshub_token}"
)
    
    run("uv run zenml artifact-store register local_store --flavor=local")
    run("uv run zenml orchestrator register local_orchestrator --flavor=local")
    run("uv run zenml model-registry register mlflow_registry --flavor=mlflow")
    run("uv run zenml stack register ml_stack -o local_orchestrator -a local_store -e mlflow_tracker -r mlflow_registry")
    run("uv run zenml stack set ml_stack")

    print("ZenML stack ready!")
    print("Next: uv run scripts/run_pipeline.py")

if __name__ == "__main__":
    setup()