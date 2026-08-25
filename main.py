"""End-to-end orchestration script."""
import subprocess
import sys

def run_module(module_name: str):
    """Runs a Python module as a subprocess."""
    print(f"\n{'='*50}")
    print(f"🚀 Running: {module_name}")
    print(f"{'='*50}")
    
    result = subprocess.run([sys.executable, "-m", module_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Pipeline failed at {module_name}. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Student Performance Prediction Pipeline...")
    
    # Run the pipeline stages in order
    modules = [
        "src.data.load_data",
        "src.data.preprocess",
        "src.models.train",
        "src.models.evaluate",
        "src.explainability.shap_explain"
    ]
    
    for mod in modules:
        run_module(mod)
        
    print("\n✅ Pipeline completed successfully!")
    print("\nTo launch the dashboard, run:")
    print("    python -m src.app.app")