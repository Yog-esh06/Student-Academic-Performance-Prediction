"""End-to-end orchestration script."""
import subprocess
import sys
from src.risk.risk_classifier import CostSensitiveRiskModel

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
    
    # Run the standard pipeline stages in order
    modules = [
        "src.data.load_data",
        "src.data.preprocess",
        "src.models.train",
        "src.models.evaluate",
        "src.explainability.shap_explain"
    ]
    
    for mod in modules:
        run_module(mod)
        
    # Train the new Risk Classifier
    print("\n--- Initializing Temporal & Cost-Sensitive Risk Engine ---")
    risk_engine = CostSensitiveRiskModel()
    risk_engine.train()
        
    print("\n✅ Pipeline completed successfully!")
    print("\nTo launch the dashboard, run:")
    print("    python -m src.app.app")