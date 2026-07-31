import importlib.metadata
import os
import sys
from pathlib import Path

REQUIRED_LOCKED_PKGS = {
    "pandas": "2.2.0",
    "numpy": "1.26.3",
    "openpyxl": "3.1.2",
    "scikit-learn": "1.4.0",
    "scipy": "1.12.0",
    "pyarrow": "15.0.0",
    "chardet": "5.2.0",
    "duckdb": "0.10.0",
    "PyPDF2": "3.0.1",
    "pdfplumber": "0.11.0",
    "pdfminer.six": "20231228",
    "xgboost": "2.0.3",
    "joblib": "1.3.2",
    "PyYAML": "6.0.1",
    "networkx": "3.2.1",
    "openai": "1.40.0",
    "mlflow": "2.10.0",
    "pydantic": "2.6.0",
    "requests": "2.31.0",
    "python-dotenv": "1.0.1",
}

REQUIRED_DIRS = ["logs", "data", "models", "outputs", "reports"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python_version():
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"
    ok = major >= 3 and minor >= 10
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Python >= 3.10 — found {version_str}")
    return ok


def check_locked_packages():
    all_ok = True
    for pkg, expected_version in REQUIRED_LOCKED_PKGS.items():
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            print(f"[FAIL] {pkg} — not installed")
            all_ok = False
            continue
        if installed == expected_version:
            print(f"[PASS] {pkg} == {installed}")
        else:
            print(f"[FAIL] {pkg} — expected {expected_version}, found {installed}")
            all_ok = False
    return all_ok


def check_env_vars():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("[PASS] OPENAI_API_KEY is set")
    else:
        print("[WARN] OPENAI_API_KEY is not set (optional)")
    return True


def check_directories():
    all_ok = True
    for d in REQUIRED_DIRS:
        dirpath = PROJECT_ROOT / d
        if dirpath.is_dir():
            print(f"[PASS] Directory '{d}/' exists")
        else:
            print(f"[FAIL] Directory '{d}/' missing")
            all_ok = False
    return all_ok


def main():
    results = []
    results.append(check_python_version())
    results.append(check_locked_packages())
    results.append(check_env_vars())
    results.append(check_directories())

    critical_pass = all(results) if results else False
    sys.exit(0 if critical_pass else 1)


if __name__ == "__main__":
    main()
