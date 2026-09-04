from __future__ import annotations
import importlib, pathlib, py_compile
from .hardware.profile import detect_hardware
from .hardware.os_optimizer import choose_os

CORE_DEPS=("torch","psutil")
OPTIONAL_DEPS={"pandas":"analytics","pyarrow":"parquet storage","duckdb":"analytical SQL","playwright":"browser collection","trafilatura":"browser text extraction"}

def run_doctor(root="."):
    root=pathlib.Path(root); issues=[]; checked=0
    for p in root.rglob("*.py"):
        if ".venv" in p.parts: continue
        checked+=1
        try: py_compile.compile(str(p),doraise=True)
        except Exception as e: issues.append({"severity":"CRITICAL","file":str(p),"error":str(e)})
    for mod in CORE_DEPS:
        try: importlib.import_module(mod)
        except Exception as e: issues.append({"severity":"CRITICAL","dependency":mod,"error":str(e)})
    for mod,feature in OPTIONAL_DEPS.items():
        try: importlib.import_module(mod)
        except Exception as e: issues.append({"severity":"WARNING","dependency":mod,"feature":feature,"error":str(e)})
    profile=detect_hardware(); critical=any(i["severity"]=="CRITICAL" for i in issues)
    return {"status":"unhealthy" if critical else "healthy_with_warnings" if issues else "healthy","python_files_checked":checked,
            "issues":issues,"hardware":profile.to_dict(),"os_optimizer":choose_os(profile)}
