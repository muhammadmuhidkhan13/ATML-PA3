"""Persistent main-thread math-verify worker; parent supplies the wall timeout."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from atml_pa3.task3 import equivalence
from atml_pa3.task3.safe_math import MathEngine
if __name__=='__main__':
    engine=MathEngine();equivalence.MATH_ENGINE=engine
    try:
        for line in sys.stdin:
            try:
                before=len(engine.timeouts);value=equivalence.semantics(**json.loads(line))
                value['operation_timeouts']=engine.timeouts[before:]
                result={'result':value}
            except Exception as exc:result={'error':repr(exc)}
            print(json.dumps(result),flush=True)
    finally:engine.close()
