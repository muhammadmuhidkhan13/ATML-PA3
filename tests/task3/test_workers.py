import json
import sys
import subprocess
from atml_pa3.task3.cache import EquivalenceWorker
from atml_pa3.task3.safe_math import MathEngine

def test_real_watchdog_worker_protocol_handles_progress(tmp_path):
    worker=EquivalenceWorker(tmp_path/'worker.log')
    try:
        result=worker.compute(dataset='math',answers=['1/2','0.5',None],gold='0.5',metadata=None)
        assert result['correct']==[True,True,False]
        assert result['equivalent'][0][1] and result['operation_timeouts']==[]
    finally:worker.close()

def test_operation_timeout_is_explicit_and_terminates_owned_child(monkeypatch):
    import atml_pa3.task3.safe_math as module
    original=subprocess.Popen
    def sleepy(args,**kwargs):return original([sys.executable,'-c','import time; time.sleep(5)'],**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',sleepy)
    worker=MathEngine(seconds=.05)
    try:
        assert worker.parsed('1') is False
        assert worker.timeouts==[{'op':'parse','answer':'1'}]
        assert worker.process.poll() is not None
    finally:worker.close()
