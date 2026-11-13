"""Portable subprocess watchdog around individual math-verify operations."""
from concurrent.futures import ThreadPoolExecutor,TimeoutError
import json,subprocess,sys
from pathlib import Path

class MathEngine:
    def __init__(self,seconds=5):
        self.seconds=seconds;self.parse_cache={};self.pair_cache={};self.timeouts=[]
        self.process=None;self.reader=ThreadPoolExecutor(max_workers=1)
    def request(self,message):
        if self.process is None or self.process.poll() is not None:
            self.process=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__).with_name('math_rpc.py'))],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,text=True,encoding='utf-8')
        self.process.stdin.write(json.dumps(message)+'\n');self.process.stdin.flush()
        task=self.reader.submit(self.process.stdout.readline)
        try:line=task.result(timeout=self.seconds)
        except TimeoutError:
            self.process.kill();self.process.wait();task.result(timeout=10)
            self.timeouts.append(message)
            print('MATH_OPERATION_TIMEOUT '+json.dumps(message),file=sys.stderr,flush=True)
            print(json.dumps({'heartbeat':True}),flush=True)
            return False
        if not line:raise RuntimeError('Math operation worker failed')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        print(json.dumps({'heartbeat':True}),flush=True)
        return bool(result['value'])
    def parsed(self,answer):
        if answer not in self.parse_cache:self.parse_cache[answer]=self.request({'op':'parse','answer':answer})
        return self.parse_cache[answer]
    def equivalent(self,a,b):
        if not self.parsed(a) or not self.parsed(b):return False
        key=(a,b)
        if key not in self.pair_cache:
            value=self.request({'op':'equivalent','a':a,'b':b})
            self.pair_cache[key]=self.pair_cache[(b,a)]=value
        return self.pair_cache[key]
    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close();self.process.wait(timeout=20)
        self.reader.shutdown(wait=True)
