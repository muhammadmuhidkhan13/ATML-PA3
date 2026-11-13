"""Private math operation process; stdout is a JSON-lines protocol."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from atml_pa3.task3.equivalence import parse_math,math_equivalent
if __name__=='__main__':
    for line in sys.stdin:
        try:
            item=json.loads(line)
            value=bool(parse_math(item['answer'])) if item['op']=='parse' else math_equivalent(item['a'],item['b'])
            result={'value':value}
        except Exception as exc:result={'error':repr(exc)}
        print(json.dumps(result),flush=True)
