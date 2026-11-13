"""Generate figures only from complete real saved-call results; absent stays pending."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atml_pa3.task3.io import read_json,sha256,atomic_json
from .live_run import validate_completed_run as validate_live_results

def export(assets,run,output):
    run,output=Path(run),Path(output)
    if output.exists():raise FileExistsError('Use new figure export directory')
    if not (run/'COMPLETE.json').exists():
        atomic_json(output/'status.json',{'live':'PENDING; no complete real live results','figures':[]})
        return {'live':'pending','figures':0}
    qs,records,rows=validate_live_results(assets,run)
    output.mkdir(parents=True)
    # Rebuild aggregate numeric values from verified records, never placeholder values.
    from .live_analysis import export_live_tables,select_live_examples,csv_rows
    csv_rows(output/'strategy_results.csv',rows)
    export_live_tables(output,qs,rows,records)
    select_live_examples(output,qs,rows,records)
    table=read_json(output/'aggregate.json');names=[r['strategy'] for r in table['strategies']];figures=[]
    for metric,label in [('correct','Answer accuracy (fraction)'),('output_tokens','Mean actual output tokens (including stop tokens)'),
        ('input_tokens','Mean actual input tokens'),('inference_seconds','Mean measured inference seconds'),
        ('extraction_failed','Selected extraction-failure fraction'),('selected_truncated','Selected response truncated (fraction)')]:
        plt.figure(figsize=(7,4));plt.bar(names,[r[metric] for r in table['strategies']]);plt.ylabel(label)
        plt.title('Separate controlled FP16 live comparison (50 MATH questions)');plt.tight_layout()
        name=f'live_{metric}.png';plt.savefig(output/name,dpi=160);plt.close();figures.append(name)
    rr=next(r for r in table['strategies'] if r['strategy']=='refinement');keys=['wrong_to_correct','correct_to_wrong','correct_to_correct','wrong_to_wrong']
    plt.figure(figsize=(8,4));plt.bar(keys,[rr[k+'_count'] for k in keys]);plt.xticks(rotation=15);plt.ylabel('Questions');plt.title('Shared initial to revised correctness');plt.tight_layout()
    plt.savefig(output/'revision_transitions.png',dpi=160);plt.close();figures.append('revision_transitions.png')
    atomic_json(output/'status.json',{'live':'complete real results','figures':figures,'source_completion_sha256':sha256(run/'COMPLETE.json')})
    return {'live':'complete','figures':len(figures)}
