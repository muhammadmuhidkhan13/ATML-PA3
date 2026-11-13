"""Regenerate numeric tables and labeled figures from saved offline results."""
from pathlib import Path
import csv
import numpy as np
from .io import atomic_json,read_json,identity
from .experiment import csv_rows

def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def export(output,*,split='evaluation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.dpi':120,'axes.grid':True,'grid.alpha':.2})
    output=Path(output);figures=output/'figures'/split
    if figures.exists():raise FileExistsError('Use a fresh figures directory; existing exports preserved')
    figures.mkdir(parents=True);generated=[]
    for dataset in ('math','mathvista'):
        directory=output/dataset/split;rows=read_csv(directory/'aggregate.csv')
        names=list(dict.fromkeys(r['method'] for r in rows))
        def save(fig,name):
            fig.tight_layout();filename=f'{dataset}_{name}.png';fig.savefig(figures/filename);plt.close(fig);generated.append(filename)
        for xkey,ykey,title in [('budget','accuracy','accuracy_budget'),('budget','oracle','oracle_budget'),
                               ('oracle','accuracy','achieved_vs_oracle'),('token_cost','accuracy','token_cost_accuracy'),
                               ('budget','distinct_groups','distinct_groups'),('budget','selection_failure','selection_failure')]:
            fig,ax=plt.subplots(figsize=(8,4))
            chosen=['SC'] if ykey in ('oracle','distinct_groups') else names
            for name in chosen:
                subset=[r for r in rows if r['method']==name]
                ax.plot([float(r[xkey]) for r in subset],[float(r[ykey]) for r in subset],marker='o',markersize=3,label=name)
            labels={'budget':'Candidate budget N','accuracy':'Accuracy (fraction)','oracle':'Oracle coverage (fraction)',
                    'token_cost':'Retained generated tokens (offline cost proxy)','distinct_groups':'Mean distinct answer groups',
                    'selection_failure':'Selection-failure rate (fraction)'}
            ax.set(xlabel=labels[xkey],ylabel=labels[ykey],title=f'{dataset}: {split}')
            if xkey=='budget':ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],['1','2','4','8','16','32'])
            ax.legend(fontsize=7,ncol=2);save(fig,title)
        quartiles=read_csv(directory/'quartiles.csv');qrows=[]
        for signal in dict.fromkeys(r['signal'] for r in quartiles):
            selected=[r for r in quartiles if r['signal']==signal and r['gap']!='']
            qrows.append({'signal':signal,'defined_questions':len(selected),'undefined_questions':len(quartiles)//len(set(r['signal'] for r in quartiles))-len(selected),
                          'top_accuracy':np.mean([float(r['top_accuracy']) for r in selected]) if selected else None,
                          'bottom_accuracy':np.mean([float(r['bottom_accuracy']) for r in selected]) if selected else None,
                          'gap':np.mean([float(r['gap']) for r in selected]) if selected else None})
        csv_rows(figures/f'{dataset}_quartile_summary.csv',qrows)
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(qrows))
        ax.bar(x-.18,[r['top_accuracy'] if r['top_accuracy'] is not None else np.nan for r in qrows],.36,label='Top quartile')
        ax.bar(x+.18,[r['bottom_accuracy'] if r['bottom_accuracy'] is not None else np.nan for r in qrows],.36,label='Bottom quartile')
        ax.set_xticks(x,[r['signal'] for r in qrows],rotation=20);ax.set_ylabel('Mean within-question correctness (fraction)');ax.legend();save(fig,'quartiles')
        fig,ax=plt.subplots(figsize=(8,4));final=[r for r in rows if int(r['budget'])==32]
        ax.bar([r['method'] for r in final],[float(r['accuracy']) for r in final]);ax.tick_params(axis='x',rotation=45)
        ax.set(ylabel='Accuracy (fraction)',title=f'{dataset}: N=32, includes shortest-rollout control');save(fig,'method_length_control')
        if dataset=='mathvista':
            correlations=read_csv(directory/'confidence_vision.csv');values=[float(r['pearson']) for r in correlations if r['pearson']!='']
            fig,ax=plt.subplots(figsize=(6,3));ax.hist(values,bins=np.linspace(-1,1,31));ax.set(xlabel='Question-mean within-rollout Pearson correlation',ylabel='Questions');save(fig,'confidence_vision')
            fig,ax=plt.subplots(figsize=(7,4))
            for name in ('HighestVisConf','VisConf_RWC','HighestTokenFusion','TokenFusion_RWC'):
                subset=[r for r in rows if r['method']==name]
                ax.plot([int(r['budget']) for r in subset],[float(r['accuracy']) for r in subset],marker='o',label=name)
            ax.set(xlabel='Candidate budget N',ylabel='Accuracy (fraction)');ax.legend();save(fig,'fusion_ablation')
        associations=read_csv(directory/'signal_associations.csv');summaries=[]
        for pair in dict.fromkeys(r['signals'] for r in associations):
            valid=[float(r['spearman']) for r in associations if r['signals']==pair and r['spearman']!='']
            summaries.append({'signals':pair,'mean_within_question_spearman':float(np.mean(valid)) if valid else None,
                              'defined_questions':len(valid)})
        csv_rows(figures/f'{dataset}_association_summary.csv',summaries)
    atomic_json(figures/'manifest.json',{'split':split,'figures':generated,'cost_units':'retained generated tokens; offline proxy',
                                        'measured_gpu_cost':None})
    return generated
