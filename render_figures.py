"""Redraw frozen paper figures on CPU, using only this repository's inputs."""
from pathlib import Path
import argparse, ast, csv, datetime, hashlib, importlib.util, json, os
import shutil, statistics, subprocess, sys, tempfile

HERE=Path(__file__).resolve().parent
JOBS=[
 ('1',None,'figures/reference/Main_Figure_1__methods.png','Method schematic','reference only'),
 ('2','draw_figure2.py','scripts/renderers/figure2_horizontal_en.png','Explain, discover, intervene, validate','automatic'),
 ('3','make_fig1b.py','scripts/renderers/F1b.png','Historical graph and token-patching summaries','automatic'),
 ('4','make_fig_clt_multi.py','scripts/renderers/F_clt.png','Dictionary reconstruction and capacity','automatic'),
 ('5','make_fig_causal.py','scripts/renderers/F_causal.png','Causal feature search','automatic'),
 ('6','make_fig_repair.py','work_outputs/figure6-completed-controls-final-20261029/F_repair.png','Repair and controls','automatic'),
 ('ED1','make_fig_validation.py','scripts/renderers/F_validation.png','Causal validation','automatic'),
 ('ED2','make_fig_dissociation.py','scripts/renderers/F_dissociation.png','Readout and intervention','automatic'),
 ('ED3','make_fig_reach.py','scripts/renderers/F_reach.png','Cross-layer reach','automatic'),
 ('ED4','make_fig_efficiency.py','scripts/renderers/F_efficiency.png','Historical efficiency','automatic'),
 ('ED5','make_fig_pubmedqa.py','scripts/renderers/F_pubmedqa.png','PubMedQA native repair','automatic'),
 ('ED6','make_fig_mededit.py','scripts/renderers/F_mededit.png','MedEditBench transfer','automatic'),
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    manifest=json.loads((HERE/'data/PLOT_MANIFEST.json').read_text('utf8'))
    for r in manifest['files']:
        p=HERE/r['path']
        if not p.is_file() or sha(p)!=r['sha256']:raise RuntimeError('Missing or changed plotting input: '+r['path'])
        if p.suffix=='.py':ast.parse(p.read_text('utf8'))
    deps={m:importlib.util.find_spec(m) is not None for m in ['numpy','matplotlib','scipy','fitz']}
    if not all(deps.values()):raise RuntimeError('Install requirements.txt: '+str(deps))
    # Independent mean and sample-SD reconstruction from the 576 run-level values.
    rows=list(csv.DictReader((HERE/'data/figure6_runs_numeric.csv').open(encoding='utf8')))
    groups={}
    for r in rows:groups.setdefault((r['model'],r['cohort'],r['arm'],r['metric']),[]).append(r)
    for group in groups.values():
        values=[float(r['value']) for r in group]
        assert len(values)==3
        assert abs(statistics.mean(values)-float(group[0]['mean']))<1e-9
        assert abs(statistics.stdev(values)-float(group[0]['sample_sd']))<1e-9
    plot=json.loads((HERE/'data/figure6_plot_data.json').read_text('utf8'))
    checks=0
    for panel,cohort,metric in [('a','test300','retain'),('b','test300','net'),('c','audit200','acc')]:
        for r in plot['panels'][panel]:
            if r['arm']=='base':continue
            values=[float(x['value']) for x in groups[(r['model_key'],cohort,r['arm'],metric)]]
            assert abs(statistics.mean(values)-r['mean'])<1e-9
            assert abs(statistics.stdev(values)-r['sample_sd'])<1e-9
            checks+=1
    return dict(status='PASS',plot_files=len(manifest['files']),figure6_run_values=len(rows),
                figure6_mean_sd_groups=len(groups),figure6_plotted_series_checked=checks,dependencies=deps)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--only',nargs='+',choices=[j[0] for j in JOBS])
    ap.add_argument('--output',type=Path,default=HERE/'build')
    ap.add_argument('--languages',nargs='+',choices=['en','zh'],default=['en'])
    ap.add_argument('--cjk-font',type=Path,help='Optional locally installed CJK font for Chinese Figure 2; never distributed.')
    args=ap.parse_args()
    if args.list:
        print(json.dumps([dict(id=n,title=t,mode=m) for n,s,p,t,m in JOBS],indent=2));return
    if (not args.only or '2' in args.only) and 'en' not in args.languages:
        ap.error('English Figure 2 is required; use --languages en zh to add Chinese.')
    validation=check()
    if args.check:print(json.dumps(validation,indent=2));return
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=[j for j in JOBS if not args.only or j[0] in args.only]
    results=[]
    with tempfile.TemporaryDirectory(prefix='figure_reproduction_',dir=output) as tmp:
        work=Path(tmp)/'repository';work.mkdir()
        for directory in ['scripts','data','figures']:
            shutil.copytree(HERE/directory,work/directory)
        (work/'scripts/sn-article-template').mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env['SCLT_RENDER_LANGUAGES']=','.join(args.languages)
        if args.cjk_font:env['SCLT_CJK_FONT']=str(args.cjk_font.resolve())
        for number,script,png,title,mode in jobs:
            if script:
                style='publication_style_ed_archived.py' if number.startswith('ED') else 'publication_style.py'
                shutil.copy2(HERE/'scripts/renderers'/style,work/'scripts/renderers/publication_style.py')
                proc=subprocess.run([sys.executable,str(work/'scripts/renderers'/script)],cwd=work,
                    env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=900)
                (output/f'Figure_{number}.stdout.txt').write_text(proc.stdout,encoding='utf8')
                (output/f'Figure_{number}.stderr.txt').write_text(proc.stderr,encoding='utf8')
                if proc.returncode:raise RuntimeError(f'Figure {number} failed; see output logs.\n'+proc.stderr[-2000:])
            source=work/png
            if not source.is_file():raise FileNotFoundError(png)
            files=[]
            stems=[source]
            if number=='2' and 'zh' in args.languages:stems.append(source.with_name('figure2_horizontal_zh.png'))
            for stem in stems:
                for extension in ['png','pdf','svg']:
                    p=stem.with_suffix('.'+extension)
                    if p.is_file():
                        dest=output/f'Figure_{number}__{p.name}';shutil.copy2(p,dest)
                        files.append(dict(path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest)))
            for p in source.parent.glob('*audit*.json'):
                if number in ['2','6'] or p.name==source.stem+'_font_audit.json':
                    shutil.copy2(p,output/f'Figure_{number}__{p.name}')
            results.append(dict(figure=number,title=title,execution='PRESERVED_REFERENCE_ONLY' if script is None else 'RENDERED',files=files))
    report=dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validation=validation,results=results,model_GPU_API_calls=0,
        scope='Frozen-data CPU redraw, not model training or experiment replay.')
    (output/'RENDER_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(status=report['status'],rendered=sum(r['execution']=='RENDERED' for r in results),
        preserved_references=sum(r['execution']=='PRESERVED_REFERENCE_ONLY' for r in results)),indent=2))
if __name__=='__main__':main()
