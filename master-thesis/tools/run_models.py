#!/usr/bin/env python3
"""Execute original notebook code in an isolated run folder, preserving supplied evidence.

Example: python3 tools/run_models.py --models Linear
Full rerun: python3 tools/run_models.py --models all
Research dependencies are separate from the dependency-free website.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import shutil
import sys

APP=Path(__file__).resolve().parents[1]
BOOKS={'Linear':'01_Linear_Regression_Executed.ipynb','ARIMA':'02_ARIMA_Executed.ipynb',
       'RandomForest':'03_Random_Forest_Executed.ipynb','XGBoost':'04_XGBoost_Executed.ipynb',
       'LightGBM':'05_LightGBM_Executed.ipynb','LSTM':'06_LSTM_Executed.ipynb',
       'TFT':'07_Temporal_Fusion_Transformer_Executed.ipynb'}
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',nargs='+',default=['Linear'],choices=['all',*BOOKS]);p.add_argument('--output',type=Path);p.add_argument('--timeout',type=int,default=14400);a=p.parse_args()
selected=list(BOOKS) if 'all' in a.models else list(dict.fromkeys(a.models))
run=(a.output or APP/'experiment-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
if run.is_relative_to((APP/'research').resolve()) or (APP/'research').resolve().is_relative_to(run):
    raise SystemExit('Choose a fresh output folder outside the preserved research directory.')
if run.exists():raise SystemExit('Output already exists. Choose a new folder to keep runs isolated.')
try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:raise SystemExit('Install the recorded research environment first. See requirements-research.txt and README.md.')
(run/'notebooks').mkdir(parents=True)
shutil.copytree(APP/'research/results/data',run/'data')
queue=[BOOKS[m] for m in selected]
if selected==list(BOOKS):queue.append('08_Cross_Model_Synthesis_Executed.ipynb')
print('Research outputs will be written to',run,flush=True)
for filename in queue:
    print('Executing',filename,flush=True)
    nb=nbformat.read(APP/'research/notebooks'/filename,as_version=4)
    client=NotebookClient(nb,timeout=a.timeout,kernel_name='python3',resources={'metadata':{'path':str(run/'notebooks')}})
    try:
        client.execute()
    finally:
        nbformat.write(nb,run/'notebooks'/filename)
print('Finished. Original supplied research is unchanged. See',run,flush=True)
