"""Close task records and stage only task-owned changes in shared registries."""
import argparse
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).parent;ROOT=HERE.parents[1]


def transform(path,text,s):
    eol='\r\n' if text.count('\r\n')>text.count('\n')/2 else '\n'
    link='[Analysis034](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md)'
    cost=f'USD {s["compute_estimate_usd"]:.2f} provisional'
    if path=='research/INDEX.md':
        marker='[Analysis034:'
        if marker in text:
            a=text.index(marker);b=text.index('[Analysis033:',a);text=text[:a]+text[b:]
        pos=text.index('[Analysis033:')
        rates=', '.join(f'{k} {v:g}' for k,v in s['selected_rates'].items())
        block=f'''[Analysis034: Adam from identical GD states](../analyses/034-2026-11-01-eos-adam-continuations/REPORT.md)
completed five-rate calibration and 27 matched-horizon cold-Adam forks.
Selected constant LRs: {rates}; 25 primary, two flagged reconstructions.
Exact raw-gradient anchored/adjacent cosines at every boundary, up to lag 2048.
Terminal 64 median adjacent cosine positive in {s['primary_terminal_median_adjacent_positive']}/25 primary profiles.
Training-only short calibration; optimizer reset, one initialization, no stability
or independent optimizer-benefit claim. Four PDFs and numerical/vector checks pass.
All {s['pods_deleted']} task Pods/storage deleted after verified retrieval; {cost}.
Runs209/210/211; H026 conditional comparison complete. Next Run212 / Analysis035 / H027.

'''
        return text[:pos]+block.replace('\n',eol)+text[pos:]
    start='<!-- Analysis034 records -->';end='<!-- End Analysis034 records -->'
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end)
        text=text[:a]+text[b:].lstrip('\r\n')
    header=next(line for line in text.splitlines() if line.startswith('|') and '---' not in line)
    opening=start+eol+header+eol+'| --- | --- | --- | --- | --- |'+eol
    output=[]
    for line in text.splitlines(keepends=True):
        if path=='runs/INDEX.md' and line.startswith(('| [209]','| [210]','| [211]')):continue
        if path=='analyses/INDEX.md' and line.startswith('| [034]'):continue
        if path=='research/HYPOTHESES.md' and line.startswith('| H026 |'):continue
        if path=='runs/INDEX.md' and line.startswith('Next run:'):
            output.append(opening)
            output.append(f'| [209](209-2026-11-01-eos-adam-calibration/README.md) | H026 cold-Adam LR calibration | Complete joint coverage with Run211; 125 canonical arms here, failed/partial originals preserved | All raw evidence locally hash verified; task cloud deleted | {link}; five rates, training-loss selection only |'+eol)
            output.append(f'| [210](210-2026-11-01-eos-adam-long-lags/README.md) | H026 Adam at matched retained GD states | Complete; 27 exact anchored/adjacent profiles through matched 256/512/2048 | Verified recovery/vectors; all task cloud deleted; {cost} joint | {link}; 25 primary, two flagged |'+eol)
            output.append(f'| [211](211-2026-11-01-eos-parallel-adam-calibration/README.md) | H026 parallel placement of unchanged Adam arms | Complete; 10 arms at two delayed Transformer states, 64 updates each | Reused warm workers; identical scientific inputs, verified retrieval | {link}; 135 canonical calibration arms jointly |'+eol+end+eol+eol)
            line='Next run: `212-YYYY-MM-DD-slug`. Record calibration-only work as such. Preserve'+eol
        if path=='analyses/INDEX.md' and line.startswith('Next analysis:'):
            output.append(opening)
            output.append(f'| [034](034-2026-11-01-eos-adam-continuations/README.md) | Calibrated Adam from identical EOS GD checkpoints | Runs209/210/211; Run205 starts and Analysis033 GD reference | Complete; 27 matched horizons, exact anchored/adjacent raw cosines, 25 primary | Four reviewed PDFs; verified retrieval; task cloud deleted; {cost} |'+eol+end+eol+eol)
            line='Next analysis: `035-YYYY-MM-DD-slug`. Analysis numbering is independent of runs;'+eol
        if path=='research/HYPOTHESES.md' and line.startswith('Next hypothesis:'):
            output.append(opening)
            output.append(f'| H026 | How do cold-Adam continuations change anchored and adjacent raw-gradient alignment at the same GD-trained states after training-loss LR calibration? | Completed conditional descriptive comparison | Runs209/210/211; {link} | Same 27 retained states and GD horizons, 25 primary; five-rate 64-step calibration, one common rate per model; exact full gradients; optimizer reset, seed uncertainty unestimated; no independent method ranking or stability claim |'+eol+end+eol+eol)
            line='Next hypothesis: `H027`. Record the operational meaning,'+eol
        output.append(line)
    return ''.join(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');args=p.parse_args()
    s=json.loads((HERE/'summary.json').read_text())
    for path in ['runs/INDEX.md','analyses/INDEX.md','research/INDEX.md','research/HYPOTHESES.md']:
        actual=ROOT/path;text=actual.read_bytes().decode('utf-8')
        actual.write_bytes(transform(path,text,s).encode('utf-8'))
        if args.stage:
            base=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT).decode('utf-8')
            scoped=transform(path,base,s).encode('utf-8')
            blob=subprocess.check_output(['git','hash-object','-w','--no-filters','--stdin'],input=scoped,cwd=ROOT).decode().strip()
            subprocess.run(['git','update-index','--add','--cacheinfo','100644',blob,path],cwd=ROOT,check=True)
    print('Task registry entries updated'+('; task-only changes staged' if args.stage else ''))


if __name__=='__main__':main()
