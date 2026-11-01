"""Update task registry entries; optionally stage only this task's HEAD changes."""
import argparse
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).parent;ROOT=HERE.parents[1]


def transform(path,text,summary):
    link='[Analysis033](../analyses/033-2026-11-01-eos-long-lags/REPORT.md)'
    cost=f'USD {summary["compute_estimate_usd"]:.2f} provisional'
    if path=='research/INDEX.md':
        start=text.index('[Analysis033:');end=text.index('[Analysis032:',start)
        old=text[start:end];eol='\r\n' if old.count('\r\n')>old.count('\n')/2 else '\n'
        block=f'''[Analysis033: exact long-lag profiles](../analyses/033-2026-11-01-eos-long-lags/REPORT.md)
completed 27 profiles / {summary['cosines']:,} anchored cosines, all through lag 256
and up to {summary['largest_measured_lag']}; 25 primary, two previously unqualified Run062 starts.
Of {summary['primary_initially_reversing']} primary initially reversing profiles, {summary['primary_terminal_half_gap_at_most_threshold']} have terminal local half-gap <=0.1;
{summary['primary_still_above_damping_threshold']} remain above it. Attenuation can rebound; no asymptotic or stability claim.
Runs206/207/208 use exact streaming and qualified checkpoint handoffs, without
compression. Four PDFs and numerical/vector/provenance checks pass.
All {summary['pods_deleted']} task Pods/storage deleted after verified retrieval; {cost}.
Next Run209 / Analysis034; H020 remains active.

'''
        return text[:start]+block.replace('\n',eol)+text[end:]
    output=[]
    for line in text.splitlines(keepends=True):
        eol='\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
        if path=='runs/INDEX.md' and line.startswith('| [206]'):
            rows=[
                f'| [206](206-2026-11-01-eos-long-lags/README.md) | H020: exact long-lag continuation of Analysis032 | Complete; {summary["canonical_profiles_by_run"].get("206",0)} canonical profiles in joint 27; maximum lag {summary["largest_measured_lag"]} | Verified raw evidence; all task cloud deleted; {cost} joint | {link} |',
                f'| [207](207-2026-11-01-eos-parallel-resnet-lags/README.md) | H020: parallel placement of six later ResNet profiles | Complete; baseline 256 before staged extensions; {summary["canonical_profiles_by_run"].get("207",0)} canonical final profiles | Reused 4090 workers; verified retrieval; joint cloud closed | {link} |',
                f'| [208](208-2026-11-01-eos-checkpoint-handoffs/README.md) | H020: qualified checkpoint handoffs | Complete; {summary["canonical_profiles_by_run"].get("208",0)} canonical final profiles with exact prefixes | Original gradient tolerance; verified retrieval; joint cloud closed | {link} |']
            output.extend(row+eol for row in rows);continue
        if path=='runs/INDEX.md' and line.startswith(('| [207]','| [208]')):continue
        if path=='runs/INDEX.md' and line.startswith('Next run:'):
            line='Next run: `209-YYYY-MM-DD-slug`. Record calibration-only work as such. Preserve'+eol
        if path=='analyses/INDEX.md' and line.startswith('| [033]'):
            line=f'| [033](033-2026-11-01-eos-long-lags/README.md) | Exact long-lag damping across nine EOS conditions | Runs205/206/207/208, seed0 | Complete; 27 profiles, 25 primary; common 256, maximum {summary["largest_measured_lag"]}; all task cloud deleted | Four reviewed PDFs, exact CPU/GPU/vector audits; {cost} |'+eol
        if path=='research/HYPOTHESES.md' and line.startswith('| H020 |'):
            cells=line.rstrip('\r\n').split('|')
            cells[3]=' Active; single-initialization 3k calibration and exact long-lag follow-up complete; broad sweeps held '
            if '033-2026-11-01-eos-long-lags' not in cells[4]:cells[4]=cells[4].rstrip()+'; '+link+' '
            legacy=f'Exact anchored profiles through common lag256/adaptive{summary["largest_measured_lag"]} show nonuniform attenuation/rebounds; one pair per lag, no asymptotic/stability claim.'
            note=f'Exact anchored profiles through common lag 256, with extensions up to {summary["largest_measured_lag"]}, show nonuniform attenuation/rebounds; one pair per lag, no asymptotic/stability claim.'
            content=cells[5].strip().replace(legacy,'').replace(note,'').rstrip().rstrip('.')
            cells[5]=' '+content+'. '+note+' '
            line='|'.join(cells)+eol
        output.append(line)
    return ''.join(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',action='store_true');args=parser.parse_args()
    summary=json.loads((HERE/'summary.json').read_text())
    for path in ['runs/INDEX.md','analyses/INDEX.md','research/INDEX.md','research/HYPOTHESES.md']:
        actual=ROOT/path;working=actual.read_bytes().decode('utf-8')
        actual.write_bytes(transform(path,working,summary).encode('utf-8'))
        if args.stage:
            original=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT).decode('utf-8')
            scoped=transform(path,original,summary).encode('utf-8')
            blob=subprocess.check_output(['git','hash-object','-w','--no-filters','--stdin'],input=scoped,cwd=ROOT).decode().strip()
            subprocess.run(['git','update-index','--add','--cacheinfo','100644',blob,path],cwd=ROOT,check=True)
    print('Task records updated'+('; only task changes staged from HEAD' if args.stage else ''))


if __name__=='__main__':main()
