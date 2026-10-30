"""Infrastructure partition must not duplicate runs or split matched comparisons."""
import importlib.util
import ast
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]/'analyses/030-2026-10-30-adam-lr-decisions'
spec=importlib.util.spec_from_file_location('parallel_worker',HERE/'09_parallel_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def test_partition_preserves_all_scientific_rows_and_seed_blocks():
    cohort=json.loads((HERE/'cohort.json').read_text())
    assignment=json.loads((HERE/'parallel_assignment.json').read_text())
    queues=[worker.assigned_queue(cohort,assignment,i) for i in range(3)]
    assert [len(q) for q in queues]==[6,6,6]
    assert sorted(r['run'] for q in queues for r in q)==sorted(r['run'] for r in cohort)
    for q in queues:
        assert len({r['seed'] for r in q})==1
        assert {r['size'] for r in q}=={'14M','70M'}
        assert {r['multiplier'] for r in q}=={.5,1.,2.}
    bad=dict(assignment);bad[cohort[0]['run']]=1
    with pytest.raises(AssertionError):worker.assigned_queue(cohort,bad,0)
    bad.pop(cohort[0]['run'])
    with pytest.raises(AssertionError):worker.assigned_queue(cohort,bad,0)


def test_launcher_remote_python_blocks_parse_before_deployment():
    tree=ast.parse((HERE/'10_parallelize.py').read_text())
    programs=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and "python - <<'PY'\n" in node.value:
            program=node.value.split("python - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
            ast.parse(program);programs.append(program)
    assert len(programs)==2
