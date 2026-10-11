"""Hidden local monitoring, verified retrieval, render checks and scoped closeout."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
import traceback

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOST = "tcml-spark@spark-2277.local"
REMOTE = "/home/tcml-spark/experiments/constructive-destructive-regimes/high-lr-2026-10-10"
PYTHON = "/home/tcml-spark/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08/venv/bin/python"
SSH = ["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",HOST]
SCP = ["scp","-o","BatchMode=yes","-o","ConnectTimeout=10"]
GIT = ["git","-c","safe.directory="+ROOT.as_posix()]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def remote(*arguments,timeout=60):
    command = shlex.join(str(a) for a in arguments)
    return subprocess.run(SSH+[command],capture_output=True,text=True,timeout=timeout,check=True).stdout


def local_path(relative):
    posix = PurePosixPath(relative)
    assert not posix.is_absolute() and ".." not in posix.parts
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT.resolve())
    return path


def copy(relative, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(SCP+[f"{HOST}:{REMOTE}/{relative}",str(destination)],check=True,timeout=3600)


def retrieve_run(record):
    run = ROOT / "runs" / record["run"]
    receipt_path = run / "prelaunch/retrieval-receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text())["accepted"]:
        return
    relative = f"runs/{record['run']}/artifacts/attempt-01"
    parent = run / "artifacts"
    parent.mkdir(exist_ok=True)
    subprocess.run(SCP+["-r",f"{HOST}:{REMOTE}/{relative}",str(parent)],check=True,timeout=3600)
    files, total = 0, 0
    inventories = []
    for folder in sorted((parent / "attempt-01").iterdir()):
        if not folder.is_dir():
            continue
        inventory_path = folder / "transfer_inventory.json"
        if not inventory_path.exists():
            raise ValueError(f"Missing terminal byte inventory: {folder}")
        relative_inventory = inventory_path.relative_to(ROOT).as_posix()
        expected = remote("sha256sum",REMOTE+"/"+relative_inventory).split()[0]
        assert sha(inventory_path) == expected
        inventory = json.loads(inventory_path.read_text())
        for item in inventory["files"]:
            path = (folder / item["path"]).resolve()
            assert path.is_relative_to(folder.resolve()) and not path.is_symlink()
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],path
            files += 1
            total += item["bytes"]
        inventories.append(dict(scope=folder.name,sha256=expected))
    save(receipt_path,dict(accepted=True,files_verified=files,bytes_verified=total,
        inventories=inventories,verified_at=now(),remote_state_retained=True,
        scientific_success_claimed=False))


def final_artifacts():
    relative = f"analyses/{HERE.name}/result_inventory.json"
    inventory_path = HERE / "result_inventory.json"
    copy(relative,inventory_path)
    assert sha(inventory_path) == remote("sha256sum",REMOTE+"/"+relative).split()[0]
    inventory = json.loads(inventory_path.read_text())
    managed = [inventory_path.relative_to(ROOT).as_posix()]
    for item in inventory["files"]:
        path = local_path(item["path"])
        copy(item["path"],path)
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
        managed.append(item["path"])
    data_path = HERE / "comparison_data.json"
    data = json.loads(data_path.read_text())
    verification = json.loads((HERE / "figure_verification.json").read_text())
    assert verification["passed"] and verification["data_sha256"] == sha(data_path)
    assert data["generator_sha256"] == sha(HERE / "03_plot.py")
    assert data["source_inventory_sha256"] == sha(HERE / "source_inventory.json")
    for figure in data["figures"]:
        pdf = HERE / "figures" / figure["path"]
        assert sha(pdf) == figure["sha256"]
        with fitz.open(pdf) as document:
            assert len(document) == 1
            page = document[0]
            assert "Unsmoothed training loss" in page.get_text()
            for block in page.get_text("blocks"):
                assert page.rect.contains(fitz.Rect(block[:4]))
            (HERE / "scratch").mkdir(exist_ok=True)
            page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False).save(str(HERE / "scratch" / (pdf.stem+"_review.png")))
    proof = HERE / "local_artifact_verification.json"
    save(proof,dict(passed=True,checked_at=now(),inventory_sha256=sha(inventory_path),
                   files_verified=len(inventory["files"]),plots_match_remote=True,
                   text_and_bounds_passed=True,final_visual_review="pending_inspection"))
    managed.append(proof.relative_to(ROOT).as_posix())
    return managed


def update_registries(managed, cohort):
    data = json.loads((HERE / "comparison_data.json").read_text())
    run_states = {item["run_id"]:data["runs"][item["run_id"]] for item in cohort["runs"]}
    for relative in ("runs/INDEX.md","analyses/INDEX.md","research/HYPOTHESES.md","research/INDEX.md"):
        path = ROOT / relative
        # A user may work in these shared registries while the detached jobs run.
        if subprocess.run(GIT+["diff","--quiet","HEAD","--",relative],cwd=ROOT).returncode:
            continue
        original = path.read_bytes()
        revised = original
        if relative == "research/INDEX.md":
            start,end = b"<!-- analysis015:start -->",b"<!-- analysis015:end -->"
            if start not in original or end not in original:
                continue
            states = "; ".join(f"Run{rid}: {r['status']}, {r['accepted_updates']}/712 updates" for rid,r in run_states.items())
            note = f"[Analysis015: larger Spark learning rates](../analyses/{HERE.name}/RESULTS.md)\n{states}. Two figures are generated and byte verified locally; final visual\nreview is pending. Matched Spark LR0.01 reference; one seed1234 and one pass.\nNo numerical failure is masked or retried at a lower LR. No cloud rental.\n"
            left,right = original.index(start)+len(start),original.index(end)
            revised = original[:left]+b"\n"+note.encode()+original[right:]
        else:
            lines = []
            for line in original.splitlines(keepends=True):
                ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
                text = line.rstrip(b"\r\n").decode("utf-8")
                columns = text.split("|")
                if relative == "runs/INDEX.md":
                    for rid,r in run_states.items():
                        if text.startswith(f"| [{rid}]("):
                            columns[3] = f" {r['status']}; {r['accepted_updates']}/712 updates; outputs retrieved/hashed "
                elif relative == "analyses/INDEX.md" and text.startswith("| [015]("):
                    columns[4] = " Terminal results retrieved/hashed; automatic figures ready; visual review pending "
                elif relative == "research/HYPOTHESES.md" and text.startswith("| H014 |"):
                    columns[3] = " Completed exploratory comparison or explicit failed prefixes; one seed; visual review pending "
                lines.append("|".join(columns).encode()+ending)
            revised = b"".join(lines)
        if revised != original:
            path.write_bytes(revised)
            managed.append(relative)


def commit_results(managed, cohort):
    # Only task-owned generated files; never sweep the working tree or active logs.
    for item in cohort["runs"]:
        run = ROOT / "runs" / item["run"]
        managed.append((run / "prelaunch/retrieval-receipt.json").relative_to(ROOT).as_posix())
        for path in (run / "artifacts/attempt-01").glob("*/*.json"):
            managed.append(path.relative_to(ROOT).as_posix())
    managed = sorted(set(managed))
    unrelated_staged = subprocess.check_output(GIT+["diff","--cached","--name-only"],cwd=ROOT,text=True).splitlines()
    if any(p not in managed for p in unrelated_staged):
        return dict(committed=False,reason="Unrelated staged changes preserved; task artifacts are ready.")
    update_registries(managed,cohort)
    managed = sorted(set(managed))
    subprocess.run(GIT+["add","--"]+managed,cwd=ROOT,check=True)
    subprocess.run(GIT+["-c","core.whitespace=cr-at-eol","diff","--cached","--check"],cwd=ROOT,check=True)
    if subprocess.run(GIT+["diff","--cached","--quiet"],cwd=ROOT).returncode == 0:
        return dict(committed=False,reason="No result changes to commit.")
    subprocess.run(GIT+["commit","--only","-m","Record DGX Spark high-LR results and automatic figures","--"]+managed,cwd=ROOT,check=True)
    return dict(committed=True,commit=subprocess.check_output(GIT+["rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip())


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    deadline = time.monotonic()+20*3600
    state = dict(status="monitoring",started_at=now(),retrieved_runs=[])
    save(HERE / "local-controller.json",state)
    completed = set()
    while time.monotonic()<deadline:
        if (HERE / "STOP_MONITORING").exists():
            state.update(status="stopped_by_user_marker",finished_at=now())
            save(HERE / "local-controller.json",state)
            return
        try:
            snapshot = json.loads(remote(PYTHON,REMOTE+f"/analyses/{HERE.name}/02_status.py"))
            save(HERE / "latest-status.json",snapshot)
            with (HERE / "monitoring.jsonl").open("a",encoding="utf-8") as file:
                file.write(json.dumps(snapshot,allow_nan=False)+"\n")
            print(json.dumps(dict(timestamp=snapshot["timestamp"],current_run=snapshot["queue"].get("current_run"),
                                 remaining_hours=snapshot.get("estimated_cohort_remaining_hours"),gpu=snapshot["gpu"])),flush=True)
            for record in cohort["runs"]:
                condition = snapshot["queue"].get("runs",{}).get(record["run_id"],{}).get("status")
                if condition in ("completed","failed") and record["run_id"] not in completed:
                    retrieve_run(record)
                    completed.add(record["run_id"])
                    state["retrieved_runs"] = sorted(completed)
                    save(HERE / "local-controller.json",state)
            if len(completed) == 2 and snapshot["queue"].get("plot_status") in ("completed","failed"):
                if snapshot["queue"]["plot_status"] == "failed":
                    raise RuntimeError("Remote figure generation failed; scientific evidence preserved. See queue_state/remote logs.")
                managed = final_artifacts()
                state.update(status="results_ready_pending_visual_review",finished_at=now(),
                             scientific_status=snapshot["queue"]["status"],commit=commit_results(managed,cohort))
                save(HERE / "local-controller.json",state)
                return
        except Exception:
            state.update(last_error=traceback.format_exc(),last_error_at=now())
            save(HERE / "local-controller.json",state)
            print(state["last_error"],flush=True)
        time.sleep(60)
    state.update(status="retrieval_deadline_reached_remote_evidence_preserved",finished_at=now())
    save(HERE / "local-controller.json",state)


if __name__ == "__main__":
    main()
