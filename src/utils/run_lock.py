"""One live job per output directory.

An output directory is a single run's state: its rolling `train_state.pt`, its
`hybrid_step*.pt`, its `metrics.jsonl`. Two jobs pointed at the same directory both
resume from the same state and then overwrite each other's checkpoints and interleave
each other's metrics lines. Nothing in slurm prevents that -- job names are free-form
and two different job ids can carry identical arguments.

This is not hypothetical here. On 2026-08-19 a user-authorized Codex agent on a 10-minute
patrol resubmitted Stage-B 58 seconds after job 42973 failed (as job 43075, under
account=white while this session was submitting under scavenger). It was right to, and it
did it carefully -- same runner, same output_dir, so it resumed from step 30. But the
Stage-A2 anchor run is now also covered by an sbatch `--dependency=afterany` relay chain,
which fires on exactly the same trigger. A patrol and a relay chain reacting to one
failure produce two concurrent jobs on the same directory, and neither can see the other's
intent. Coordination by agreement does not survive that race; a lock does.

The lock is a file naming the job that holds it, and liveness is decided by asking slurm
about that job rather than by the file's existence -- so a crashed run leaves nothing to
clean up by hand, which matters because the failure mode this guards against IS crashing.
"""

import getpass
import os
import socket
import subprocess
import time

LOCK_NAME = "RUNNING"

# Enumerate RELEASED, not LIVE, so an unrecognised state counts as still owning the
# directory. Listing the live states was the wrong way round: it silently omitted the
# whole preempt-requeue family, and this cluster is PreemptMode=REQUEUE with A1/A2 on the
# preemptible scavenger QOS, so "not running right now but slurm will bring it back" is a
# state our own runs pass through. While the holder sat in one, check() would have called
# it stale and handed the directory away; the held job then resumes into exactly the
# two-jobs-on-one-output_dir collision this exists to prevent. The relay chain would not
# have caused it (afterany fires on an end, not a requeue) -- the patrol would, by seeing
# no progress and concluding the run had died.
#
# slurm's own list, from `squeue --states=<bad>`, is: pending running suspended completed
# cancelled failed timeout node_fail preempted boot_fail deadline out_of_memory completing
# configuring expediting resizing resv_del_hold requeued requeue_fed requeue_hold revoked
# signaling special_exit stage_out stopped. Everything below has finished executing and
# will not come back; everything else -- PREEMPTED, REQUEUE_HOLD, REQUEUE_FED,
# SPECIAL_EXIT, RESV_DEL_HOLD, STOPPED, SIGNALING, and any state a later slurm adds --
# still owns the directory.
#
# COMPLETING and STAGE_OUT are released for the reason documented above: their ranks have
# exited, and treating CG as live would make a relay chain refuse itself at every
# handover, since a link fires the moment its predecessor ends and the predecessor is
# usually still draining then. "" is a job slurm has forgotten.
_RELEASED = {"", "COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT",
             "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED",
             "STAGE_OUT"}


def _job_state(job_id, attempts=3):
    """slurm's state for job_id, "" if it is gone, or None if slurm could not be asked.

    Retried, because the answer decides whether we may write into a directory another job
    may own: a momentarily unreachable slurmctld would otherwise degrade the lock to a
    no-op at the worst possible time. Three tries is enough for a blip and still bounded
    well under the model load that follows.
    """
    last = None
    for attempt in range(attempts):
        try:
            out = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            last = repr(exc)
            if isinstance(exc, FileNotFoundError):
                return None              # no slurm client here; retrying cannot help
        else:
            if out.returncode == 0:
                return out.stdout.strip().split("\n")[0].strip()

            # squeue exits non-zero for an unknown job id, which is the common case here:
            # the previous holder finished long ago and slurm has forgotten it. Any other
            # non-zero exit is slurm failing to answer, which is worth retrying.
            if "Invalid job id" in out.stderr:
                return ""
            last = out.stderr.strip()
        if attempt + 1 < attempts:
            time.sleep(5)
    print(f"WARNING: could not ask slurm about job {job_id} in {attempts} tries: {last}",
          flush=True)
    return None


def read_lock(output_dir):
    """(job_id, raw_text) of the current lock, or (None, None)."""
    path = os.path.join(output_dir, LOCK_NAME)
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None, None
    for line in text.splitlines():
        if line.startswith("jobid="):
            return line.split("=", 1)[1].strip(), text
    return None, text


def check(output_dir, my_job_id=None):
    """Return None if this job may proceed, else a string explaining why not."""
    holder, text = read_lock(output_dir)
    if not holder:
        return None
    my_job_id = my_job_id or os.environ.get("SLURM_JOB_ID", "")
    if holder == str(my_job_id):
        return None                      # our own lock, e.g. after a slurm requeue
    state = _job_state(holder)
    if state is None:
        print(f"WARNING: {LOCK_NAME} names job {holder} but slurm could not be queried; "
              f"proceeding, because refusing on an unanswerable question would strand "
              f"every run on a node without slurm clients.", flush=True)
        return None
    if state in _RELEASED:
        return None                      # stale: the holder has finished
    return (f"job {holder} is {state or 'in an unknown state'} and still owns "
            f"{output_dir}.\n"
            f"{text.strip()}\n"
            f"Two jobs on one output_dir resume from the same train_state.pt and then\n"
            f"overwrite each other's checkpoints. If you are a patrol or a relay link\n"
            f"that concluded this run was dead, it is not -- do nothing. If job {holder}\n"
            f"is genuinely wedged, scancel it first, then this job can take the lock.")


def acquire(output_dir, my_job_id=None):
    """Record this job as the owner. Rank 0 only; call after check() passes."""
    my_job_id = my_job_id or os.environ.get("SLURM_JOB_ID", "")
    path = os.path.join(output_dir, LOCK_NAME)
    tmp = path + ".tmp"
    body = (f"jobid={my_job_id}\n"
            f"user={getpass.getuser()}\n"
            f"host={socket.gethostname()}\n"
            f"nodes={os.environ.get('SLURM_JOB_NODELIST', '?')}\n"
            f"pid={os.getpid()}\n")
    with open(tmp, "w") as fh:
        fh.write(body)
    os.replace(tmp, path)                # atomic: the watcher polls this directory


def release(output_dir, my_job_id=None):
    """Drop the lock if we still hold it. Best effort -- a crash leaves it, and the
    liveness query in check() is what makes that harmless."""
    my_job_id = my_job_id or os.environ.get("SLURM_JOB_ID", "")
    holder, _ = read_lock(output_dir)
    if holder == str(my_job_id):
        try:
            os.remove(os.path.join(output_dir, LOCK_NAME))
        except OSError:
            pass
