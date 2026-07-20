# Git Workflow

GitHub is the source of truth for code, configuration, tests, scripts, and
documentation. The local workstation and each server keep a checkout of the
same `main` branch.

## Required flow

1. Edit and test on the local workstation.
2. Commit the local change with a meaningful message.
3. Push the commit to `origin/main`.
4. On an idle server, run `git pull --ff-only origin main` before launching a
   new job.
5. After the first checkout, initialize pinned upstream sources with `git
   submodule update --init --recursive`.
6. Record the commit SHA in the experiment registry and run summary.

Only the local workstation performs `git push`. Servers are pull-only and must
never be used to publish commits to GitHub.

Do not pull or switch branches in a checkout while a training or evaluation
process is using that checkout. Finish or stop the job first, then update the
code and launch a new process.

## Server-only state

The following are intentionally excluded from Git and must remain on the
servers:

- SEVIR data and other files under `data/`, including `*.h5` and `*.hdf5`;
- model checkpoints under `ckp/`, `checkpoints/`, or any `*.pt`, `*.pth`, and
  `*.ckpt` file;
- run outputs under `artifacts/`, `wandb/`, and local caches.

Never use `git clean -fd`, `git reset --hard`, or a broad recursive delete in a
server project directory. The repository controls code only; it does not own
the server data or experiment state.

## Remote authentication

The canonical remote is:

```text
git@github.com:wzq-come-on/PhyRD.git
```

The workstation may use its existing GitHub SSH credentials. Servers need a
read-only deploy key or an approved HTTPS credential helper; private tokens
must never be stored in this repository or in shell scripts.
