# weather-30537 Training Run — legacy v10.1 record

> Historical evidence only. This run used the retired compact 2D U-Net deterministic branch and must not be treated as validation or a launch recipe for v10.2 SDIR.

## Deployment

- Host: `weather-30537` (`3v4t8phjsj6jp-0`)
- tmux session/window: `wzq:phyrd-537`
- monitor window: `wzq:phyrd-monitor`
- project: `/test1/wzq/PhyRD`
- data: `/test1/wzq/data/sevir`
- conda environment: `PhyRD`
- package sources: Tsinghua conda and PyPI mirrors
- runtime: Python 3.11, PyTorch 2.4.1+cu121, CUDA 12.1

No file, directory or environment deletion command was used.

## Experiment

The active run is one seed (`42`) trained with eight synchronous PyTorch DDP ranks. All ranks begin from the same parameters, use disjoint `DistributedSampler` shards, participate in NCCL gradient averaging, and advance one shared optimizer step. Only rank 0 writes logs and checkpoints.

| Stage | GPUs | Per-rank batch | Global batch | Length |
|---|---:|---:|---:|---:|
| deterministic | 8 | 64 | 512 | 50 epochs |
| residual + physics | 8 | 32 | 256 | 100 epochs |

Immediately before launch, all eight NVIDIA H800 GPUs had zero allocated memory, zero utilization and temperatures of 25–30°C. The automatic launcher threshold is 75°C and would exclude a hot or occupied GPU.

The active experiment root is:

```text
/test1/wzq/PhyRD/artifacts/experiments/phryd_v10_1_ddp8_seed42_weather30537
```

The launcher stores frozen configs, initial GPU state and code hashes at the experiment root. Each stage has a live console log, rank-0 checkpoints, a machine-readable train log and a final run summary. After 50 deterministic epochs, `checkpoint_final.pt` is loaded automatically by the 100-epoch residual stage.

## Monitoring

```bash
tmux attach -t wzq
nvidia-smi
tail -f /test1/wzq/PhyRD/artifacts/experiments/phryd_v10_1_ddp8_seed42_weather30537/deterministic_console.log
```

DDP probes completed successfully for both stages. The deterministic probe measured 6.80 GiB peak allocated memory per rank and about 292 global samples/s; the residual probe measured 3.92 GiB and about 92 global samples/s. The production deterministic stage completed all 50 epochs / 1,150 shared optimizer steps at 668.68 global samples/s and wrote `checkpoint_final.pt`. The launcher then loaded it into the residual 8-rank job automatically. The final handoff audit observed residual step 1,260 with rank-0 checkpoints at steps 470 and 940, 7.5–8.1 GiB currently reserved per GPU, temperatures of 29–36°C, and no NCCL error. Earlier deterministic continuous sampling observed roughly 10.8–11.0 GiB reserved per GPU and compute utilization up to 84%.

The earlier `phyrd_v10_1_8seed_128_weather30537` run used independent seeds and was stopped after the user clarified that synchronous DDP was required. Its artifacts were preserved and are not part of the active result.
