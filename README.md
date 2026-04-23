# stockfish-trainer

this is a faster rewrite of the [stockfish nnue pytorch trainer](https://github.com/official-stockfish/nnue-pytorch)

improvements:

- tile lang based custom cuda kernel for spare forward and backwards passes. ~4x speedup
- vendored & rewritten ranger21 optimizer. this enabled kernel fusion via torch.compile for a ~5x speedup
- rust based rewritten dataloader. it scales across cpu cores much better (its not constrained by locking), and also supports much faster & better DDP (e.g. multigpu and even multi node) support. speedup is hard to measure due to how much better it scales across large amounts of cpu, but it is ~10x in some cases. it also provides a better data distribution across a large amount of .binpack'd positions than the previous cpp implementation.
- stacks = moe, stacks = none, stacks = layer. e.g. support for the smart router/MoE based experts that we are researching
- ray based distributed data decode/encode support. this basically just splits up the cpu work across multiple servers/vms and sends the data across the network efficiently with good batching, mainly to enable training on GT PACE ICE, which has slow cpus on ~all of its GPU nodes.

## layout

`src/rust/` - rust dataloader
`src/data/` - python wrappers around rust dataloader
`src/model/` - model definition, kernels, etc
`src/train/` - train code
`src/ranger22/` - faster rewrite of Ranger21 optimizer
`src/distributed/` - distributed training code (use ray to split dataloading across multiple nodes)
`stockfish/` - stockfish fork that supports moe stacks, layer stacks, and no stacks with compile time switching
`BUILD.md` - build instructions & cutechess cli examples

## commands

data download

```bash
uvx --from huggingface_hub hf download --repo-type dataset --local-dir . official-stockfish/master-binpacks nodes5000pv2_UHO.binpack wrongIsRight_nodes5000pv2.binpack

uvx hf download official-stockfish/master-binpacks multinet_pv-2_diff-100_nodes-5000.binpack dfrc_n5000.binpack --repo-type dataset --local-dir .

uvx hf download vondele/from_kaggle_2 T60T70wIsRightFarseerT60T74T75T76.split_0.binpack T60T70wIsRightFarseerT60T74T75T76.split_1.binpack T60T70wIsRightFarseerT60T74T75T76.split_2.binpack T60T70wIsRightFarseerT60T74T75T76.split_3.binpack T60T70wIsRightFarseerT60T74T75T76.split_4.binpack --repo-type dataset --local-dir .
```

example training command:

```bash
nix develop -c uv run --no-sync python -m src.train.singlenode \
      /tmp/nnue-data/*.binpack \
      /mnt/external/nnue-data/*.binpack \
      --max-epochs 100 --batch-size 65536 \
      --features Full_Threats+HalfKAv2_hm^ \
      --l1 1040 --l2 31 --l3 32 \
      --lr 4.375e-4 --gamma 0.995 \
      --start-lambda 1.0 --end-lambda 0.75 \
      --early-fen-skipping 12 \
      --w1 3.3553547771220007 --w2 0.7006821612968052 \
      --lambda 1.0 \
      --stacks moe --num-experts 8 --router-features 64 \
       --aux-loss-alpha 0.0005 --z-loss-alpha 0.0 \
       --gumbel-tau 0.2
```

single-node multi-GPU DDP (`--batch-size` is the global batch size and is split evenly across ranks):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m src.train.singlenode \
      /tmp/nnue-data/*.binpack \
      /mnt/external/nnue-data/*.binpack \
      --max-epochs 100 --batch-size 65536 \
      --features Full_Threats+HalfKAv2_hm^
```

On 4 GPUs, `--batch-size 65536` means `16384` samples per GPU.

`src.train.multinode` is still the Ray-fed single-trainer-process path; for 1 VM / 4xH100 training use `src.train.singlenode` under `torchrun`.
