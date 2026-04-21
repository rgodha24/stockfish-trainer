# stockfish-trainer

## layout

`src/rust/` - rust dataloader
`src/data/` - python wrappers around rust dataloader
`src/model/` - model definition, kernels, etc
`src/train/` - train code
`src/ranger22/` - faster rewrite of Ranger21 optimizer
`src/distributed/` - distributed training code (use ray to split dataloading across multiple nodes)

## commands

data download

```bash
uvx --from huggingface_hub hf download --repo-type dataset --local-dir . official-stockfish/master-binpacks nodes5000pv2_UHO.binpack wrongIsRight_nodes5000pv2.binpack

uvx hf download official-stockfish/master-binpacks multinet_pv-2_diff-100_nodes-5000.binpack dfrc_n5000.binpack --repo-type dataset --local-dir .

uvx hf download vondele/from_kaggle_2 T60T70wIsRightFarseerT60T74T75T76.split_0.binpack T60T70wIsRightFarseerT60T74T75T76.split_1.binpack T60T70wIsRightFarseerT60T74T75T76.split_2.binpack T60T70wIsRightFarseerT60T74T75T76.split_3.binpack T60T70wIsRightFarseerT60T74T75T76.split_4.binpack --repo-type dataset --local-dir .
```

training command:

```bash
nix develop -c uv run --no-sync python -m src.train.singlenode \
      /tmp/nnue-data/*.binpack \
      /mnt/external/nnue-data/*.binpack \
      --max-epochs 100 --batch-size 65536 \
      --features Full_Threats+HalfKAv2_hm^ \
      --l1 1040 --l2 31 --l3 32 \
      --lr 4.375e-4 --gamma 0.995 \
      --start-lambda 1.0 --end-lambda 0.75 \
      --random-fen-skipping 10 --early-fen-skipping 12 \
      --w1 3.3553547771220007 --w2 0.7006821612968052 \
      --lambda 1.0 \
      --stacks moe --num-experts 8 --router-features 64 \
      --aux-loss-alpha 0.0005 --z-loss-alpha 0.0 \
      --gumbel-tau 0.2
```
