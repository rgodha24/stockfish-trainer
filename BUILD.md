# Stockfish + nets

Builds live under `stockfish/`. Training checkpoints are exported with `src/scripts/serialize.py` and must match the binary’s `STACKS` mode.

## `STACKS` (make flag)

| `STACKS` | Role |
| -------- | ---- |
| `layer`  | 8 material buckets, default big-net layout |
| `none`   | 1 shared FC bucket |
| `moe`    | 8 experts + learned router |
| (unset)  | same as `layer` |

## Build

```bash
make -C stockfish/src clean
make -C stockfish/src -j"$(nproc)" ARCH=native STACKS=moe all   # or layer | none
cp -f stockfish/src/stockfish stockfish/stockfish-moe-tournament
```

**Default big-net names** (`stockfish/src/evaluate.h`): `layer` → `nn-f68ec79f0fe3.nnue`, `none` → `nn-nonstacks-big.nnue`, `moe` → `nn-moe-big.nnue`. For a different net: `setoption name EvalFile value /abs/path/file.nnue`.

**Fetch/check defaults:** `make -C stockfish/src net STACKS=layer` (and similarly for `none` / `moe`).

## Serialize checkpoint → `.nnue`

`--stack-mode` must match how you built Stockfish.

```bash
uv run python -m src.scripts.serialize /path/to/last.pt /path/to/out.nnue --stack-mode moe
```

WandB checkpoints: `wandb/latest-run/files/checkpoints/final.pt` (or `last.pt`).

## Quick smoke test

```bash
printf "uci\nisready\nsetoption name EvalFile value /path/to/net.nnue\nisready\ngo depth 1\nquit\n" | ./stockfish/src/stockfish
```

## cutechess (UHO, concurrency 32)

Use the binary that matches the net: **MoE** `STACKS=moe`, **LayerStacks** `STACKS=layer`, **NoStacks** `STACKS=none` (a layer binary cannot load a MoE `.nnue`).

```bash
nix shell nixpkgs#cutechess -c cutechess-cli \
  -tournament round-robin \
  -engine name=MoE-TeacherCE-Only3 cmd=/home/rgodha/Developer/stockfish-trainer/stockfish/stockfish-moe-teacher-ce-tournament dir=/home/rgodha/Developer/stockfish-trainer/stockfish option.EvalFile=/tmp/stockfish-validate/moe-teacher-ce-only3-latest.nnue \
  -engine name=MoE-Curriculum cmd=/home/rgodha/Developer/stockfish-trainer/stockfish/stockfish-moe-teacher-ce-tournament dir=/home/rgodha/Developer/stockfish-trainer/stockfish option.EvalFile=/tmp/stockfish-validate/moev3-curriculum-latest.nnue \
  -engine name=LayerStacks-1 cmd=/home/rgodha/Developer/stockfish-trainer/stockfish/stockfish-layerstacks-1-local dir=/home/rgodha/Developer/stockfish-trainer/stockfish option.EvalFile=/tmp/stockfish-validate/layerstacks-1-latest.nnue \
  -engine name=NoStacks cmd=/home/rgodha/Developer/stockfish-trainer/stockfish/stockfish-nonstacks-local dir=/home/rgodha/Developer/stockfish-trainer/stockfish option.EvalFile=/tmp/stockfish-validate/nonstacks-none-fixed.nnue \
  -each proto=uci tc=10+0.1 option.Threads=1 option.Hash=128 \
  -openings file=$HOME/Downloads/opening/UHO_2024/UHO_2024_+100_+109/UHO_2024_8mvs_big_+095_+114.epd format=epd order=random \
  -games 60 -repeat -concurrency 32 -ratinginterval 20 \
  -event "Stack modes UHO STC" -pgnout /tmp/stack_modes_uho.pgn
```

Four-way round-robin: the schedule is usually more than `-games` board games. Swap the `.epd` for another file under `~/Downloads/opening/UHO_2024/` if you want a different book slice.
