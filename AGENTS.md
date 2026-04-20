ok so basically this is a rewrite of the stockfish nnue-pytorch repo
github.com/rgodha24/nnue-pytorch is my personal fork of the original, which inlcudes the rewritten rust dataloader, and a few other optimizations

main stuff we r doing is just to get the training FAST so that i can do my actual ML class project, which is to explore alternatatives to the LayerStacks method of choosing the hidden layer (e.g. including using a learned router, having no stacks (as a control), etc etc)

my desktop is a ryzen 9 9950x (16 core 32 thrreads 64gb ram) and 5060ti 16gb gpu
i also have access to GT PACE ICE (a slurm cluster w gpus). pace is harder to use because it becomes very very cpu bound. doing some work on ray & parallelizing across nodes to acc feed the data to the gpus rn.

we are following the training recipe from this commit & repo.
https://github.com/vondele/nettest/blob/94da8f63ff49a53a24a072c4205187f4a7e78e94/threats.yaml

the ghfs skill is very useful here

layout:
src/train/singlenode.py is the local single-node trainer entrypoint.
src/train/multinode.py is the Ray-backed trainer entrypoint.
src/rust/ contains loader code (in rust)
src/model/ contains model related code (eg kernels etc too)
src/data/ contains more loader code. mostly just thin wrappers around rust code. eventually networking stuff
src/distributed/ contains Ray/distributed loader code and smoke-test tooling

quick ops (build + tournament):

- serialize checkpoint -> nnue:
  `uv run python -m src.scripts.serialize <model.pt> <out.nnue>`
- build layerstacks:
  `cp <layerstacks.nnue> stockfish/src/nn-f68ec79f0fe3.nnue && cp /mnt/external/models/nn-47fc8b7fff06.nnue stockfish/src/nn-47fc8b7fff06.nnue && make -C stockfish/src clean && make -C stockfish/src -j16 ARCH=native moe=no all && cp stockfish/src/stockfish stockfish/stockfish-layerstacks-local`
- build moe:
  `cp <moe.nnue> stockfish/src/nn-f68ec79f0fe3.nnue && cp /mnt/external/models/nn-47fc8b7fff06.nnue stockfish/src/nn-47fc8b7fff06.nnue && make -C stockfish/src clean && make -C stockfish/src -j16 ARCH=native moe=yes all && cp stockfish/src/stockfish stockfish/stockfish-moe-local`
- smoke test:
  `printf "uci\nisready\nposition startpos\ngo depth 1\nquit\n" | ./stockfish/stockfish-layerstacks-local`
- tournament (moe vs layerstacks):
  `nix shell nixpkgs#cutechess -c cutechess-cli -engine name=LayerStacks cmd="/home/rgodha/Developer/stockfish-trainer/stockfish/stockfish-layerstacks-local" dir="/home/rgodha/Developer/stockfish-trainer/stockfish" -engine name=MOE cmd="/home/rgodha/Developer/stockfish-trainer/stockfish/stockfish-moe-local" dir="/home/rgodha/Developer/stockfish-trainer/stockfish" -each proto=uci tc=60+0.6 option.Threads=1 option.Hash=16 -concurrency 8 -rounds 20 -pgnout "/tmp/moe_vs_layerstacks.pgn"`
- notes: reuse `nn-47fc8b7fff06.nnue` as small net; `EvalFile*` options are optional if defaults are embedded/available.
