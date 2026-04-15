ok so basically this is a rewrite of the stockfish nnue-pytorch repo
github.com/rgodha24/nnue-pytorch is my personal fork of the original, which inlcudes the rewritten rust dataloader, and a few other optimizations

main stuff we r doing is just to get the training FAST so that i can do my actual ML class project, which is to explore alternatatives to the LayerStacks method of choosing the hidden layer (e.g. including using a learned router, having no stacks (as a control), etc etc)

my desktop is a ryzen 9 9950x (16 core 32 thrreads 64gb ram) and 5060ti 16gb gpu
i also have access to GT PACE ICE (a slurm cluster w gpus). pace is harder to use because it becomes very very cpu bound. doing some work on ray & parallelizing across nodes to acc feed the data to the gpus rn.

we are following the training recipe from this commit & repo.
https://github.com/vondele/nettest/blob/94da8f63ff49a53a24a072c4205187f4a7e78e94/threats.yaml

the ghfs skill is very useful here
