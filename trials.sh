for n_trials in 0 1 2 3 4
do
    for k in 0 1 2 3 4 5
    do
        python . -d Point --epochs 1 --dont-write --test-freq 1 -t Box --test out/convSmall.pynet -D=CIFAR10 -k=$k
    done
done