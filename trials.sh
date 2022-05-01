radius=".01"

for trial in 0 1 2 3 4
do
    for k in 0 1 2 3 4 5 6
    # for k in 6 7 8
    do
        # python . -d Point --epochs 1 --dont-write --test-freq 1 -t Box --test out/convSmall.pynet -D=CIFAR10 -k=$k
        python . -d Point --epochs 1 --dont-write --test-freq 1 -t "Box($radius)" --test out/convSmall.pynet -D=CIFAR10 -k=$k --save_suffix="-$radius"
    done
done