import argparse
import sys
import matplotlib.pyplot as plt


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log_file",
        help="path to log file"
    )
    parser.add_argument('-log', action='store_true', default=False,
                        help='graph in log scale')
    parser.add_argument('-max', type=int, default=10,
                        help='Only plot loss values under this value')
    args = parser.parse_args()

    f = open(args.log_file)

    lines = [line.rstrip("\n") for line in f.readlines()]

    numbers = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}

    iters = []
    losses = []
    learning_rate = []
    time = []
    images = 0

    fig, ax = plt.subplots()

    prev_line = ""
    for line in lines:
        line = line.split(' ')
        if line[0][-1:] == ':' and line[0][0] in numbers:
            loss = float(line[2])
            if loss < args.max:
                iters.append(int(line[0][:-1]))
                losses.append(loss)
                learning_rate.append(float(line[4]))
                time.append(float(line[6]))
                images += int(line[8])


    if args.log:
        ax.set_yscale('log')
    ax.plot(iters, losses)
    plt.xlabel('Iterations current:' + str(len(iters)))
    plt.ylabel('Loss')
    plt.grid()



    ticks = range(0, 250, 10)

    # ax.set_yticks(ticks)
    plt.show()


if __name__ == "__main__":
    main(sys.argv)