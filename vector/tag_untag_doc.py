__author__ = 'HyNguyen'
import argparse

def untag(fi, fo):
    fo = open(fo, mode="w")
    with open(fi, mode="r") as f:
        for counter, line in enumerate(f):
            words = line.split()
            fo.write(" ".join(words[1:]) + "\n")
    fo.close()

def tag(fi, fo, name):
    fo = open(fo, mode="w")
    with open(fi, mode="r") as f:
        for counter, line in enumerate(f):
            fo.write( "{0}_{1} ".format(name,counter) + line )
    fo.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', required=True, type=str)
    parser.add_argument('-fo', required=True, type=str)
    parser.add_argument('-mode', required=True, type=str)
    parser.add_argument('-name', type=str, default="hyhy")
    args = parser.parse_args()
    fi = args.fi
    fo = args.fo
    mode = args.mode
    name = args.name

    if mode == "tag":
        tag(fi,fo,name)
    elif mode == "untag":
        untag(fi,fo)

