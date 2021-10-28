import glob
import tqdm
import numpy as np
import multiprocessing


def func(p):
    try:
        zf = np.load(p)
        if zf.files[0] == "arr":
            ...
        else:
            # print(zf.files[0])
            arr = zf[zf.files[0]]
            np.savez_compressed(p, arr=arr)

    except:
        print(p)


path_list = glob.glob("./data/train/*/*.npz")
if __name__ == "__main__":
    with multiprocessing.Pool(56) as pool:
        gen = pool.imap(func, path_list)
        for _ in tqdm.tqdm(gen, total=len(path_list)):
            ...
