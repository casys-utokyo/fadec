import numpy as np
from path import Path
import os
import struct
import matplotlib.pyplot as plt

def compute_mse(gt, pred):
    valid1 = gt >= 0.5
    valid2 = gt <= 20.0
    valid = valid1 & valid2
    gt = gt[valid]
    pred = pred[valid]

    if len(gt) == 0:
        return np.nan, np.nan

    differences = gt - pred
    squared_differences = np.square(differences)
    mse = np.mean(squared_differences)

    return mse


if __name__ == '__main__':
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    gt_dir = base_dir / ("../dev/dataset_converter/7scenes/data_7scenes")
    test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"]
    gts = [np.load(gt_dir / test_dataset_name + ".npz")['ground_truth'][1:] for test_dataset_name in test_dataset_names]
    mses_mean = []
    for name in ["cpp", "cpp_with_ptq", "fadec"]:
        mses_mean.append([])
        for i, test_dataset_name in enumerate(test_dataset_names):
            if name == "fadec":
                files = base_dir / 'fadec/depths' / test_dataset_name + ".npz"
                predictions = np.load(files)['depths'][:,0,0,:,:]
            else:
                files = sorted((base_dir / name / 'results_7scenes' / test_dataset_name).files("*.bin"))
                predictions = []
                for file in files:
                    data = []
                    with open(file, 'rb') as f:
                        while True:
                            d = f.read(4)
                            if len(d) != 4:
                                break
                            data.append(struct.unpack('f', d))
                    predictions.append(np.array(data).reshape(64, 96))

            print(name + "/" + test_dataset_name)
            assert len(predictions) == len(gts[i]), (name, test_dataset_name, len(predictions), len(gts[i]))

            mses = []
            for j, prediction in enumerate(predictions):
                mse = compute_mse(gts[i][j], prediction)
                mses.append(mse)
                print('%s: %.3f' % (j, mse))

            mses_mean[-1].append(np.mean(mses))

    labels = ['C++', 'C++ (w/ PTQ)', 'Ours']
    colors = ['r', 'y', 'b']

    print("Dataset Nemes: ", test_dataset_names)
    print("MSE:")

    x = np.arange(len(test_dataset_names))
    width = 0.2
    plt.figure()
    save_output = {}
    for i, label in enumerate(labels):
        print("\t%12s: " % label, np.round(mses_mean[i], 2))
        plt.bar(x + (i-len(labels)/2) * width, mses_mean[i], label=labels[i], color=colors[i], width=width, align='edge')
    plt.ylim(0, 1.15)
    plt.legend()
    plt.ylabel('MSE')
    plt.xticks(x, test_dataset_names, rotation=30)
    plt.savefig(base_dir / "./errors.png", bbox_inches="tight", pad_inches=0.1)