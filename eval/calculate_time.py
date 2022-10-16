import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


print("CPU-only:")
try:
    with open(os.path.join(base_dir, 'cpp/time_cpp.txt'), 'r') as f:
        data = f.read().split('\n')

    speeds = []
    for d in data:
        if len(d) > 0 and d[0] == '1':
            speeds.append(float(d.split()[0]))
    speeds.sort()
    speeds = np.array(speeds)
    print("\tmedian %.3f" % np.median(speeds))
    print("\tstd: %.3f" % np.std(speeds))
except Exception as e:
    print("\tSkipping... cpp/time_cpp.txt does not exist.")


print("CPU-only (w/ PTQ):")
try:
    with open(os.path.join(base_dir, 'cpp_with_ptq/time_cpp_with_ptq.txt'), 'r') as f:
        data = f.read().split('\n')

    speeds = []
    for d in data:
        if len(d) > 0 and d[0] == '1':
            speeds.append(float(d.split()[0]))
    speeds.sort()
    speeds = np.array(speeds)
    print("\tmedian %.3f" % np.median(speeds))
    print("\tstd: %.3f" % np.std(speeds))
except Exception as e:
    print("\tSkipping... cpp_with_ptq/time_cpp_with_ptq.txt does not exist.")


print("PL + CPU (ours): ")
try:
    with open(os.path.join(base_dir, 'fadec/time_fadec.txt'), 'r') as f:
        data = f.read().split('\n')

    speeds = []
    for d in data:
        if len(d) > 0 and d[0] == '0':
            speeds.append(float(d))
    speeds.sort()
    speeds = np.array(speeds)
    print("\tmedian %.3f" % np.median(speeds))
    print("\tstd: %.3f" % np.std(speeds))
except Exception as e:
    print("\tSkipping... fadec/time_fadec.txt does not exist.")