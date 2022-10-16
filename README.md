# FADEC

FPGA-based Acceleration of Video Depth Estimation by HW/SW Co-design with NNgen

Copyright 2022, Nobuho Hashimoto and Shinya Takamaeda-Yamazaki


## License

Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


## Summary

TBD


## Requirement

TBD


## Procedure

1. Prepare datasets and a model
1. Adjust datasets to our implementation
1. Quantize weights and activation
1. Export network input and output
1. Export HDL
1. Generate bitstream
1. Execute FADEC on FPGA
1. Evaluate results


## 1. Prepare datasets and a model

TBD


## 2. Adjust datasets to our implementation

TBD


## 3. Quantize weights and activation

TBD


## 4. Export network input and output

TBD


## 5. Export HDL

TBD


## 6. Generate bitstream

TBD

- The design image is shown in the figure below.

    ![design image](./img/design_1.png)


## 7. Execute FADEC on FPGA

- Place [`./eval/fadec`](./eval/fadec) on ZCU104.
- Place `design_1.bit` and `design_1.hwh` in `fadec` directory on ZCU104.
    - [`./dev/vivado/move_bitstream.sh`](./dev/vivado/move_bitstream.sh) is helpful to find and move these files.

        ```sh
        $ cd dev/vivado
        $ ./move_bitstream.sh /path/to/vivado_project_directory pynq:/path/to/fadec
        # The project directory name should be "dvmvs".
        # You can also specify a remote directory for the project directory.
        ```

    - You can also download these files from https://projects.n-hassy.info/storage/fadec/design_1.zip
- Execute [`7scenes_evaluation.ipynb`](./eval/fadec/7scenes_evaluation.ipynb) on ZCU104.
    - Outputs will be stored in [`depths`](./eval/fadec/depths) and [`time_fadec.txt`](./eval/fadec/time_fadec.txt).
    - If the following error happens in the 7th cell, reboot ZCU104 and retry.

        ```
        Output exceeds the size limit. Open the full output data in a text editor
        ---------------------------------------------------------------------------
        RuntimeError                              Traceback (most recent call last)
        <ipython-input-7-6778394090c9> in <module>()
            1 memory_size = 1024 * 1024 * 192
        ----> 2 buf = allocate(shape=(memory_size,), dtype=np.uint8)
            3 buf[param_offset:param_offset + params.size] = params.view(np.int8)

        /usr/local/lib/python3.6/dist-packages/pynq/buffer.py in allocate(shape, dtype, target, **kwargs)
            170     if target is None:
            171         target = Device.active_device
        --> 172     return target.allocate(shape, dtype, **kwargs)

        /usr/local/lib/python3.6/dist-packages/pynq/pl_server/device.py in allocate(self, shape, dtype, **kwargs)
            292
            293         """
        --> 294         return self.default_memory.allocate(shape, dtype, **kwargs)
            295
            296     def reset(self, parser=None, timestamp=None, bitfile_name=None):

        /usr/local/lib/python3.6/dist-packages/pynq/xlnk.py in allocate(self, *args, **kwargs)
            255
            256         """
        --> 257         return self.cma_array(*args, **kwargs)
            258
            259     def cma_array(self, shape, dtype=np.uint32, cacheable=0,
        ...
        --> 226             raise RuntimeError("Failed to allocate Memory!")
            227         self.bufmap[buf] = length
            228         return self.ffi.cast(data_type + "*", buf)

        RuntimeError: Failed to allocate Memory!
        ```

## 8. Evaluate results

- Place [`./eval/cpp`](./eval/cpp) and [`./eval/cpp_with_ptq`](./eval/cpp_with_ptq) on ZCU104.
- Execute C++ implementations by the following commands.

    ```bash
    $ cd /path/to/cpp
    $ make
    $ ./a.out > time_cpp.txt
    $ cd /path/to/cpp_with_ptq
    $ make
    $ ./a.out > time_cpp_with_ptq.txt
    ```

    - Outputs will be stored in `results` and (`time_cpp.txt` or `time_cpp_with_ptq.txt`).
- Execute [`calculate_time.py`](./eval/calculate_time.py) by the following commands.

    ```bash
    $ cd /path/to/eval
    $ python3 calculate_time.py
    ```
