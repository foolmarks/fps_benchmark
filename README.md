# Benchmarking a model throughput with Sima.ai ModelSDK #

This tutorial shows how to benchmark a model throughput using using ModelSDK which is included in Palette 1.6


## Design Flow ##

Start the SDK docker container - just reply with './' when asked for the work directory:

```shell
python ./start.py
```

The output in the console should look like this:

```shell
user@ubmsh:~/tutorials/modelsdk_pytorch$ ./start.py 
Set no_proxy to localhost,127.0.0.0
Using port 49152 for the installation.
Checking if the container is already running...
Enter work directory [/home/user/tutorials/modelsdk_pytorch]: ./
Starting the container: palettesdk_1_6_0_Palette_SDK_master_B163
Checking SiMa SDK Bridge Network...
SiMa SDK Bridge Network found.
Creating and starting the Docker container...
b376b867257233623491103715372c56d56d0403ca3c0da4d68a3cde2c7c6a27
Successfully copied 3.07kB to /home/user/tutorials/modelsdk_pytorch/passwd.txt
Successfully copied 3.07kB to palettesdk_1_6_0_Palette_SDK_master_B163:/etc/passwd
Successfully copied 2.56kB to /home/user/tutorials/modelsdk_pytorch/shadow.txt
Successfully copied 2.56kB to palettesdk_1_6_0_Palette_SDK_master_B163:/etc/shadow
Successfully copied 2.56kB to /home/user/tutorials/modelsdk_pytorch/group.txt
Successfully copied 2.56kB to palettesdk_1_6_0_Palette_SDK_master_B163:/etc/group
Successfully copied 3.58kB to /home/user/tutorials/modelsdk_pytorch/sudoers.txt
Successfully copied 3.58kB to palettesdk_1_6_0_Palette_SDK_master_B163:/etc/sudoers
Successfully copied 2.05kB to palettesdk_1_6_0_Palette_SDK_master_B163:/home/docker/.simaai/.port
user@b376b8672572:/home$ 
```

Navigate to the workspace:

```shell
cd docker/sima-cli
```


### The PyTorch model ###

Download the PyTorch ResNext101 model:

```shell
cd pyt
python make_resnext.py
cd ..
```


### Quantize & Compile ###

The run_modelsdk.py script will do the following:

* load the floating-point PyTorch model.
* quantize using random calibration data and default quantization parameters.
* compile to generate a tar.gz

```shell
python run_modelsdk.py
```

If this runs correctly, the final output messages in the console will be like this:

```shell
2025-07-07 05:57:32,420 - afe.backends.mpk.interface - INFO - ==============================
2025-07-07 05:57:32,420 - afe.backends.mpk.interface - INFO - Compilation summary:
2025-07-07 05:57:32,420 - afe.backends.mpk.interface - INFO - ------------------------------
2025-07-07 05:57:32,420 - afe.backends.mpk.interface - INFO - Desired batch size: 1
2025-07-07 05:57:32,420 - afe.backends.mpk.interface - INFO - Achieved batch size: 1
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO - ------------------------------
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO - Plugin distribution per backend:
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO -   MLA : 1
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO -   EV74: 5
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO -   A65 : 0
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO - ------------------------------
2025-07-07 05:57:32,421 - afe.backends.mpk.interface - INFO - Generated files: resnext101_32x8d_wsl_stage1_mla_stats.yaml, resnext101_32x8d_wsl_mpk.json, boxdecoder.json, postproc.json, preproc.json, process_mla.json, resnext101_32x8d_wsl_stage1_mla.elf, quanttess.json
Compiled model written to build/resnext101_32x8d_wsl
```

The output files will be written to build/resnext101_32x8d_wsl.


### Run benchmarking on a devkit ###

Ensure that you have working ssh connection between the target board and the host system.


Run the following command from within the Palette docker container - note that you will have to modify the IP address specified by the --dv_host argument to match the one assigned to your board:


```shell
python ./get_fps/network_eval/network_eval.py \
    --model_file_path   ./build/resnext101_32x8d_wsl/resnext101_32x8d_wsl_stage1_mla.elf \
    --mpk_json_path     ./build/resnext101_32x8d_wsl/resnext101_32x8d_wsl_mpk.json \
    --dv_host           192.168.8.20 \
    --dv_user           sima \
    --image_size        224 224 3 \
    --verbose \
    --bypass_tunnel \
    --max_frames        1000 \
    --batch_size        1
```


The measured throughtput (fps) will be reported like this:


```shell
Running model in MLA-only mode
Copying the model files to DevKit
sima@192.168.8.20's password: 
ZMQ Connection successful.
FPS = 405
FPS = 410
FPS = 412
FPS = 413
FPS = 413
FPS = 414
.
.
FPS = 417
FPS = 417
FPS = 417
Ran 1000 frame(s)
```


### Changing batch size ###

The model can be compiled for different batch sizes...just execute run_modelsdk.py with the --batch_size argument set to a number > 1:


```shell
python run_modelsdk.py --batch_size 4
```


The final part of the console output will report the requested and achieved batch sizes:



```shell
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - ==============================
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - Compilation summary:
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - ------------------------------
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - Desired batch size: 4
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - Achieved batch size: 4
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - ------------------------------
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO - Plugin distribution per backend:
2025-07-07 06:50:03,798 - afe.backends.mpk.interface - INFO -   MLA : 1
2025-07-07 06:50:03,799 - afe.backends.mpk.interface - INFO -   EV74: 5
2025-07-07 06:50:03,799 - afe.backends.mpk.interface - INFO -   A65 : 0
2025-07-07 06:50:03,799 - afe.backends.mpk.interface - INFO - ------------------------------
```



Now we can run the benchmarking with a batch size of 4:



```shell
python ./get_fps/network_eval/network_eval.py \
    --model_file_path   ./build/resnext101_32x8d_wsl/resnext101_32x8d_wsl_stage1_mla.elf \
    --mpk_json_path     ./build/resnext101_32x8d_wsl/resnext101_32x8d_wsl_mpk.json \
    --dv_host           192.168.8.175 \
    --dv_user           sima \
    --image_size        224 224 3 \
    --verbose \
    --bypass_tunnel \
    --max_frames        1000 \
    --batch_size        4
```


..and this will give higher throughput:




```shell
Copying the model files to DevKit
sima@192.168.8.20's password: 
ZMQ Connection successful.
FPS = 796
FPS = 805
FPS = 806
FPS = 808
FPS = 808
.
.

FPS = 816
FPS = 816
FPS = 816
FPS = 816
FPS = 816
FPS = 816
Ran 1000 frame(s)
```



### Choosing target platform ###


When quantizing and compiling, the user can choose between the Generation 1 device (MLSoC) or the second generation Modalix device.

The default is to compile for MLSoC, users can choose Modalix like this:

```shell
python run_modelsdk.py --generation 2
```


