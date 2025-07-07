Running Networks on DevKit and Obtaining FPS KPIs

The software toolchain compiles ML networks into a combination of .lm and .so
files.  Additionally, it produces an MPK JSON file.  The network_eval.py script
runs the generated .lm/.elf and .so files on the DevKit with random input frames, and
emits FPS KPIs to the console.

Example 1: Obtain KPIs for ONNX ResNet50 (224x224x3 input size), which runs entirely
on the MLA.

python network_eval.py --model_file_path resnet50_MLA_0.lm --mpk_json_path resnet50.json --dv_host 192.168.91.21 --image_size 224 224 3

Example 2: Obtain KPIs for ONNX MobileNet V3 (224x224x3 input size), which has been
compiled to run entirely on the A65.

python network_eval.py --model_file_path mobilenet_v3_large.tar.gz --mpk_json_path mobilenet_v3_large.json --dv_host 192.168.91.21 --image_size 224 224 3

Example 3: Obtain per layer statistics such as runtime for each layer. Need to supply
the Layer YAML file as an input. Output will be <layer_input_filename>_output.yaml and
will be stored in the current directory. The script will only do a single run and will
exit after copying the output YAML file.

NOTE: Only valid for MLA Only mode (i.e. with .lm/.elf file)
python network_eval.py --model_file_path UR_onnx_resnet50-v1-80-sparse_fp32_224_224_stage1_mla.lm --mpk_json_path UR_onnx_resnet50-v1-80-sparse_fp32_224_224_mpk.json --dv_host 192.168.91.21 --max_frames 1 --image_size 224 224 3 --layer_stats_path UR_onnx_resnet50-v1-80-sparse_fp32_224_224_stage1_mla_stats.yaml

In the case of cloning the repo, if facing an import issue as ModuleNotFoundError : No module named 'devkit_inference_models', you can export the python-path as:
export PYTHONPATH={PATH_TO_REPO}/devkit-inference-examples

In the case of using Palette SDK CLI package, you can export the python-path as:
export PYTHONPATH={PATH_TO_SDK_EXTRACTED}/{Palette_SDK_PACKAGE_NAME}/sdk_examples/app_zoo/model-accelerator-mode/devkit-inference-examples

