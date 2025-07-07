'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''

"""
Quantize and compile the PyTorch model.
Usage from inside Palette docker: python run_modelsdk.py
"""

"""
Author: Mark Harvey
"""

import os
import sys
import argparse
import numpy as np
import logging
import tarfile

# Palette-specific imports
from afe.load.importers.general_importer import ImporterParams, pytorch_source
from afe.apis.defines import default_quantization, gen1_target,gen2_target
from afe.apis.loaded_net import load_model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version


# user imports
DIVIDER = '-'*50




def implement(args):

    enable_verbose_error_messages()

    # get filename from full path
    filename = os.path.splitext(os.path.basename(args.model_path))[0]

    # set an output path for saving results
    output_path = f'{args.build_dir}/{filename}'

    # load the floating-point model
    print('Loading..')
    if (args.generation==2):
        target=gen2_target
    else:
        target=gen1_target
    input_names = ['x']
    input_shapes = [(1, 3, 224, 224)]
    importer_params: ImporterParams = pytorch_source(args.model_path, input_names, input_shapes)
    loaded_net = load_model(importer_params,target=target)

    # calibration data (always NHWC)
    # calibration data must be an iterable type
    calibration_data=[]
    data=dict()
    for n,s in zip(input_names,input_shapes):
        data_in=np.random.rand(s[0],s[2],s[3],s[1])
        data={**data, n:data_in} 
    calibration_data.append(data)

    # quantize
    print('Quantizing..')
    quant_model = loaded_net.quantize(
        calibration_data=calibration_data,
        quantization_config=default_quantization,
        model_name=filename,
        log_level=logging.INFO
    )
    quant_model.save(model_name=filename, output_directory=output_path)
    print("Quantized and saved to", output_path)

    # compile
    print('Compiling...')
    quant_model.compile(
        output_path=output_path,
        batch_size=args.batch_size,
        log_level=logging.INFO)
    print("Compiled model written to", output_path)

    # extract compiled model
    model_tar = f'{output_path}/{filename}_mpk.tar.gz'
    with tarfile.open(model_tar) as model:
        model.extractall(output_path)

def run_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd', '--build_dir', type=str, default='build', help='Path of build folder. Default is build')
    ap.add_argument('-m', '--model_path', type=str, default='./pyt/resnext101_32x8d_wsl.pt', help='path to FP32 model')
    ap.add_argument('-b', '--batch_size', type=int, default=1, help='requested batch size of compiled model. Default is 1')
    ap.add_argument('-g', '--generation', type=int, choices=[1,2], default=1, help='Specify target platform, choices are 1 (MLSoC) or 2 (Modalix). Default is 1')
    args = ap.parse_args()

    # ensure build directory exists
    os.makedirs(args.build_dir, exist_ok=True)

    print(DIVIDER)
    print('Model SDK version', get_model_sdk_version())
    print(sys.version)
    print(DIVIDER)

    implement(args)


if __name__ == '__main__':
    run_main()