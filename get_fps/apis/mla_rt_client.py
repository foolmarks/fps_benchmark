#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
# Code owner: Vimal Nakum
#########################################################
"""
This is remote procedure call (RPC) client.
"""
import rpyc
import argparse
import shutil
import os
import time
import ctypes
import numpy as np
import zmq

from devkit_inference_models.utils.common_utils import check_tar_gz_file

MLA_RPC_CLIENT_VERSION=1.07
DEFAULT_ZMQ_PORT = 43777

# use this to store daddr/len pair for run_model_multi_phys api
class DADDR_LEN:
    def __init__(self, daddr, len):
        self.daddr = daddr
        self.len = len

class Mla_rpc_client:

    def __init__(self, host, port, pipeline):
        self.remote = None
        self.host = host
        self.port = port
        self.conn = None
        self.model = None
        self.ifm = None
        self.ofm = None
        self.pipeline = pipeline
        # This version has to match with the server version for compatibility
        return

    def connect(self):
        # Check if secure ssl connection is requested
        self.conn = rpyc.connect(self.host, self.port,config={'allow_public_attrs': True, "allow_pickle":True, "sync_request_timeout":None})
        if (not self.conn):
            raise Exception("[ ERROR ] Failed to connect to {}:{}",format(self.host, self.port))

        # test connection
        # if rpyc server already has an active connection, it would close the new
        # connection, which results in EOFError, so catch that error and return
        # error to the caller
        try:
            if (self.conn.root.ping("ping") != "pong"):
                raise Exception("[ ERROR ] Ping test failed")
        except EOFError:
            print(f"RPYC Connection failed as DevKit {self.host} is busy")
            return False
        # check versions
        rc = self.conn.root.check_version(MLA_RPC_CLIENT_VERSION)
        
        if (rc) is not True:
            raise Exception("[ ERROR ] Version check failed server version {} client version {}".format(rc, MLA_RPC_CLIENT_VERSION))

        return True

    def dv_connect(self, layer_stats_path=None):
        """
        connects to DV board using non-pipelined mode
        """
        # pass the layer file path to connect, which would
        # pass it to get_handle function eventually
        layer_file_path = None
        if layer_stats_path:
            layer_file_path = "/home/sima/" + os.path.basename(layer_stats_path)
        rc = self.conn.root.dv_connect(layer_file_path)
        if (rc) is not True:
            raise Exception("[ ERROR ] Connecting dv failed")
        
        # Context and socket using the local port forwarded by the tunnel
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PAIR)
        try:
            self.zmq_socket.connect(f"tcp://localhost:{DEFAULT_ZMQ_PORT}")
            print("ZMQ Connection successful.")
        except zmq.ZMQError as e:
            print(f"Failed to connect: {e}")
        except Exception as e:
            print(f"Failed to connect: {e}")
        
        #Set NO TIMEOUT for ZMQ
        t = -1 # NO TIMEOUT
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, t)
        return rc

    def pm_connect(self, tar_file_path):
        """
        connects to DV board using pipelined mode
        """        
        # keep only the file name
        tar_file_name = os.path.basename(tar_file_path)
        assert check_tar_gz_file(tar_file_name), f"{tar_file_name} must be in tar format."
        rc = self.conn.root.pm_connect(tar_file_name)
        if (rc) is not True:
                raise Exception("[ ERROR ] Connecting to DV Pipeline failed")
        return rc

    # load the model that is already present on the card
    # Support the lm and elf file
    def load_model_only(self, model_name):
        # Execute load model call on the server
        self.model = self.conn.root.load_model(model_name)
        if not self.model:
            raise Exception("[ ERROR ] Load model failed")
        return self.model

    # allocate ifm, later use upload_ifm to upload data into this ifm
    def allocate_ifm(self, ifm_len):
        if not ifm_len:
            logger.error("invalid ifm length 0")
            return None

        self.ifm = self.conn.root.allocate_ifm(ifm_len)
        return self.ifm

    def allocate_array_ifm(self, ifm_len, array_size: int=2):
        """
        Function to allocare array of IFM.
        :params ifm_len: the len of ifm
        :params array_size: the size of IFM array

        return: the array of virtual memory buffer IFM if successful
        """
        if not ifm_len:
            logger.error("invalid ifm length 0")
            return None

        return self.conn.root.allocate_array_ifm(ifm_len, array_size)

    def allocate_array_ifm_with_diff_size(self, array_ifm_len, array_size: int=2):
        """
        Function to allocate the array of virtual memory for IFM with different size of each IFM.
        :params array_ifm_len: list of the length of IFM.
        :params array_size: the size of array of virtual memory buffer IFM.

        return: the array of virtual memory buffer IFM if successful
        """
        if not array_ifm_len or not isinstance(array_ifm_len, list):
            logger.error("invalid ifm length 0")
            return None

        return self.conn.root.allocate_array_ofm_with_diff_size(array_ifm_len, array_size)

    def upload_array_ifm(self, arr_mem_buff_ifm, array_ifm_data):
        """
        Function to upload the array of IFM data to the card
        :params arr_mem_buff_ifm: the array of virtual memory buffer IFM
        :params array_ifm_data: the data of its array IFM (needs to be in bytes)

        return: True if successful to upload IFM.
        """
        if not arr_mem_buff_ifm or not array_ifm_data:
            logger.error("Invalid the array of IFM or the array of IFM data")
            return False

        rc = self.conn.root.upload_array_ifm(arr_mem_buff_ifm, array_ifm_data)
        return rc

    # upload the provided ifm_data to the card allocated ifm buffer
    # ifm_data needs to be in bytes
    def upload_ifm(self, ifm, ifm_data):
        if not ifm or not ifm_data:
            logger.error("invalid ifm or ifm_data")
            return False

        rc = self.conn.root.upload_ifm(ifm, ifm_data)
        return rc

    # use zmq to upload ifm data, as it is more efficient than
    # using rpyc
    def upload_ifm_zmq(self, ifm, ifm_data):
        if not ifm or not ifm_data:
            logger.error("invalid ifm or ifm_data")
            return False
        self.zmq_socket.send(ifm_data)
        rc = self.conn.root.upload_ifm_zmq(ifm, len(ifm_data))
        return rc

    # allocate OFM on the card
    def allocate_ofm(self, ofm_len):
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return None
        self.ofm = self.conn.root.allocate_ofm(ofm_len)
        return self.ofm

    def allocate_array_ofm(self, ofm_len, array_size: int=1):
        """
        Function to allocate the array of OFM virtual memory.
        :params ofm_len: the length of OFM.
        :params array_size: the size of array of virtual memory buffer OFM.

        return: True, pointer of the array of virtual memory buffer OFM.
        """
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return False, None

        return self.conn.root.allocate_array_ofm(ofm_len, array_size)

    def allocate_array_ofm_with_diff_size(self, array_ofm_len, array_size: int=1):
        """
        Function to allocate the array of OFM virtual memory.
        :params array_ofm_len: the length of OFM.
        :params array_size: the size of array of virtual memory buffer OFM.

        return: True, pointer of the array of virtual memory buffer OFM.
        """
        if not array_ofm_len or not isinstance(array_ofm_len, list):
            logger.error("invalid ofm length 0")
            return False, None

        return self.conn.root.allocate_array_ofm_with_diff_size(array_ofm_len, array_size)

    # download ofm data from the card and return as bytes
    def download_ofm_only(self, ofm, ofm_len):
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return None
        # copy the ofm data from model result on the ard
        ofm_bytes = self.conn.root.download_ofm(ofm, ofm_len)
        return ofm_bytes

    def download_array_ofm(self, arr_mem_buff_ofm, ofm_len):
        """
        Function to download the array of OFM.
        :params arr_mem_buff_ofm: the array of virtual memory buffer OFM.
        :params ofm_len: the length of OFM.

        return: True, the array of OFM data.
        """
        # download ofm data from the card and return as bytes
        if not arr_mem_buff_ofm or not ofm_len:
            logger.error("Invalid the array of OFM or the length of OFM")
            return False, None
        rc, array_ofm_buff = self.conn.root.download_array_ofm(arr_mem_buff_ofm, ofm_len)
        if rc:
            return True, array_ofm_buff


    # download ofm data from the devkit and return as bytes
    # use zmq to download data as it is more efficient than
    # using rpyc
    def download_ofm_only_zmq(self, ofm, ofm_len):
        if not ofm_len:
            logger.error("invalid ofm length 0")
            return None
        # copy the ofm data from model result on the devkit
        self.conn.root.download_ofm_zmq(ofm, ofm_len)
        ofm_bytes = self.zmq_socket.recv()
        return ofm_bytes
    
    def free_ifm(self, ifm):
        if ifm:
            self.conn.root.free_ifm(ifm)
        return

    def free_array_ifm(self, arr_mem_buff_ifm):
        """
        Function to free the array of IFM virtual memory buffer
        """
        if arr_mem_buff_ifm:
            self.conn.root.free_array_ifm(arr_mem_buff_ifm)
        return

    def free_array_ofm(self, arr_mem_buff_ofm):
        """
        Function to free the array of OFM virtual memory buffer
        """
        if arr_mem_buff_ofm:
            self.conn.root.free_array_ofm(arr_mem_buff_ofm)
        return

    def free_ofm(self, ofm):
        if ofm:
            self.conn.root.free_ofm(ofm)
        return

    # run the model
    def run_model_phys(self, model, ifm, ofm):
        if model and ifm and ofm:
            rc, run_time = self.conn.root.run_model_phys(model, ifm, ofm)
            return rc, run_time
        else:
            return -1, 0

    # run the batch model
    def run_batch_model_phys(self, model, batch_size, arr_mem_buff_ifm, ifm_size, arr_mem_buff_ofm, ofm_size):
        if model and arr_mem_buff_ifm and arr_mem_buff_ofm and batch_size and ifm_size and ofm_size:
            rc, run_time = self.conn.root.run_batch_model_phys(model, batch_size, arr_mem_buff_ifm, ifm_size, arr_mem_buff_ofm, ofm_size)
            return rc, run_time
        else:
            return -1, 0

    # run the model with addr/len pair array
    def run_model_multi_phys(self, model, arr_ifm_len, arr_buff_len_ifm, arr_ofm_len, arr_buff_len_ofm):
        if model and arr_buff_len_ifm and arr_buff_len_ofm and arr_ifm_len and arr_ofm_len:
            rc, run_time = self.conn.root.run_model_multi_phys(model, arr_ifm_len, arr_buff_len_ifm, arr_ofm_len, arr_buff_len_ofm)
            return rc, run_time
        else:
            return -1, 0

    # free model
    def free_model(self, model):
        if model:
            self.conn.root.free_model(model)
        return

    # create virtual env
    def create_virtual_env(self, virtual_env_name):
        rc, result = self.conn.root.create_virtual_env(virtual_env_name)
        return rc, result

    # create virtual env
    def delete_virtual_env(self, virtual_env_name):
        rc = self.conn.root.delete_virtual_env(virtual_env_name)
        return rc

    # run remote model script
    def run_remote_model(self, full_command, model_file_path, timeout, virtual_env_name):
        rc, output, error = self.conn.root.run_remote_model(full_command, model_file_path, timeout, virtual_env_name)
        return rc, output, error

    # a65/mla pipeline related functions
    def pm_run_pipeline(self, ifm):
        #print(f"ifm = {ifm}")
        ofm_raw = self.conn.root.pm_run_pipeline(ifm)
        #print(f"ofm_raw = {ofm_raw}")
        if ofm_raw is not None:
            # by default rpyc does not do deep copy of the np array
            # use obtain call to get the entire np array
            ofm_np_array = rpyc.classic.obtain(ofm_raw)
            return ofm_np_array
        else:
            return None

    def pm_free_model(self):
        return self.conn.root.pm_free_model()
    
    def pm_free_frames(self):
        return self.conn.root.pm_free_frames()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Set Verbose", action="store_true", default=False)
    parser.add_argument("-d", "--debug", help="Set Debug")
    parser.add_argument("--host", help="Hostname (default localhost)", default=None)
    parser.add_argument("--port", help="Port (default 1000)", default=None)
    parser.add_argument("--keyfile", help="SSL Server Key file", default=None)
    parser.add_argument("--certfile", help="SSL Server Cert file", default=None)
    return parser.parse_args()

'''
if __name__ == '__main__':
    options = parse_arguments()
    port = 8000
    if options.port:
        port = options.port

    ### Testing for run_model_multi_phys with llama
    rpc_client = Mla_rpc_client("dm7.sjc.sima.ai", port, None)
    rpc_client.connect()
    rpc_client.dv_connect()

    ifm_file_contents_file_name = [ "/Users/lam.nguyen/llama_7b/ifm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ifm.b0.bin", \
                            "/Users/lam.nguyen/llama_7b/ifm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ifm.persistent.cur_key.b0.bin", \
                            "/Users/lam.nguyen/llama_7b/ifm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ifm.persistent.cur_value.b0.bin", \
                            "/Users/lam.nguyen/llama_7b/ifm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ifm.persistent.prev_key.b0.bin", \
                            "/Users/lam.nguyen/llama_7b/ifm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ifm.persistent.prev_value.b0.bin" ]

    ### Allocate the IFM 
    array_ifm_len = []
    for ifm_file in ifm_file_contents_file_name:
        array_ifm_len.append(os.path.getsize(ifm_file))

    # Allocate the ARRAY of IFM with different size
    rc, arr_mem_buff_ifm = rpc_client.allocate_array_ifm_with_diff_size(array_ifm_len, len(array_ifm_len))
    if rc:
        print("Get arr_mem_buff_ifm")

    # Create the ifm_buf_len_arr
    ifm_buf_len_arr = []
    i_num = 0
    for ifm_mem in range(0, len(array_ifm_len)):
        ifm_buf_len_arr.append(DADDR_LEN(arr_mem_buff_ifm[ifm_mem].buffer_phys_addr, array_ifm_len[ifm_mem]))
        #print(f"{ifm_mem} : {arr_mem_buff_ifm[ifm_mem].buffer_phys_addr}, {array_ifm_len[ifm_mem]}\n")
        ifm_mem += 1

    ### Allocate the OFM
    ofm_file_contents_file_name = [ "/Users/lam.nguyen/llama_7b/ofm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ofm.b0.bin", \
                                   "/Users/lam.nguyen/llama_7b/ofm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ofm.persistent.value.b0.bin", \
                                   "/Users/lam.nguyen/llama_7b/ofm/llama-7b_transformer_block_cache_token_101_int8_stage1_mla_ofm.persistent.key.b0.bin" ]
    array_ofm_len = []
    for ofm_file in ofm_file_contents_file_name:
        array_ofm_len.append(os.path.getsize(ofm_file))
    
    # Allocate the ARRAY of OFM with different size
    rc, arr_mem_buff_ofm = rpc_client.allocate_array_ofm_with_diff_size(array_ofm_len, len(array_ofm_len))
    if rc:
        print("Get arr_mem_buff_ofm")
    
    # Create the ofm_buf_len_arr
    ofm_buf_len_arr = []
    for ofm_mem in range(0, len(array_ofm_len)):
        ifm_buf_len_arr.append(DADDR_LEN(arr_mem_buff_ofm[ofm_mem].buffer_phys_addr, array_ofm_len[ofm_mem]))
        #print(f"{ofm_mem} : {arr_mem_buff_ofm[ofm_mem].buffer_phys_addr}, {array_ofm_len[ofm_mem]}\n")
        ofm_mem += 1

    # UPLOAD IFM
    for i in range(0, len(ifm_file_contents_file_name)):
        with open(ifm_file_contents_file_name[i], mode='rb') as ifm_file:
            ifm_content = ifm_file.read()
            rc = rpc_client.upload_ifm(arr_mem_buff_ifm[i], ifm_content)
        i = i + 1

    ### Load model
    model = rpc_client.load_model_only("/home/sima/llama-7b_transformer_block_cache_token_101_int8_stage1_mla.lm")

    ### Run mla_run_model_multi_phys
    rc, run_time = rpc_client.run_model_multi_phys(model, len(ifm_buf_len_arr), ifm_buf_len_arr, len(ofm_buf_len_arr), ofm_buf_len_arr)
    if rc:
        print("Get OFM")
    
    ofm_b0_rc = rpc_client.download_ofm_only(arr_mem_buff_ofm[0], array_ofm_len[0])
    ofm_persistent_value_b0_rc = rpc_client.download_ofm_only(arr_mem_buff_ofm[1], array_ofm_len[1])
    ofm_persistent_key_b0_rc = rpc_client.download_ofm_only(arr_mem_buff_ofm[2], array_ofm_len[2])

    #with open("/Users/lam.nguyen/ofm_b0_rc.bin", "wb") as file:
        # Example binary data
    #    file.write(ofm_b0_rc)
    
    #with open("/Users/lam.nguyen/ofm_persistent_value_b0_rc.bin", "wb") as file:
        # Example binary data
    #    file.write(ofm_persistent_value_b0_rc)

    #with open("/Users/lam.nguyen/ofm_persistent_key_b0_rc.bin", "wb") as file:
        # Example binary data
    #    file.write(ofm_persistent_key_b0_rc)
    #import ipdb; ipdb.set_trace(context=30)
    rpc_client.free_array_ifm(arr_mem_buff_ifm)
    rpc_client.free_array_ofm(arr_mem_buff_ofm)
    rpc_client.free_model(model)
'''
