from pathlib import Path

import torch
import onnx
import onnxruntime


def select_device(device='', is_onnx = True):
    # device = 'cpu' or 'gpu'
    assert device.lower() in ['cpu', 'cuda'], f'ERROR: {device} not a legal value'
    cpu = device.lower() == 'cpu'
    if is_onnx:
        providers = ['CPUExecutionProvider']
        if not cpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() and onnxruntime.get_device().lower() == device.lower():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return providers
    else:
        cuda = not cpu and torch.cuda.is_available()
        return torch.device('cuda:0' if cuda else 'cpu')

def check_file(file):
    file = Path(str(file).strip().replace("'", '').lower())
    assert file.exists(), 'File Not Found: %s' % file  # assert file was found


def attempt_load(path, providers=None):
    check_file(path)
    onnx_model = onnx.load(path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX load success, loaded as %s' % path)
    # Finish
    session = onnxruntime.InferenceSession(path, providers=providers)
    return session


def load_onnx_model(path, providers):
    model = attempt_load(path, providers)
    return model
