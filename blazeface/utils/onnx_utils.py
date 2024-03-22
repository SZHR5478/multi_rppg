from pathlib import Path

import onnx
import onnxruntime


def select_device(device=''):
    # device = 'cpu' or 'gpu'
    assert device.lower() in ['cpu', 'gpu'], f'ERROR: {device} not a legal value'
    providers = ['CPUExecutionProvider']
    cpu = device.lower() == 'cpu'
    if not cpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() and onnxruntime.get_device().lower() == device.lower():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return providers


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
