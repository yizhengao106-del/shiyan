#!/usr/bin/env bash
# 在服务器运行；只读取环境，不启动训练。
PYTHON=/home/gyz/anaconda3/envs/zoomnext/bin/python
{
 date -Is
 uname -a
 cat /etc/os-release
 nvidia-smi
 "$PYTHON" - <<'PY'
import sys, torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print("GPU",i,p.name,"total_memory_bytes",p.total_memory)
PY
 "$PYTHON" -m pip freeze
} > "server_environment_$(date +%Y%m%d_%H%M%S).txt" 2>&1
