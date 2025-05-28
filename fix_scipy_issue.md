# 修复 scipy/numpy 版本冲突问题

## 问题描述
```
ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)
```

## 解决方案

### 方案1: 重新安装兼容版本
```bash
# 卸载冲突的包
pip uninstall scipy numpy -y

# 安装兼容版本
pip install numpy==1.24.3
pip install scipy==1.10.1
```

### 方案2: 使用最新兼容版本
```bash
# 卸载并重新安装
pip uninstall scipy numpy -y
pip install numpy>=1.21.0,<1.25.0
pip install scipy>=1.9.0,<1.11.0
```

### 方案3: 如果使用conda环境
```bash
conda install numpy=1.24.3 scipy=1.10.1 -c conda-forge
```

### 方案4: 完全重建环境（推荐）
```bash
# 创建新的conda环境
conda create -n ensemble-fix python=3.9
conda activate ensemble-fix

# 安装基础包
conda install numpy=1.24.3 scipy=1.10.1 -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install transformers
pip install llamafactory
pip install fastapi uvicorn
pip install lm-eval
```

## 验证修复
```bash
python -c "import scipy; import numpy; print('Success!')"
```