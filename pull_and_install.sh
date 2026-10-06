#!/bin/bash
# ===== "script nhỏ: pull source + cài thư viện" trong sơ đồ =====
# Ưu tiên dùng conda/anaconda/miniconda ĐÃ CÓ trên máy. Chỉ cài Miniconda
# mới nếu không tìm thấy conda nào. Không cần sudo/apt-get.
set -e

REPO_URL="https://github.com/khanhchien23/Finetune-Qwen3.5.git"
SOURCE_DIR=~/source_code
ENV_NAME="qwen_env"
PY_VERSION="3.10"
CUDA_VERSION="12.8.0"   # phải khớp bản torch==2.8.0+cu128 cài bên dưới

# ---------------------------------------------------------------------
# Tìm conda đã có sẵn (anaconda / miniconda / conda ở các vị trí phổ biến)
# In ra đường dẫn base nếu tìm thấy, return 1 nếu không có.
# ---------------------------------------------------------------------
find_conda_base() {
    local c
    # 1) conda đang có trong PATH / shell function
    if command -v conda &>/dev/null; then
        c="$(conda info --base 2>/dev/null || true)"
        if [ -n "$c" ] && [ -f "$c/etc/profile.d/conda.sh" ]; then
            echo "$c"; return 0
        fi
    fi
    # 2) các vị trí cài đặt phổ biến
    for c in \
        "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/anaconda" "$HOME/miniconda" \
        /opt/anaconda3 /opt/miniconda3 /opt/conda \
        /usr/local/anaconda3 /usr/local/miniconda3 /usr/local/conda; do
        if [ -f "$c/etc/profile.d/conda.sh" ]; then
            echo "$c"; return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------
# 0) Dùng conda có sẵn; nếu chưa có thì mới cài Miniconda vào $HOME
# ---------------------------------------------------------------------
if CONDA_DIR="$(find_conda_base)"; then
    echo ">> Đã có conda tại: $CONDA_DIR -> bỏ qua bước cài Miniconda."
else
    CONDA_DIR=~/miniconda3
    echo ">> Không tìm thấy conda/anaconda, cài Miniconda vào $CONDA_DIR..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
export CONDA_DIR
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Dùng hẳn kênh conda-forge, KHÔNG dùng kênh "defaults" của Anaconda
# (tránh lỗi CondaToSNonInteractiveError do chưa accept Terms of Service).
# --add có thể báo trùng nếu đã có, nên chỉ thêm khi chưa có.
conda config --show channels 2>/dev/null | grep -q conda-forge || conda config --add channels conda-forge
conda config --set channel_priority strict

# ---------------------------------------------------------------------
# 1) Pull code mới nhất từ GitHub
# ---------------------------------------------------------------------
if [ -d "$SOURCE_DIR/.git" ]; then
    echo ">> Đã có repo, pull bản mới nhất..."
    git -C "$SOURCE_DIR" pull
else
    echo ">> Clone repo lần đầu..."
    git clone "$REPO_URL" "$SOURCE_DIR"
fi

# ---------------------------------------------------------------------
# 2) Tạo + kích hoạt conda environment (chỉ tạo nếu chưa có)
# ---------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo ">> Tạo conda environment lần đầu (Python ${PY_VERSION})..."
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=${PY_VERSION}" pip
fi
conda activate "$ENV_NAME"
echo ">> Đang dùng: $(python --version) (conda env: $ENV_NAME)"

# ---------------------------------------------------------------------
# 3) Cài CUDA Toolkit (gồm nvcc) + thư viện Python - CHỈ chạy nếu chưa cài
#    Marker nằm trong chính env đang active ($CONDA_PREFIX), đúng cả khi
#    conda nằm ở /opt/... và env được tạo ở ~/.conda/envs.
# ---------------------------------------------------------------------
MARKER="$CONDA_PREFIX/.deps_installed"
if [ ! -f "$MARKER" ]; then
    echo ">> Cài CUDA Toolkit ${CUDA_VERSION} qua conda (có nvcc, không cần apt)..."
    conda install -y --override-channels -c "nvidia/label/cuda-${CUDA_VERSION}" -c conda-forge cuda-toolkit
    echo ">> nvcc: $(nvcc --version | tail -1)"

    echo ">> Cài trình biên dịch C/C++ qua conda (causal_conv1d cần gcc/g++ để build)..."
    conda install -y --override-channels -c conda-forge c-compiler cxx-compiler

    echo ">> Cài thư viện Python (sẽ mất vài phút)..."
    pip install --upgrade -qqq pip uv

    uv pip install -qqq \
        "torch==2.8.0" "triton>=3.3.0" numpy pillow torchvision bitsandbytes xformers==0.0.32.post2 \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    uv pip install -qqq --no-deps "torchcodec==0.7.0"
    uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
    uv pip install transformers==5.2.0
    uv pip uninstall -qqq flash-linear-attention fla-core || true
    uv pip install --no-build-isolation causal_conv1d==1.6.0
    uv pip install --no-deps --upgrade "torchao>=0.16.0"

    uv pip install huggingface_hub wandb datasets

    if [ -f "$SOURCE_DIR/requirements.txt" ]; then
        uv pip install -r "$SOURCE_DIR/requirements.txt"
    fi

    touch "$MARKER"
    echo ">> Cài thư viện xong."
else
    echo ">> Thư viện đã cài từ trước, bỏ qua."
fi

echo ">> pull_and_install.sh xong. Code ở: $SOURCE_DIR | conda: $CONDA_DIR | env: $ENV_NAME"