#!/bin/bash
# ===== CHẠY FILE NÀY LÀ ĐỦ - mọi thứ còn lại tự động =====
# Dữ liệu: Kaggle Dataset khanhchien/anh-mo-phong-1 (tải về máy, không cần Google Drive/FUSE)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_DATASET="khanhchien/anh-mo-phong-1"
DATA_DIR=~/kaggle_data            # nơi chứa dữ liệu sau khi tải + giải nén
SOURCE_DIR=~/source_code
ENV_NAME="qwen_env"
RUN_DIR=~/train_run               # train.py ghi outputs/ và qwen_lora/ vào đây


# 1. Gỡ bỏ phiên bản cũ của cả hai thư viện
pip uninstall -y kagglehub kagglesdk

# 2. Cài đặt cặp phiên bản tương thích đã biết
pip install kagglehub==1.0.2 kagglesdk==0.1.16
# ---------------------------------------------------------------------
# 1) Pull code + cài thư viện (script con tự dò/cài conda, tạo + activate env)
# ---------------------------------------------------------------------
source "$SCRIPT_DIR/pull_and_install.sh"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Kaggle CLI: chạy bản MỚI NHẤT trong một Python 3.12 tách biệt bằng uv (uv đã được cài ở pull_and_install.sh).
# Lý do: Python 3.10 của qwen_env chỉ cài được kaggle 1.7.x, bản này không hiểu token mới (KGAT_...)
# và báo "KeyError: 'username'". Không đụng gì tới môi trường train.
# kaggle_cli() { uv tool run --python 3.12 kaggle "$@"; }
# ---------------------------------------------------------------------
# 2) Kiểm tra đăng nhập Kaggle / HF / wandb (thiếu cái nào dừng ngay, khỏi tải xong mới lỗi)
# ---------------------------------------------------------------------
if [ -z "$KAGGLE_API_TOKEN" ] && [ -z "$KAGGLE_KEY" ] \
   && [ ! -f ~/.kaggle/access_token ] && [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "!! CẢNH BÁO: chưa có Kaggle API token."
    echo "   Cách nhanh: export KAGGLE_API_TOKEN='<token>'   (hoặc đặt file ~/.kaggle/kaggle.json)"
    exit 1
fi
if [ -z "$HF_TOKEN" ] && ! hf auth whoami &>/dev/null; then
    echo "!! CẢNH BÁO: chưa đăng nhập Hugging Face. Chạy: export HF_TOKEN='<token>'  hoặc  hf auth login"
    exit 1
fi
if [ -z "$WANDB_API_KEY" ] && { [ ! -f ~/.netrc ] || ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; }; then
    echo "!! CẢNH BÁO: chưa đăng nhập wandb. Chạy: export WANDB_API_KEY='<key>'  hoặc  wandb login"
    exit 1
fi

# ---------------------------------------------------------------------
# 3) Tải dataset từ Kaggle (bỏ qua nếu đã tải xong từ lần trước)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# 3) Tải dataset từ Kaggle (dùng kagglehub, bỏ qua nếu đã tải xong)
# ---------------------------------------------------------------------
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/.downloaded" ]; then
    echo ">> Tải dataset $KAGGLE_DATASET bằng kagglehub..."
    pip install -q kagglehub
    python -c "
import kagglehub, os, shutil, sys
try:
    cache_path = kagglehub.dataset_download('${KAGGLE_DATASET}')
    print('>> Cache:', cache_path)
    data_dir = os.path.expanduser('${DATA_DIR}')
    for item in os.listdir(cache_path):
        s, d = os.path.join(cache_path, item), os.path.join(data_dir, item)
        shutil.copytree(s, d, dirs_exist_ok=True) if os.path.isdir(s) else shutil.copy2(s, d)
    print('>> Đã copy sang:', data_dir)
except Exception as e:
    print('!! Lỗi kagglehub:', e, file=sys.stderr); sys.exit(1)
"
    touch "$DATA_DIR/.downloaded"
else
    echo ">> Dữ liệu đã tải từ trước, bỏ qua."
fi

# ---------------------------------------------------------------------
# 4) Sau khi train xong (hoặc lỗi/ngắt): đẩy adapter cuối qwen_lora lên cùng repo HF.
#    Checkpoint trong quá trình train đã tự lên HF nhờ push_to_hub trong train.py.
# ---------------------------------------------------------------------
upload_results() {
    set +e
    HF_REPO=$(grep -oP 'hub_model_id\s*=\s*"\K[^"]+' "$SOURCE_DIR/train.py" | head -1)
    if [ -d "$RUN_DIR/qwen_lora" ] && [ -n "$HF_REPO" ]; then
        echo ">> Upload qwen_lora lên Hugging Face: $HF_REPO ..."
        hf upload "$HF_REPO" "$RUN_DIR/qwen_lora" qwen_lora
    fi
}
trap upload_results EXIT

# ---------------------------------------------------------------------
# 5) Train (cd vào RUN_DIR để outputs/ và qwen_lora/ nằm gọn ở đó)
# ---------------------------------------------------------------------
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
echo ">> Bắt đầu train..."
python "$SOURCE_DIR/train.py"
