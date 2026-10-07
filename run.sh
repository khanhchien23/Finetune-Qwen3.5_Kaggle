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
uv tool install --python 3.12 --force kaggle
export PATH="$HOME/.local/bin:$PATH"
hash -r
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
mkdir -p "$DATA_DIR"

# Tải dataset: thử lần lượt nhiều cách, cách nào xong thì dừng.
# Lý do: "kaggle datasets files" chạy được nhưng lệnh tải CẢ dataset (DownloadDataset) có thể trả 404
# (vd. dataset quá lớn / bản dataset chưa xử lý xong), nên cần đường dự phòng gọi thẳng REST API bằng curl.
# Có file dữ liệu thật trong DATA_DIR chưa (không tính file đánh dấu)
has_data() { [ -n "$(find "$DATA_DIR" -type f ! -name '.downloaded' ! -name '_dataset.zip' -print -quit)" ]; }

# Tự chữa: file đánh dấu .downloaded có nhưng thư mục rỗng (do lần tải lỗi trước đó) -> xóa để tải lại
if [ -f "$DATA_DIR/.downloaded" ] && ! has_data; then
    echo ">> Có file đánh dấu .downloaded nhưng $DATA_DIR rỗng -> tải lại."
    rm -f "$DATA_DIR/.downloaded"
fi

download_dataset() {
    local owner="${KAGGLE_DATASET%%/*}" slug="${KAGGLE_DATASET##*/}"
    local zip="$DATA_DIR/_dataset.zip"
    local url="${KAGGLE_DL_URL:-https://www.kaggle.com/api/v1/datasets/download/$owner/$slug}"

    echo ">> [Cách 1] kaggle CLI: tải cả dataset..."
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR" --unzip || true
    if has_data; then
        return 0
    fi
    echo ">> Cách 1 không tải được file nào (Kaggle CLI có thể in lỗi nhưng vẫn thoát mã 0)."

    echo ">> [Cách 2] curl gọi thẳng REST API (tự thử lại, tiếp tục được nếu đứt mạng)..."
    local mode
    for mode in with_token no_token; do
        local auth=()
        if [ "$mode" = with_token ]; then
            [ -n "$KAGGLE_API_TOKEN" ] && auth=(-H "Authorization: Bearer $KAGGLE_API_TOKEN")
        fi
        if curl -fL --retry 5 --retry-delay 5 -C - "${auth[@]}" -o "$zip" "$url" \
           && python -m zipfile -l "$zip" >/dev/null 2>&1; then
            echo ">> Giải nén $zip ..."
            python -m zipfile -e "$zip" "$DATA_DIR" && rm -f "$zip"
            return 0
        fi
        rm -f "$zip"
    done
    return 1
}

if [ ! -f "$DATA_DIR/.downloaded" ]; then
    echo ">> Kiểm tra truy cập dataset $KAGGLE_DATASET ..."
    if ! LISTING="$(kaggle datasets files "$KAGGLE_DATASET" 2>&1)"; then
        echo "$LISTING" | tail -5
        echo "!! Không truy cập được dataset. Kiểm tra: KAGGLE_API_TOKEN đúng chưa, tên dataset đúng chưa, dataset đã Public chưa."
        exit 1
    fi
    echo "$LISTING" | head -15      # xem nhanh dataset có file gì, dung lượng bao nhiêu
    echo ">> Tải dataset $KAGGLE_DATASET về $DATA_DIR (có thể mất một lúc)..."
    if ! download_dataset; then
        echo "!! Cả 2 cách tải đều thất bại. Mở trang dataset trên Kaggle, tab Data, kiểm tra:"
        echo "   (1) có hiển thị danh sách file và không còn trạng thái đang xử lý (processing);"
        echo "   (2) thử bấm Download trên web. Nếu web cũng không tải được, hãy tạo lại version dataset"
        echo "       hoặc chia nhỏ dataset thành vài file .zip (mỗi file vài GB) rồi upload lại."
        exit 1
    fi
    if ! has_data; then
        echo "!! Tải xong nhưng $DATA_DIR vẫn rỗng."
        exit 1
    fi
    touch "$DATA_DIR/.downloaded"
else
    echo ">> Dữ liệu đã tải từ trước ($DATA_DIR), bỏ qua."
fi

# Nếu nhãn còn nằm trong file .zip lồng bên trong dataset thì giải nén ra
if [ -z "$(find "$DATA_DIR" -name '*.frame_data.json' -print -quit)" ]; then
    echo ">> Chưa thấy *.frame_data.json, thử giải nén các file .zip trong dataset..."
    find "$DATA_DIR" -name '*.zip' | while read -r z; do
        python -m zipfile -e "$z" "$(dirname "$z")"
    done
fi

N_LABEL=$(find "$DATA_DIR" -name '*.frame_data.json' | wc -l)
N_IMG=$(find "$DATA_DIR" -name '*.png' | wc -l)
echo ">> Tìm thấy: $N_LABEL file nhãn, $N_IMG ảnh png trong $DATA_DIR"
if [ "$N_LABEL" -eq 0 ] || [ "$N_IMG" -eq 0 ]; then
    echo "!! Không đủ nhãn/ảnh. Cấu trúc thư mục hiện có:"
    find "$DATA_DIR" -maxdepth 3 | head -40
    exit 1
fi
export DATA_DIR

# Nếu train.py trên GitHub CHƯA phải bản mới (còn đọc ~/gdrive_mount) -> tạo symlink tương thích
if ! grep -q 'environ.get("DATA_DIR"' "$SOURCE_DIR/train.py"; then
    echo "!! train.py trên GitHub là bản cũ (đọc ~/gdrive_mount) -> tạo symlink tạm. Nên push bản train.py mới."
    mkdir -p ~/gdrive_mount
    [ -e ~/gdrive_mount/labels ] || ln -s "$DATA_DIR" ~/gdrive_mount/labels
    [ -e ~/gdrive_mount/images ] || ln -s "$DATA_DIR" ~/gdrive_mount/images
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
