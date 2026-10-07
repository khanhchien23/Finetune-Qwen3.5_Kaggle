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
# Đếm file dữ liệu thật trong DATA_DIR (không tính file đánh dấu / file tạm của script)
count_files() { find "$DATA_DIR" -type f ! -name '.downloaded' ! -name '_dataset.zip' ! -name '_filelist.txt' | wc -l; }
has_data()    { [ "$(count_files)" -gt 0 ]; }

# Tự chữa: file đánh dấu .downloaded có nhưng thư mục rỗng (do lần tải lỗi trước đó) -> xóa để tải lại
if [ -f "$DATA_DIR/.downloaded" ] && ! has_data; then
    echo ">> Có file đánh dấu .downloaded nhưng $DATA_DIR rỗng -> tải lại."
    rm -f "$DATA_DIR/.downloaded"
fi

# --- Liệt kê TOÀN BỘ file của dataset (có nhiều trang) vào file $1 ---
list_all_files() {
    local out token="" prev="" ps="--page-size 200"
    : > "$1"
    while :; do
        if [ -n "$token" ]; then
            out="$(kaggle datasets files "$KAGGLE_DATASET" $ps --page-token "$token" -v 2>/dev/null)" || out=""
        else
            out="$(kaggle datasets files "$KAGGLE_DATASET" $ps -v 2>/dev/null)" || out=""
        fi
        if [ -z "$out" ] && [ -n "$ps" ]; then ps=""; continue; fi      # page-size 200 bị từ chối -> dùng mặc định
        [ -z "$out" ] && break
        token="$(printf '%s\n' "$out" | sed -n 's/^Next Page Token = //p' | head -1)"
        printf '%s\n' "$out" | grep -vE '^(Next Page Token|name[ ,]|-{3,}|Warning|$)' \
            | awk -F'[, ]+' '{print $1}' >> "$1"
        { [ -z "$token" ] || [ "$token" = "$prev" ]; } && break
        prev="$token"
    done
    sort -u -o "$1" "$1"
    [ -s "$1" ]
}

# --- Tải MỘT file (có thử lại 3 lần, bỏ qua nếu đã có) ---
dl_one() {
    local f="$1" d b i
    d="$DATA_DIR/$(dirname "$f")"; b="$(basename "$f")"
    [ -s "$d/$b" ] && return 0
    mkdir -p "$d"
    for i in 1 2 3; do
        kaggle datasets download -d "$KAGGLE_DATASET" -f "$f" -p "$d" -q >/dev/null 2>&1 || true
        if [ -s "$d/$b" ] || find "$d" -name "$b" -type f -size +0 2>/dev/null | grep -q .; then
            return 0
        fi
        sleep 2
    done
    echo "FAIL $f" >&2
    return 1
}
export -f dl_one
export DATA_DIR KAGGLE_DATASET

# Số file trong danh sách mà máy chưa có
count_missing() {
    local f n=0
    while IFS= read -r f; do
        [ -s "$DATA_DIR/$f" ] || n=$((n + 1))
    done < "$1"
    echo "$n"
}

download_dataset() {
    local owner="${KAGGLE_DATASET%%/*}" slug="${KAGGLE_DATASET##*/}"
    local zip="$DATA_DIR/_dataset.zip"
    local url="${KAGGLE_DL_URL:-https://www.kaggle.com/api/v1/datasets/download/$owner/$slug}"
    local before after

    # ---- Cách 1: kaggle CLI tải cả dataset ----
    echo ">> [Cách 1] kaggle CLI: tải cả dataset..."
    before="$(count_files)"
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR" --unzip || true
    after="$(count_files)"
    if [ "$after" -gt "$before" ]; then
        return 0
    fi
    echo ">> Cách 1 không tải được file nào (Kaggle CLI có thể in lỗi nhưng vẫn thoát mã 0)."

    # ---- Cách 2: curl gọi thẳng REST API ----
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
    echo ">> Cách 2 cũng không tải được."

    # ---- Cách 3: tải TỪNG FILE theo danh sách, chạy song song (không cần file nén của cả dataset) ----
    echo ">> [Cách 3] tải từng file theo danh sách..."
    local list="$DATA_DIR/_filelist.txt" total missing round
    if ! list_all_files "$list"; then
        echo "!! Không lấy được danh sách file của dataset."
        return 1
    fi
    total="$(wc -l < "$list")"
    echo ">> Dataset có $total file. Tải song song ${DL_JOBS:-6} luồng (có thể mất khá lâu)..."
    for round in 1 2 3; do
        xargs -a "$list" -P "${DL_JOBS:-6}" -I{} bash -c 'dl_one "$1"' _ {} || true
        missing="$(count_missing "$list")"
        echo ">> Vòng $round: còn thiếu $missing / $total file."
        [ "$missing" -eq 0 ] && return 0
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
        echo "!! Cả 3 cách tải đều thất bại. Mở trang dataset trên Kaggle, tab Data, kiểm tra:"
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
