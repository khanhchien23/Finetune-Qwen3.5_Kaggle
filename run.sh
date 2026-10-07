#!/bin/bash
# ===== CHẠY FILE NÀY LÀ ĐỦ - mọi thứ còn lại tự động =====
# Dữ liệu: Kaggle Dataset khanhchien/anh-mo-phong-1 (tải về máy, không cần Google Drive/FUSE)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_DATASET="${KAGGLE_DATASET:-khanhchien/anh-mo-phong-1}"   # đổi sang dataset zip mới khi bạn upload lại
KAGGLE_KERNEL="${KAGGLE_KERNEL:-}"   # nếu đặt (vd. khanhchien/zip-anh-mo-phong): tải OUTPUT của notebook (các file .zip) thay vì tải dataset
HF_DATA_REPO="${HF_DATA_REPO:-}"   # nếu đặt (vd. KhanhChien/anh-mo-phong-zips): tải dữ liệu từ Hugging Face dataset (thường nhanh nhất)
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
if [ -z "$HF_DATA_REPO" ] && [ -z "$KAGGLE_API_TOKEN" ] && [ -z "$KAGGLE_KEY" ] \
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
count_files() { find "$DATA_DIR" -type f \( -name '*.zip' -o -name '*.png' -o -name '*.json' \) ! -name '_dataset.zip' ! -path '*/.cache/*' | wc -l; }
has_data()    { [ "$(count_files)" -gt 0 ]; }

# Nguồn dữ liệu hiện tại. File đánh dấu .downloaded ghi lại nguồn đã tải; nếu nguồn đổi
# (hoặc file đánh dấu là bản cũ không có nội dung) thì dữ liệu cũ không đáng tin -> xóa và tải lại.
SOURCE_ID="hf=$HF_DATA_REPO|kernel=$KAGGLE_KERNEL|dataset=$KAGGLE_DATASET"
if [ -f "$DATA_DIR/.downloaded" ] && [ "$(cat "$DATA_DIR/.downloaded" 2>/dev/null)" != "$SOURCE_ID" ]; then
    echo ">> Dữ liệu cũ trong $DATA_DIR không khớp nguồn hiện tại ($SOURCE_ID) -> xóa và tải lại."
    find "$DATA_DIR" -mindepth 1 -delete
fi

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
        kaggle datasets download -d "$KAGGLE_DATASET" -f "$f" -p "$d" -q > "$d/.err_$b" 2>&1 || true
        if [ -s "$d/$b" ] || find "$d" -name "$b" -type f -size +0 2>/dev/null | grep -q .; then
            rm -f "$d/.err_$b"
            return 0
        fi
        sleep $((i * 10))                      # lùi dần: phòng trường hợp bị giới hạn tốc độ
    done
    tail -3 "$d/.err_$b" > "$DATA_DIR/_last_error.txt" 2>/dev/null
    rm -f "$d/.err_$b"
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

# --- Hiển thị tiến độ tải: số file, %, tốc độ, thời gian còn lại ước tính, dung lượng ---
fmt_time() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }
progress_monitor() {
    local total="$1" pid="$2" t0 n0 n now el rate left size
    t0="$(date +%s)"; n0="$(count_files)"
    while kill -0 "$pid" 2>/dev/null; do
        sleep "${PROGRESS_SEC:-30}"
        kill -0 "$pid" 2>/dev/null || break
        n="$(count_files)"; now="$(date +%s)"; el=$(( now - t0 )); size="$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)"
        if [ "$n" -gt "$n0" ] && [ "$el" -gt 0 ]; then
            rate="$(awk -v a="$n" -v b="$n0" -v t="$el" 'BEGIN{printf "%.1f", (a-b)/t}')"
            left="$(awk -v a="$n" -v T="$total" -v r="$rate" 'BEGIN{ if (r>0) printf "%d", (T-a)/r; else print 0 }')"
            echo ">> Tiến độ: $n/$total ($(( 100 * n / total ))%) | $rate file/s | còn ~$(fmt_time "$left") | $size"
        else
            echo ">> Tiến độ: $n/$total ($(( 100 * n / total ))%) | chưa có file mới sau ${el}s | $size"
        fi
    done
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
    echo ">> Dataset có $total file."
    if [ "$total" -gt 3000 ]; then
        echo "!! CẢNH BÁO: $total file lẻ là quá nhiều để tải từng file qua API (dễ bị giới hạn tốc độ, rất lâu)."
        echo "   Khuyến nghị: gom thành vài file .zip rồi upload lại Kaggle (xem make_zips.py)."
    fi
    echo ">> Tải song song ${DL_JOBS:-4} luồng (có thể mất khá lâu)..."
    for round in 1 2 3; do
        xargs -a "$list" -P "${DL_JOBS:-4}" -I{} bash -c 'dl_one "$1"' _ {} &
        local xpid=$!
        progress_monitor "$total" "$xpid" &
        local mpid=$!
        wait "$xpid" || true
        kill "$mpid" 2>/dev/null; wait "$mpid" 2>/dev/null || true
        missing="$(count_missing "$list")"
        echo ">> Vòng $round: còn thiếu $missing / $total file."
        [ -s "$DATA_DIR/_last_error.txt" ] && { echo ">> Lỗi gần nhất từ Kaggle:"; cat "$DATA_DIR/_last_error.txt"; }
        [ "$missing" -eq 0 ] && return 0
    done
    return 1
}

# ---- Đường tải qua Hugging Face dataset: tải song song nhiều đoạn (hf_xet), thường nhanh hơn Kaggle ----
if [ -n "$HF_DATA_REPO" ] && [ ! -f "$DATA_DIR/.downloaded" ]; then
    echo ">> Tải dữ liệu từ Hugging Face dataset $HF_DATA_REPO về $DATA_DIR ..."
    python -c "import hf_xet" 2>/dev/null || uv pip install -q hf_xet || echo "!! Không cài được hf_xet, sẽ tải bằng cách thường (chậm hơn)."
    export HF_XET_HIGH_PERFORMANCE=1
    hf download "$HF_DATA_REPO" --repo-type dataset --local-dir "$DATA_DIR" || true
    if ! has_data; then
        echo "!! Hugging Face dataset $HF_DATA_REPO không có file dữ liệu (.zip/.png/.json). Các file đã tải về:"
        find "$DATA_DIR" -type f ! -path '*/.cache/*' | head -10
        echo "   Mở https://huggingface.co/datasets/$HF_DATA_REPO/tree/main kiểm tra repo đã có các file zip chưa."
        echo "   Nếu chưa: chạy lại ô upload trong notebook Kaggle (cần Internet bật và secret HF_TOKEN)."
        echo "   Nếu có rồi: kiểm tra HF_TOKEN có quyền đọc repo này và tên repo đúng (TenTaiKhoan/ten-repo)."
        exit 1
    fi
    echo "$SOURCE_ID" > "$DATA_DIR/.downloaded"
fi

# ---- Đường tải qua notebook Kaggle: chỉ vài file .zip lớn, nhanh và không bị giới hạn như tải hàng nghìn file lẻ ----
if [ -n "$KAGGLE_KERNEL" ] && [ ! -f "$DATA_DIR/.downloaded" ]; then
    echo ">> Tải output của notebook $KAGGLE_KERNEL về $DATA_DIR ..."
    kaggle kernels output "$KAGGLE_KERNEL" -p "$DATA_DIR" || true
    if ! has_data; then
        echo "!! Không tải được output của notebook $KAGGLE_KERNEL."
        echo "   Kiểm tra: tên notebook đúng chưa (khanhchien/<slug trên URL>), đã Save Version -> Save & Run All xong chưa."
        exit 1
    fi
    echo "$SOURCE_ID" > "$DATA_DIR/.downloaded"
fi

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
    echo "$SOURCE_ID" > "$DATA_DIR/.downloaded"
else
    echo ">> Dữ liệu đã tải từ trước ($DATA_DIR), bỏ qua."
fi

# Nếu nhãn còn nằm trong file .zip lồng bên trong dataset thì giải nén ra
if [ -z "$(find "$DATA_DIR" -name '*.frame_data.json' -print -quit)" ]; then
    echo ">> Chưa thấy *.frame_data.json, thử giải nén các file .zip trong dataset..."
    find "$DATA_DIR" -name '*.zip' | while read -r z; do
        echo "   giải nén $z"
        python -m zipfile -e "$z" "$(dirname "$z")" && rm -f "$z"
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
