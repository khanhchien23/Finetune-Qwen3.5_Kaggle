#!/usr/bin/env python
# coding: utf-8

# # Qwen3.5 (4B) Vision GRPO — phân loại ảnh mô phỏng
# 
# Giữ nguyên khung notebook Unsloth gốc (LoRA 16-bit, GSPO, `formatting_reward_func`, `correctness_reward_func`), chỉ thay **dữ liệu** và **prompt**.
# 
# **Tối ưu hóa so với bản gốc:**
# 1. **Mapping dữ liệu**: dùng `batched=True` + `num_proc` → nhanh hơn 6–30 lần
# 2. **Warmup**: dùng `warmup_steps=50` (cố định) thay vì `warmup_ratio=0.1` — với dataset 13k ảnh, ratio 0.1 sẽ lãng phí ~600 bước
# 3. **Tham số huấn luyện**: tinh chỉnh cho NVIDIA L40S (48GB VRAM) — `TRAIN_BATCH=96, GRAD_ACC=1, NUM_GEN=4`
# 
# **Bài toán:** mỗi ảnh có đúng 1 đối tượng; model phải trả về
# - `<REASONING>...</REASONING>`: lập luận
# - `<SOLUTION>tên_lớp</SOLUTION>`: tên lớp, ví dụ `Air.KC135`

# In[1]:


import os, glob, json, time
_T0 = time.time()
def _tick(msg):
    print(f"[timing] {msg}: {time.time() - _T0:.0f}s tổng kể từ lúc bắt đầu", flush=True)

# Dữ liệu được run.sh tải từ Kaggle Dataset về máy (biến môi trường DATA_DIR, mặc định ~/kaggle_data).
DATA_DIR  = os.path.expanduser(os.environ.get("DATA_DIR", "~/kaggle_data"))
LABEL_DIR = DATA_DIR
IMG_DIR   = DATA_DIR

label_files = sorted(glob.glob(f"{LABEL_DIR}/**/*.frame_data.json", recursive=True))
img_index = {os.path.basename(p): p for p in glob.glob(f"{IMG_DIR}/**/*.png", recursive=True)}
print("label files:", len(label_files), "| images found:", len(img_index))
assert label_files, f"Không thấy file nhãn *.frame_data.json trong {LABEL_DIR}"
assert img_index,  f"Không thấy ảnh *.png trong {IMG_DIR}"

# ### Unsloth

# In[2]:


import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"   # tắt torch.compile của Unsloth, tránh lỗi FX/dynamo trên đa-GPU

# In[3]:


from unsloth import FastVisionModel
import torch
max_seq_length = 16384 # Must be this long for VLMs
lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-4B",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = False, # Enable vllm fast inference
)

# In[4]:


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
)

# ### Data Prep
# Mỗi file nhãn có đúng 1 đối tượng; ta lấy `labelName` của nó làm đáp án. Nhãn được dùng **nguyên văn** (có cả `GRD.ZIL131` viết hoa `GRD`, khác các lớp `Grd.*`).

# In[5]:


_tick("đã tải xong model")
records, missing = [], 0
for p in label_files:
    cap = json.load(open(p))["captures"][0]
    fn = cap["filename"]
    if fn not in img_index:
        missing += 1
        continue
    labels = {v["labelName"] for a in cap["annotations"] if "BoundingBox2D" in a["@type"] for v in a["values"]}
    assert len(labels) == 1, f"{p}: có {len(labels)} lớp (code này giả định 1 lớp/ảnh)"
    records.append({"image": img_index[fn], "answer": labels.pop()})
print(f"usable samples: {len(records)} | skipped (no image): {missing}")

CLASS_NAMES = sorted({r["answer"] for r in records})
print("num classes:", len(CLASS_NAMES))
_tick("đã đọc nhãn, tạo records")

# Tách một phần nhỏ **giữ lại để đánh giá** (2 ảnh mỗi lớp), không dùng để train.

# In[6]:


import random
from collections import defaultdict
from datasets import Dataset, Image as HFImage

HOLDOUT_PER_CLASS = 2
rng = random.Random(3407)
by_cls = defaultdict(list)
for r in records: by_cls[r["answer"]].append(r)

train_records, eval_records = [], []
for c, items in by_cls.items():
    rng.shuffle(items)
    eval_records += items[:HOLDOUT_PER_CLASS]
    train_records += items[HOLDOUT_PER_CLASS:]
rng.shuffle(train_records); rng.shuffle(eval_records)
print("train:", len(train_records), "| eval:", len(eval_records))

train_dataset = Dataset.from_list(train_records).cast_column("image", HFImage())
eval_dataset  = Dataset.from_list(eval_records).cast_column("image", HFImage())

# #### ⚡ TỐI ƯU HÓA: Resize ảnh theo lô + đa tiến trình

# In[7]:


# ===== TỐI ƯU HÓA: batched + multiprocessing =====
NUM_PROC = int(os.environ.get("DS_NUM_PROC", min(os.cpu_count() or 1, 8)))
print("map resize với num_proc =", NUM_PROC)

def resize_and_rgb_batched(examples):
    """Xử lý theo lô: resize 512x512 + convert RGB cho nhiều ảnh cùng lúc."""
    images = [img.resize((512, 512)) for img in examples["image"]]
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
    return {"image": images}

train_dataset = train_dataset.map(
    resize_and_rgb_batched,
    batched = True,
    batch_size = 32,
    num_proc = NUM_PROC,
    desc = "Resize train images",
)
eval_dataset = eval_dataset.map(
    resize_and_rgb_batched,
    batched = True,
    batch_size = 32,
    num_proc = NUM_PROC,
    desc = "Resize eval images",
)
_tick("xong bước resize ảnh (batched + multiprocessing)")

# We then create the conversational template that is needed to collate the dataset for RL:

# In[8]:


# Define the delimiter variables for clarity and easy modification
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

QUESTION = (
    "Which vehicle is shown in the image? "
    f"Choose the class from this list: {', '.join(CLASS_NAMES)}"
)

def make_conversation(example):
    text_content = (
    f"{QUESTION}. "
    f"Respond with EXACTLY two tags and nothing else - no other text, no <think> block, no extra commentary. "
    f"First a SHORT reasoning (at most 3 short sentences) between {REASONING_START} and {REASONING_END}, "
    f"then your final answer between {SOLUTION_START} and (exactly one class name from the list, nothing else) {SOLUTION_END}. "
    f"Do not restate the list. Stop right after {SOLUTION_END}."
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": text_content},
            ],
        },
    ]
    return {"prompt": prompt}   # 'image' và 'answer' giữ nguyên trong dataset

# Now let's apply the chat template across the entire dataset:

# In[9]:


def apply_template(example):
    return {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize = False,
            add_generation_prompt = True, # Must add assistant
            enable_thinking = False,
        )
    }
# Prompt không phụ thuộc từng mẫu -> tạo MỘT lần rồi thêm thành cột, thay cho 2 lần .map
_prompt_text = apply_template({"prompt": make_conversation({})["prompt"]})["prompt"]
train_dataset = train_dataset.add_column("prompt", [_prompt_text] * len(train_dataset))
eval_dataset  = eval_dataset.add_column("prompt", [_prompt_text] * len(eval_dataset))
_tick("xong bước thêm prompt, sẵn sàng train")

# Kiểm tra độ dài prompt (gồm cả token ảnh) phải < max_prompt_length ở cấu hình train bên dưới
_ex = train_dataset[0]
_n = tokenizer(_ex["image"], _ex["prompt"], add_special_tokens = False, return_tensors = "pt")["input_ids"].shape[1]
print("prompt tokens:", _n, "| answer:", _ex["answer"])

# ## Reward functions

# In[10]:


# Reward functions
import re

def formatting_reward_func(completions,**kwargs):
    import re
    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    scores = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0

        # Fix up addCriterion issues
        if len(completion) != 0:
            removal = completion.replace("addCriterion", "").replace("\n", "")
            if (len(completion)-len(removal))/len(completion) >= 0.5:
                score -= 2.0

        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    completions = [(c[0]["content"] if c else "") if isinstance(c, list) else c for c in completions]
    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    q = prompts[0]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")
    return [
        2.0 if len(r)==1 and a == r[0].replace('\n','') else 0.0
        for r, a in zip(responses, answer)
    ]

# ### Train the model
# 
# #### ⚡ Tham số tối ưu cho NVIDIA L40S (48GB VRAM) + NUM_GEN=4

# In[11]:


import os
os.environ["WANDB_PROJECT"] = "Finetune-Qwen3.5"  # <-- SỬA đúng tên project wandb của bạn

from trl import GRPOConfig, GRPOTrainer

# ---- Tham số tối ưu cho L40S (48GB VRAM) + NUM_GEN=4 ----
# Mục tiêu: dùng ~40-45GB VRAM. Nếu OOM, giảm TRAIN_BATCH xuống 64.
TRAIN_BATCH = int(os.environ.get("TRAIN_BATCH", 96))
NUM_GEN     = int(os.environ.get("NUM_GEN", 4))
GRAD_ACC    = int(os.environ.get("GRAD_ACC", 1))
EPOCHS      = float(os.environ.get("EPOCHS", 1))
MAX_STEPS   = int(os.environ.get("MAX_STEPS", -1))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 50))

assert (TRAIN_BATCH * GRAD_ACC) % NUM_GEN == 0, "TRAIN_BATCH * GRAD_ACC phải chia hết cho NUM_GEN"
_imgs_per_step = TRAIN_BATCH * GRAD_ACC // NUM_GEN
_est_steps = MAX_STEPS if MAX_STEPS > 0 else int(-(-len(train_dataset) * EPOCHS // _imgs_per_step))
print(f"[config] L40S-optimized | model={os.environ.get('MODEL_NAME', 'unsloth/Qwen3.5-4B')} | "
      f"{_imgs_per_step} ảnh/bước, {NUM_GEN} completion/ảnh | {len(train_dataset)} ảnh train | "
      f"epochs={EPOCHS} | ~{_est_steps} bước | warmup={WARMUP_STEPS} bước")

training_args = GRPOConfig(
    # ===== Learning rate & optimizer =====
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_steps = WARMUP_STEPS,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    max_grad_norm = 0.1,

    # ===== Batch & Generation =====
    per_device_train_batch_size = TRAIN_BATCH,   # 96 (L40S 48GB)
    gradient_accumulation_steps = GRAD_ACC,
    num_generations = NUM_GEN,                   # 4 (theo yêu cầu)
    max_prompt_length = 2048,
    max_completion_length = 512,                 # Tăng từ 384 -> 512 để dùng thêm VRAM

    # ===== Training schedule =====
    num_train_epochs = EPOCHS,
    max_steps = MAX_STEPS,
    logging_steps = 1,

    # ===== Checkpointing =====
    save_strategy = "steps",
    save_steps = 50,
    save_total_limit = 3,

    # ===== Logging & Hub =====
    report_to = "wandb",
    run_name = "qwen3.5-vehicle-grpo-l40s",
    push_to_hub = True,
    hub_model_id = "KhanhChien/qwen3.5-vehicle-lora",  # <-- SỬA đúng username HF của bạn
    hub_strategy = "checkpoint",
    hub_private_repo = True,
    output_dir = "outputs",

    # ===== GSPO =====
    importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = "dr_grpo",
)

# In[12]:


trainer = GRPOTrainer(
    model = model,
    args = training_args,
    processing_class = tokenizer,
    reward_funcs = [
        formatting_reward_func,
        correctness_reward_func,
    ],
    train_dataset = train_dataset,
)

trainer.train()

# <a name="Save"></a>
# ### Saving, loading finetuned models

# In[13]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving