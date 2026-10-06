#!/usr/bin/env python
# coding: utf-8

# # Qwen3.5 (4B) Vision GRPO — phân loại ảnh mô phỏng
# 
# Giữ nguyên khung notebook Unsloth gốc (LoRA 16-bit, GSPO, `formatting_reward_func`, `correctness_reward_func`), chỉ thay **dữ liệu** và **prompt**. Không thêm hàm thưởng nào mới.
# 
# **Bài toán:** mỗi ảnh có đúng 1 đối tượng; model phải trả về
# - `<REASONING>...</REASONING>`: lập luận
# - `<SOLUTION>tên_lớp</SOLUTION>`: tên lớp, ví dụ `Air.KC135`
# 
# **Hàm thưởng (tối đa 4.0 mỗi completion), cả hai giữ nguyên bản gốc:**
# | Hàm | Điểm |
# |---|---|
# | `formatting_reward_func` | 0–2 (−2 nếu lỗi `addCriterion`) |
# | `correctness_reward_func` | 2 nếu tên lớp trong `<SOLUTION>` khớp tuyệt đối với nhãn, ngược lại 0 |

# ### Installation

# (Cài đặt thư viện đã do run.sh lo - xem file run.sh)
# ### Chuẩn bị dữ liệu trên Colab
# Cần 2 thứ:
# 1. `labels.zip` (các file `*.frame_data.json`) — chỉ dùng trường `labelName` làm nhãn lớp
# 2. Thư mục ảnh `anh_mo_phong` (`seq_step*.camera_0.png`, `sim_step*.camera_0.png`)
# 
# Upload lên `/content` (hoặc mount Google Drive rồi sửa 2 đường dẫn bên dưới). Ảnh nào không có trong thư mục sẽ tự bị bỏ qua.

# In[2]:


import os, glob, json

# Dữ liệu được run.sh tải từ Kaggle Dataset về máy (biến môi trường DATA_DIR, mặc định ~/kaggle_data).
# Nhãn (*.frame_data.json) và ảnh (*.png) đều được tìm ĐỆ QUY trong DATA_DIR bên dưới,
# nên không phụ thuộc cấu trúc thư mục bên trong dataset.
DATA_DIR  = os.path.expanduser(os.environ.get("DATA_DIR", "~/kaggle_data"))
LABEL_DIR = DATA_DIR
IMG_DIR   = DATA_DIR

label_files = sorted(glob.glob(f"{LABEL_DIR}/**/*.frame_data.json", recursive=True))
img_index = {os.path.basename(p): p for p in glob.glob(f"{IMG_DIR}/**/*.png", recursive=True)}
print("label files:", len(label_files), "| images found:", len(img_index))
assert label_files, f"Không thấy file nhãn *.frame_data.json trong {LABEL_DIR}"
assert img_index,  f"Không thấy ảnh *.png trong {IMG_DIR}"

# ### Unsloth

# **Sửa lỗi Kaggle:** notebook gốc chỉ kiểm chứng trên 1 GPU T4. Trên Kaggle T4 x2, `torch.compile` của Unsloth va với gradient checkpointing khi có 2 GPU và gây lỗi `RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function` ngay tại `trainer.train()`. Cell dưới tắt `torch.compile` của Unsloth trước khi import, để tránh lỗi này (train sẽ chậm hơn một chút vì không được compile, nhưng ổn định hơn là bị crash giữa chừng). Nếu bạn đổi Accelerator trong Settings về **GPU T4 x1**, có thể không cần cell này nữa — nhưng để an toàn thì mình vẫn giữ nó.

# In[3]:


import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"   # tắt torch.compile của Unsloth, tránh lỗi FX/dynamo trên đa-GPU

# In[4]:


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

# In Unsloth, we share vLLM's weights directly, reducing VRAM usage by > 50%. vLLM also does not yet support LoRA on the vision layers, so we can only add them on the language layers. Vision GRPO still works though!

# In[5]:


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
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# ### Data Prep
# <a name="Data"></a>
# 
# Mỗi file nhãn có đúng 1 đối tượng; ta lấy `labelName` của nó làm đáp án. Nhãn được dùng **nguyên văn** (có cả `GRD.ZIL131` viết hoa `GRD`, khác các lớp `Grd.*`). Vì hàm thưởng so khớp chuỗi tuyệt đối, danh sách lớp trong prompt được lấy thẳng từ nhãn để model chép đúng.

# In[6]:


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

# Tách một phần nhỏ **giữ lại để đánh giá** (2 ảnh mỗi lớp), không dùng để train — để có số đo trước/sau train thật sự.

# In[7]:


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

# We resize the images to be 512 by 512 pixels to make the images manageable in context length, and convert them to RGB (giống notebook gốc).

# In[8]:


# Resize to (512, 512) then convert to RGB
def resize_and_rgb(example):
    image = example["image"].resize((512, 512))
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["image"] = image
    return example

train_dataset = train_dataset.map(resize_and_rgb)
eval_dataset  = eval_dataset.map(resize_and_rgb)

# We then create the conversational template that is needed to collate the dataset for RL:

# In[9]:


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

train_dataset = train_dataset.map(make_conversation)
eval_dataset  = eval_dataset.map(make_conversation)

# Now let's apply the chat template across the entire dataset:

# In[10]:


def apply_template(example):
    return {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize = False,
            add_generation_prompt = True, # Must add assistant
            enable_thinking = False,
        )
    }
train_dataset = train_dataset.map(apply_template)
eval_dataset  = eval_dataset.map(apply_template)

# Kiểm tra độ dài prompt (gồm cả token ảnh) phải < max_prompt_length ở cấu hình train bên dưới
_ex = train_dataset[0]
_n = tokenizer(_ex["image"], _ex["prompt"], add_special_tokens = False, return_tensors = "pt")["input_ids"].shape[1]
print("prompt tokens:", _n, "| answer:", _ex["answer"])

# ## Reward functions
# 
# Hai hàm thưởng dưới đây **giữ nguyên bản gốc**. `correctness_reward_func` vẫn so khớp chuỗi tuyệt đối, nay áp dụng cho tên lớp trong `<SOLUTION>`.
# 
# (Hàm `correctness_reward_func` gốc có `print` prompt đầu tiên mỗi bước; prompt giờ chứa cả danh sách lớp nên log sẽ dài — có thể comment dòng `print` nếu thấy rối.)

# In[11]:


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
        # See https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks
        # Penalize on excessive addCriterion and newlines
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

# <a name="Inference"></a>
# ### Đánh giá trên tập giữ lại (trước khi train)
# Đo tỷ lệ đúng định dạng và tỷ lệ đúng tên lớp (cùng tiêu chí với `correctness_reward_func`). Chạy lại đúng hàm này sau khi train để so sánh.

# In[12]:


# import numpy as np
# from transformers import TextStreamer

# def evaluate(n = 96, seed = 0, max_new_tokens = 512):
#     idx = np.random.RandomState(seed).permutation(len(eval_dataset))[:n]
#     fmt = cls_ok = 0
#     for i in idx:
#         ex = eval_dataset[int(i)]
#         inputs = tokenizer(ex["image"], ex["prompt"], add_special_tokens = False, return_tensors = "pt").to("cuda")
#         with torch.no_grad():
#             out = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True, do_sample = False)
#         text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True)
#         sol = re.findall(f'{SOLUTION_START}(.*?){SOLUTION_END}', text, re.DOTALL)
#         think = re.findall(f'{REASONING_START}(.*?){REASONING_END}', text, re.DOTALL)
#         fmt    += int(len(sol) == 1 and len(think) == 1)
#         cls_ok += int(len(sol) == 1 and sol[0].replace("\n", "") == ex["answer"])
#     n = len(idx)
#     res = {"n": n, "format_ok": fmt/n, "class_acc": cls_ok/n}
#     print(res); return res

# baseline = evaluate(n = 96)

# ### Train the model
# 
# Cấu hình quay về **giá trị gốc** của notebook Unsloth (batch=1, `num_generations=2`, `max_steps=60`), chỉ khác đúng 1 chỗ: `max_prompt_length` 1024 → 2048, vì prompt có thêm danh sách 96 lớp cộng token ảnh.
# 
# Log của notebook gốc trên 1x T4 cho thấy 60 bước mất ~1h20. Mỗi bước chỉ dùng khoảng 1 ảnh, nên 60 bước chỉ chạm khoảng 60/1395 ảnh train — đừng kỳ vọng độ đúng lớp đổi nhiều sau lần chạy này. Muốn train lâu hơn thì tăng `max_steps` (hoặc bật `num_train_epochs = 1` — với 1.395 ảnh train, ước tính sẽ mất rất nhiều giờ, cân nhắc giới hạn 12h/phiên của Kaggle trước khi bật).
# 
# **Checkpoint:** `save_steps = 50` sẽ tự lưu adapter LoRA vào `outputs/checkpoint-50` (và `checkpoint-60` khi kết thúc, vì `max_steps=60`). `save_total_limit = 3` giữ tối đa 3 checkpoint gần nhất để đỡ tốn dung lượng `/kaggle/working` (mỗi checkpoint chỉ khoảng vài chục MB vì chỉ lưu phần LoRA, không phải full 4B tham số). Nếu phiên Kaggle bị ngắt giữa chừng, chạy lại `trainer.train(resume_from_checkpoint = "outputs/checkpoint-50")` (sửa số bước cho đúng checkpoint gần nhất) để train tiếp thay vì phải chạy lại từ đầu.

# ### Train tiếp từ checkpoint
# 
# Checkpoint `checkpoint-480` nằm trong 1 Kaggle Model (`/kaggle/input/...`), tức **chỉ đọc**. `resume_from_checkpoint` chỉ cần **đọc** từ đó (model, optimizer, scheduler, `trainer_state.json`) để khôi phục đúng trạng thái đang train dở — không cần ghi gì vào đường dẫn này. Checkpoint mới sinh ra trong phiên này vẫn được lưu bình thường vào `output_dir = "outputs"` (trên `/kaggle/working`, có quyền ghi).
# 
# **Quan trọng:** checkpoint-480 nghĩa là lần trước đã chạy **đúng 480/480 bước** (`max_steps` cũ = 480). Nếu giữ nguyên `max_steps = 480` ở lần này, Trainer sẽ thấy `global_step` đã bằng `max_steps` và **không train thêm bước nào cả**. Vì vậy phải đặt `RESUME_TOTAL_STEPS` bên dưới **lớn hơn 480** — đây là **tổng số bước tính từ đầu (bước 0)**, không phải số bước train thêm. Ví dụ muốn train thêm 300 bước nữa thì đặt `960` (không phải `300`).
# 
# Lưu ý nhỏ: `lr_scheduler_type = "cosine"` và `warmup_ratio` được tính lại dựa trên tổng số bước mới (`RESUME_TOTAL_STEPS`), nên đường cong learning rate ở các bước tiếp theo sẽ hơi khác so với nếu bạn train một mạch 960 bước ngay từ đầu — ảnh hưởng nhỏ, không đáng ngại.


import os
os.environ["WANDB_PROJECT"] = "Finetune-Qwen3.5"  # <-- SỬA đúng tên project wandb của bạn

# In[13]:


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    log_completions = False,
    per_device_train_batch_size = 4,   # gốc; Unsloth tự nâng lên bằng num_generations
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # gốc; tăng lên 4 nếu muốn tín hiệu advantage bớt nhiễu và đủ VRAM
    max_prompt_length = 2048,
    max_completion_length = 384,
    num_train_epochs = 2, # Set to 1 for a full training run - sẽ rất lâu, xem ghi chú markdown
    # max_steps = RESUME_TOTAL_STEPS,  # PHẢI > 480 (số bước đã chạy trong checkpoint),
    #                                   # nếu không Trainer nghĩ đã train xong và sẽ không chạy thêm
    save_strategy = "steps",
    save_steps = 50,  # lưu checkpoint mỗi 50 bước (chỉ lưu adapter LoRA, không phải full model)
    save_total_limit = 3,  # giữ tối đa 3 checkpoint gần nhất, tránh đầy ổ /kaggle/working
    max_grad_norm = 0.1,
    report_to = "wandb",
    run_name = "qwen3.5-vehicle-grpo",

    push_to_hub = True,
    hub_model_id = "KhanhChien/qwen3.5-vehicle-lora",  # <-- SỬA đúng username HF của bạn
    hub_strategy = "checkpoint",
    hub_private_repo = True,
    output_dir = "outputs",

    # Below enables GSPO:
    importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = "dr_grpo",
)

# And let's run the trainer! Cột `reward` là tổng của 2 hàm thưởng (tối đa 4.0). Theo dõi riêng `rewards / correctness_reward_func / mean`; nó có thể vẫn ≈ 0 trong nhiều bước đầu.

# In[ ]:


trainer = GRPOTrainer(
    model = model,
    args = training_args,
    # Pass the processor to handle multimodal inputs
    processing_class = tokenizer,
    reward_funcs = [
        formatting_reward_func,
        correctness_reward_func,
    ],
    train_dataset = train_dataset,
)

trainer.train()

# ### Đánh giá lại sau khi train (cùng tập giữ lại, cùng seed)

# In[ ]:


# after = evaluate(n = 96)
# print("Sau:", after)

# In[ ]:


# after = evaluate(n = 96)
# print("\nTrước:", baseline)
# print("Sau:  ", after)

# In[ ]:


# # Xem thử 1 mẫu
# ex = eval_dataset[0]
# inputs = tokenizer(ex["image"], ex["prompt"], add_special_tokens = False, return_tensors = "pt").to("cuda")
# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024,
#                    use_cache = True, temperature = 1.0, min_p = 0.1)
# print("\nĐáp án thật:", ex["answer"])

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, use Hugging Face’s `push_to_hub` for online saving, or `save_pretrained` for local storage.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model.

# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving

# In[14]:


# """
# Inference với model Qwen3.5-4B Vision đã fine-tune bằng GRPO (adapter LoRA "qwen_lora").
# Dựa trên code baseline bạn gửi, chỉ đổi phần load model + prompt cho khớp với
# lúc train trong notebook qwen3-5-rl.ipynb (đúng câu chữ prompt, enable_thinking=False,
# parse thẻ <REASONING>/<SOLUTION> thay vì "chỉ trả tên lớp, không gì khác" như baseline).
# """

# import os
# import re
# from pathlib import Path
# from PIL import Image
# import pandas as pd
# import torch
# import tqdm
# from unsloth import FastVisionModel

# # =====================================================================
# # 1. CẤU HÌNH ĐƯỜNG DẪN KAGGLE
# # =====================================================================
# TEST_DIR = "/kaggle/input/datasets/chizus2602/air-craft/test"

# if not os.path.exists(TEST_DIR):
#     for alt_path in [
#         "/kaggle/input/aif-craft/test",
#         "/kaggle/input/aif-craft",
#         "/kaggle/input/khanhchien/aif-craft/test",
#     ]:
#         if os.path.exists(alt_path):
#             TEST_DIR = alt_path
#             break

# print(f"Thư mục test sử dụng: {TEST_DIR}")
# OUTPUT_CSV = "/kaggle/working/submission.csv"

# # =====================================================================
# # 2. MODEL ĐÃ FINE-TUNE (BASE + LoRA ADAPTER "qwen_lora")
# # =====================================================================
# # Nếu chạy tiếp trong CÙNG session vừa train xong:
# #   MODEL_NAME = "qwen_lora"          # thư mục local vừa save_pretrained
# # Nếu chạy ở session/notebook KHÁC (khuyến nghị cho việc test riêng):
# #   1) Upload thư mục "qwen_lora" (chứa adapter_config.json, adapter_model.safetensors...)
# #      thành 1 Kaggle Dataset, add vào notebook này
# #   2) Trỏ MODEL_NAME vào đường dẫn của dataset đó, ví dụ:
# MODEL_NAME = "/kaggle/input/models/chizus2602/model-checkpoint/pytorch/default/1/outputs/checkpoint-480"  # <-- SỬA cho đúng đường dẫn adapter của bạn

# if not os.path.exists(MODEL_NAME) and MODEL_NAME != "qwen_lora":
#     raise FileNotFoundError(
#         f"Không tìm thấy adapter tại {MODEL_NAME}. "
#         "Hãy upload thư mục qwen_lora (từ model.save_pretrained('qwen_lora') lúc train) "
#         "làm Kaggle Dataset và sửa lại MODEL_NAME."
#     )

# max_seq_length = 16384  # phải khớp giá trị lúc train

# # =====================================================================
# # 3. QUÉT DANH SÁCH ẢNH TEST
# # =====================================================================
# IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# test_images = []

# for root, _, files in os.walk(TEST_DIR):
#     for f in files:
#         if f.lower().endswith(IMAGE_EXTENSIONS):
#             test_images.append(os.path.join(root, f))

# test_images = sorted(test_images)
# print(f"-> Tìm thấy tổng cộng: {len(test_images)} ảnh test")

# # =====================================================================
# # 4. DANH SÁCH 96 CATEGORIES - PHẢI KHỚP Y HỆT CLASS_NAMES LÚC TRAIN
# #    (lấy nguyên văn từ list bạn gửi; đã kiểm tra khớp CLASS_NAMES trong notebook train,
# #    kể cả 'GRD.ZIL131' viết hoa GRD khác các lớp 'Grd.*' còn lại)
# # =====================================================================
# CATEGORIES = [
#     'Air.Mig29', 'Grd.Kraz255', 'Air.JH7', 'Air.F22', 'Air.Casa212', 'Air.Mi8', 'Air.AS365',
#     'Sea.Kuznetsov', 'Grd.SU100', 'Grd.T34', 'Grd.BMP1', 'Sea.Gorshkov', 'Air.F35', 'Grd.M113',
#     'Grd.T72', 'Air.Z9', 'Grd.Merkava', 'Air.Ka52', 'Air.T6Texan', 'Sea.OsaII', 'Air.J11',
#     'Grd.BTR80', 'Air.B29', 'Sea.Independence', 'Air.A10', 'Grd.PT76', 'Air.CH53', 'Air.J7',
#     'Air.IL76', 'Air.Yak130', 'Sea.Type001.Liaoning', 'Air.Z8', 'Sea.Type002.Shantong',
#     'Sea.Type054A.JiangkaiII', 'Air.B2', 'Grd.T54', 'Air.C17', 'Air.Yak52', 'Air.Mig35',
#     'Grd.2S1', 'Sea.Buyan', 'Grd.T90', 'Sea.TypeKangDing', 'Air.Mi28', 'Grd.Bradley',
#     'Grd.Humvee', 'Grd.Kraz6322', 'Air.Su22', 'Air.Mi17', 'Grd.M1Abrams', 'Sea.TypeChengKung',
#     'Sea.TypeAsahi', 'Grd.ZSU57', 'Sea.Molniya', 'GRD.ZIL131', 'Air.Su27', 'Air.Rafale',
#     'Grd.ZSU234', 'Grd.Ural4320', 'Sea.TypeMaya', 'Air.Su30', 'Sea.Kirov', 'Air.Mi24',
#     'Grd.BM27', 'Air.AH64', 'Grd.Himars', 'Grd.BMP2', 'Sea.ArleighBurke', 'Sea.Ticonderoga',
#     'Grd.BTR152', 'Sea.Type051B.Luhai', 'Grd.SU76', 'Grd.BTR90', 'Grd.Maz537', 'Grd.BTR60',
#     'Grd.LAV25', 'Sea.Type055.Renhai', 'Sea.Kilo', 'Grd.BM21', 'Grd.Kamaz', 'Grd.BRDM2',
#     'Sea.Type022.Houbei', 'Sea.Nanuchka', 'Air.L39', 'Sea.Wasp', 'Sea.Nimitz', 'Air.CH47',
#     'Grd.T62', 'Sea.GeraldFord', 'Air.C130', 'Grd.MTLB', 'Sea.Type053H3.JiangweiII',
#     'Air.KC135', 'Air.Ka27', 'Grd.BM30', 'Grd.Gaz66',
# ]
# assert len(CATEGORIES) == 96, f"Danh sách phải có 96 lớp, đang có {len(CATEGORIES)}"
# CATEGORY_SET = set(CATEGORIES)
# FALLBACK_LABEL = CATEGORIES[0]  # nhãn dự phòng khi không parse được gì (baseline dùng 'Air.Mig29')

# # =====================================================================
# # 5. PROMPT - Y HỆT lúc train (cell "Define the delimiter variables" trong
# #    qwen3-5-rl.ipynb), không dùng lại prompt kiểu "chỉ output tên lớp" của baseline,
# #    vì model đã được GRPO thưởng theo đúng cặp thẻ <REASONING>/<SOLUTION>.
# # =====================================================================
# REASONING_START = "<REASONING>"
# REASONING_END = "</REASONING>"
# SOLUTION_START = "<SOLUTION>"
# SOLUTION_END = "</SOLUTION>"

# QUESTION = (
#     "Which vehicle is shown in the image? "
#     f"Choose the class from this list: {', '.join(CATEGORIES)}"
# )
# TEXT_CONTENT = (
#     f"{QUESTION}. "
#     f"Respond with EXACTLY two tags and nothing else - no other text, no <think> block, no extra commentary. "
#     f"First a SHORT reasoning (at most 3 short sentences) between {REASONING_START} and {REASONING_END}, "
#     f"then your final answer between {SOLUTION_START} and (exactly one class name from the list, nothing else) {SOLUTION_END}. "
#     f"Do not restate the list. Stop right after {SOLUTION_END}."
# )


# def extract_category(text: str) -> str:
#     """Ưu tiên lấy nhãn trong <SOLUTION>...</SOLUTION>; nếu không có, dò trong toàn văn bản."""
#     # Bỏ mọi <think> còn sót (phòng khi enable_thinking=False không có tác dụng)
#     cleaned_full = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

#     # 1) Ưu tiên nội dung trong <SOLUTION>...</SOLUTION> - đúng tiêu chí lúc train
#     sol = re.findall(f"{SOLUTION_START}(.*?){SOLUTION_END}", cleaned_full, re.DOTALL)
#     if len(sol) == 1:
#         cand = sol[0].strip()
#         if cand in CATEGORY_SET:
#             return cand
#         for cat in CATEGORIES:  # không phân biệt hoa/thường, phòng model gõ sai case
#             if cat.lower() == cand.lower():
#                 return cat

#     # 2) Không có (hoặc sai) thẻ SOLUTION -> dò trong toàn bộ text còn lại
#     cleaned = cleaned_full.strip()
#     if cleaned in CATEGORY_SET:
#         return cleaned
#     for cat in CATEGORIES:
#         if cat in cleaned:
#             return cat
#     cleaned_lower = cleaned.lower()
#     for cat in CATEGORIES:
#         if cat.lower() in cleaned_lower:
#             return cat

#     # 3) Fallback cuối: dòng đầu tiên
#     first_line = cleaned.split("\n")[0].strip()
#     for cat in CATEGORIES:
#         if cat.lower() in first_line.lower():
#             return cat

#     return FALLBACK_LABEL


# # =====================================================================
# # 6. LOAD MODEL ĐÃ FINE-TUNE (base 4-bit cho GPU Kaggle 16GB + adapter LoRA)
# #    Lưu ý: lúc train adapter được học trên base 16-bit (load_in_4bit=False).
# #    Nạp lại base ở 4-bit cho inference là cách làm chuẩn để tiết kiệm VRAM,
# #    kết quả có thể lệch nhẹ so với lúc evaluate() trong lúc train (không lượng tử hoá).
# # =====================================================================
# print(f"Đang tải model đã fine-tune từ: {MODEL_NAME}")
# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name=MODEL_NAME,
#     max_seq_length=max_seq_length,
#     load_in_4bit=False,  # BẮT BUỘC cho GPU 16GB Kaggle
# )

# FastVisionModel.for_inference(model)

# # =====================================================================
# # 7. INFERENCE VÒNG LẶP
# # =====================================================================
# results = []

# for img_path in tqdm.tqdm(test_images, desc="Inference (finetuned)"):
#     try:
#         image = Image.open(img_path).convert("RGB").resize((512, 512))  # khớp bước resize lúc train

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": TEXT_CONTENT},
#                 ],
#             }
#         ]

#         # enable_thinking=False: tắt khối <think> mặc định của Qwen3.5, đúng như lúc train.
#         # Nếu bản tokenizer/transformers không nhận tham số này, xoá enable_thinking=False
#         # và dùng cách "mớm sẵn thẻ đóng think" như baseline (dòng bị comment bên dưới).
#         prompt_text = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             enable_thinking=False,
#         )
#         # prompt_text += "<think>\n\n</think>\n"  # bật lại nếu enable_thinking không khả dụng

#         inputs = tokenizer(
#             image,
#             prompt_text,
#             add_special_tokens=False,
#             return_tensors="pt",
#         ).to("cuda")

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=400,   # đủ cho <REASONING> (tối đa 3 câu) + <SOLUTION>, khớp lúc train (384)
#                 use_cache=True,
#                 do_sample=False,      # greedy, giống evaluate() lúc train
#             )

#         generated_ids = outputs[0][inputs.input_ids.shape[1]:]
#         raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

#         final_label = extract_category(raw_output)

#         results.append({"image_name": Path(img_path).name, "prediction": final_label})

#     except Exception as e:
#         print(f"Lỗi ảnh {img_path}: {e}")
#         results.append({"image_name": Path(img_path).name, "prediction": FALLBACK_LABEL})

# # =====================================================================
# # 8. XUẤT SUBMISSION
# # =====================================================================
# df = pd.DataFrame(results)
# df.to_csv(OUTPUT_CSV, index=False)

# print(f"\nĐã lưu file kết quả tại: {OUTPUT_CSV}")
# print("Mẫu 10 kết quả dự đoán thực tế:")
# print(df.head(10))
