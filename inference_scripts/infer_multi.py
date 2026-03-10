import os
import json
import time
from io import BytesIO
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoProcessor
import transformers
from model.language_models import Qwen2_5_VLForConditionalGeneration
from transformers import AutoConfig
from model.qwen_vl_utils import process_vision_info

from torch.utils.data import DataLoader, DistributedSampler
from qwenvl.data.data_qwen import make_supervised_data_module, DataCollatorForSupervisedDataset

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    EvalArguments,
)

def run_worker(rank, world_size, model_args, data_args, eval_args, training_args, attn_implementation="flash_attention_2"):   
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map={"": rank},  # 每个rank只加载到自己对应的卡
    ).eval()
    model.to(device).bfloat16()           

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, 
        use_fast=True, 
        min_pixels=data_args.min_pixels, 
        max_pixels=data_args.max_pixels
    )
    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]
    collate = data_module["data_collator"]

    total_samples = len(dataset)
    per_rank = (total_samples + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, total_samples)
    subset_indices = list(range(start, end))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate,
        shuffle=False
    )

    if rank == 0:
        print(f"Total samples: {len(dataset)} | Per GPU: {len(subset)}")

    results = []
    ids_accum = []

    for batch_idx, batch in enumerate(tqdm(loader, disable=(rank != 0))):
        model_inputs = {}
        for k, v in batch.items():
            if k in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']:
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        v = v.to(device=device, dtype=torch.bfloat16)
                    else:
                        v = v.to(device)
                model_inputs[k] = v
        
        with torch.no_grad():          
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=eval_args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                # do_sample=True,     # 启用采样
                # num_beams=1,        # 采样时必须保持 num_beams=1
                # temperature=0.7,    # (可选) 温度，越低越接近贪婪，越高越随机
                # top_p=0.9,          # (可选) Top-p (Nucleus) 采样，0.9 是常用值
                # top_k=50,
            )
        trimmed_ids = [out[len(inp):] for inp, out in zip(model_inputs['input_ids'], generated_ids)]
        decoded_texts = processor.batch_decode(
            trimmed_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        results.extend(decoded_texts)
        batch_ids_list = batch['id']
        ids_accum.extend(batch_ids_list)

    # 每个rank写自己的临时输出文件
    tmp_path = eval_args.eval_save_path + f".rank{rank}.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": sid, "predict": text} for sid, text in zip(ids_accum, results)],
            f, ensure_ascii=False, indent=2
        )

    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        # 合并所有 rank 的 JSON 文件
        all_results = []
        for r in range(world_size):
            tmp_path = eval_args.eval_save_path + f".rank{r}.json"
            try:
                with open(tmp_path, "r", encoding="utf-8") as fin:
                    all_results.extend(json.load(fin))
                os.remove(tmp_path)
            except Exception as e:
                print(f"Warning: Could not load or remove {tmp_path}. Error: {e}")

        seen = set()
        merged = []
        for item in sorted(all_results, key=lambda x: str(x.get("id", ""))):
            item_id = item.get("id")
            if item_id is None:
                continue # 跳过没有ID的数据
                
            if item_id not in seen:
                merged.append(item)
                seen.add(item_id)
            else:
                print(f"Warning: Duplicate ID found during merge: {item_id}")

        # 保存合并后的最终结果
        with open(eval_args.eval_save_path, "w", encoding="utf-8") as fout:
            json.dump(merged, fout, ensure_ascii=False, indent=2)

        print(f"Total predictions merged: {len(merged)}")
        print(f"Saved all predictions to {eval_args.eval_save_path}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, EvalArguments, TrainingArguments)
    )
    model_args, data_args, eval_args, training_args = parser.parse_args_into_dataclasses()
 
    world_size = torch.cuda.device_count()
    print(f"Launching DDP inference on {world_size} GPUs...")
    
    mp.spawn(
        run_worker,
        args=(world_size, model_args, data_args, eval_args, training_args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":    

    main()