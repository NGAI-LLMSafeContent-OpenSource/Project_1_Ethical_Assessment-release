
import os
import re
import json

import tqdm
import torch
import datasets
import transformers
import sentence_transformers
from detoxify import Detoxify, detoxify
from typing import Callable, Dict, Any, Optional

from modules.utils import ModelType, finetune_model_with_swift, prepare_swift_dataset, get_model_and_tokenizer_offline
from modules.diffs_funcs import DiffsFuncs


class DetoxifyTokenProcessor(transformers.LogitsProcessor):
    def __init__(
        self, toxic_model: transformers.PreTrainedModel, vocab_size: int, 
        max_gen_repeat_times: int, max_cache_len: int, detox_mode_name: str, 
        detox_args: Dict[str, Any], diffs_fn: Optional[Callable], diffs_fn_args: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.toxic_model = toxic_model
        self.vocab_size = vocab_size
        self.max_gen_repeat_times = max_gen_repeat_times
        self.detox_mode_name = detox_mode_name
        self.detox_args = detox_args
        self.diffs_fn = diffs_fn
        self.diffs_fn_args = diffs_fn_args
        self._max_cache_len = max_cache_len
        self._first_step = True
        self._pos = 0
        self._toxic_cache = transformers.StaticCache(
            config=self.toxic_model.config,
            max_cache_len=self._max_cache_len,
            offloading=False,
            offload_only_non_sliding=False
        )

    def reset(self) -> None:
        del self._toxic_cache
        self._toxic_cache = transformers.StaticCache(
            config=self.toxic_model.config,
            max_cache_len=self._max_cache_len,
            offloading=False,
            offload_only_non_sliding=False
        )
        self._first_step = True
        self._pos = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.detox_mode_name == "prompt_only":
            return scores
        
        with torch.inference_mode():
            if self._first_step:
                self._pos = input_ids.size(1)
                cache_position = torch.arange(self._pos, device=input_ids.device)
                toxic_outputs = self.toxic_model(
                    input_ids=input_ids, cache_position=cache_position, 
                    past_key_values=self._toxic_cache, use_cache=True
                )
                self._first_step = False
            else:
                cache_position = torch.tensor([self._pos], device=input_ids.device)
                toxic_outputs = self.toxic_model(
                    input_ids[..., -1:], cache_position=cache_position,
                    past_key_values=self._toxic_cache, use_cache=True
                )
                self._pos += 1
            
        toxic_logits: torch.Tensor = toxic_outputs.logits[..., -1, :self.vocab_size].to(torch.float32)
        tail_scores = scores[:, self.vocab_size:]
        scores = scores[:, :self.vocab_size]
        
        if self.detox_mode_name == "vanilla_cd":
            if self.diffs_fn is not None:
                cd_logits = torch.empty_like(scores)
                for batch_idx in range(scores.shape[0]):
                    diff_score = self.diffs_fn(scores[batch_idx].unsqueeze(0), toxic_logits[batch_idx].unsqueeze(0), **self.diffs_fn_args)
                    cutoff_score = torch.log(diff_score) + scores[batch_idx].max(dim=-1, keepdim=True).values
                    # 类似：http://arxiv.org/abs/2309.09117
                    # 差异是 logits 差，不是 alpha
                    diffs = (1 + self.detox_args["beta1"]) * scores[batch_idx] - self.detox_args["beta2"] * toxic_logits[batch_idx]
                    cd_logits[batch_idx] = diffs.masked_fill(scores[batch_idx] < cutoff_score, -1e8)
            else:
                # http://arxiv.org/abs/2309.09117
                # beta1 = 0, beta2 = 1 时，即为 https://aclanthology.org/2023.acl-long.687
                cutoff_score = torch.log(torch.tensor(self.detox_args["alpha"])) + scores.max(dim=-1, keepdim=True).values
                diffs = (1 + self.detox_args["beta1"]) * scores - self.detox_args["beta2"] * toxic_logits
                cd_logits = diffs.masked_fill(scores < cutoff_score, -1e8)
            scores = cd_logits

        elif self.detox_mode_name == "socd":
            V = scores.size(-1)
            for batch_idx in range(scores.shape[0]):
                raw_diff = self.diffs_fn(
                    scores[batch_idx].unsqueeze(0), toxic_logits[batch_idx].unsqueeze(0), **self.diffs_fn_args
                )
                if not torch.is_tensor(raw_diff):
                    raw_diff = torch.tensor(raw_diff, dtype=scores.dtype, device=scores.device)
                raw_diff = raw_diff.to(dtype=torch.float32, device=scores.device)

                if not torch.isfinite(raw_diff):
                    alpha = torch.tensor(0.0, dtype=torch.float32, device=scores.device)
                else:
                    t = torch.log1p(torch.clamp(raw_diff, min=0.0))
                    alpha = (t / (1.0 + t)).clamp(0.0, 1.0)

                if self.detox_args.get("k_setting") == "dynamic":
                    k = self.detox_args.get("k_max", 0.5)
                    chosen_vocab = int(torch.clamp((alpha * V * k).floor(), min=self.detox_args.get("min_vocab", 10), max=0.5 * V).item())
                else:
                    chosen_vocab = self.detox_args.get("min_vocab", 10)

                lp_base = torch.log_softmax(scores[batch_idx], dim=-1)
                lp_toxic = torch.log_softmax(toxic_logits[batch_idx], dim=-1)
                diff = lp_toxic - lp_base

                diff_pos = torch.where(diff > 0, diff, torch.tensor(float('-inf'), device=diff.device, dtype=diff.dtype))
                topk_vals, topk_idx = torch.topk(diff_pos, chosen_vocab, dim=-1)
                valid_mask = torch.isfinite(topk_vals)
                if valid_mask.any():
                    sel_idx = topk_idx[valid_mask]
                    scores[batch_idx, sel_idx] = scores[batch_idx, sel_idx] - alpha * toxic_logits[batch_idx, sel_idx]
            # return torch.cat([scores, tail_scores], dim=-1)
        
        elif self.detox_mode_name == "top_k_cd":
            for batch_idx in range(scores.shape[0]):
                # Step 1: 计算差值与绝对差值
                diff_score = self.diffs_fn(scores[batch_idx].unsqueeze(0), toxic_logits[batch_idx].unsqueeze(0), **self.diffs_fn_args)
                cutoff_score = diff_score.item()
                logits_diffs = scores[batch_idx] - toxic_logits[batch_idx]
                logits_diffs_abs = torch.abs(logits_diffs)

                # Step 2: 找出大于等于 cutoff_score 的索引
                cutoff_mask: torch.Tensor = logits_diffs_abs >= cutoff_score
                eligible_indices = cutoff_mask.nonzero(as_tuple=False).squeeze(-1)

                # Step 3: 提取 top_k_num 绝对差值最大的索引
                if eligible_indices.numel() > 0:
                    selected_diffs = logits_diffs_abs[cutoff_mask]
                    topk_vals, topk_indices_in_mask = torch.topk(selected_diffs, min(self.detox_args["top_k_num"], selected_diffs.numel()))
                    topk_indices = eligible_indices[topk_indices_in_mask]
                    # 在 scores 上批量修改
                    # 最终 logits = (1 + beta1) * 原始 logits - beta2 * logits 差异的绝对值
                    scores[batch_idx, topk_indices] = (1 + self.detox_args["beta1"]) * scores[batch_idx, topk_indices] - self.detox_args["beta2"] * logits_diffs_abs[topk_indices]
                    
        else:
            raise ValueError(f"Unsupported detox mode: {self.detox_mode_name}.")
        
        return torch.cat([scores, tail_scores], dim=-1)


def text_detox(seed: int, loop_idx: int, final_output_dir_prefix: str, data_args: dict, model_args: dict, gen_args: dict, detox_args: dict, **kwargs) -> Dict[str, str | int]:
    print(f"\n>>> Starting Text Detoxification Generation")
    transformers.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.recompile_limit = 2**31
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args["model_path"])
    if model_args["model_type"] == ModelType.GPT2.value:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    def example_process(example: dict, model_type: str, tok: transformers.PreTrainedTokenizer, max_length: int, text_field: str, shots_num: int = 0) -> dict:
        system_prompt = "You are a helpful assistant. Your task is to transcribe directly or indirectly harmful content in the text, into legal and reasonable content."
        detox_shots_list = ["\nHere is an example input: this is a terrible fact.\nOutput: <answer>this is a bad fact.</answer>\n"]
        detox_shots = "".join(detox_shots_list[:shots_num]) if shots_num > 0 else ""
        detox_prefix = "You should return the result between \'<answer>\' and \'</answer>\'."
        detox_tail =  "\nHere is the text that may contain harmful content:\n\n"
        assistant_prompt = "Here is the transcription result:\n\n<answer>"
        
        if model_type == ModelType.QWEN2.value:
            input_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{detox_prefix}{detox_shots}{detox_tail}{example[text_field]}<|im_end|>\n<|im_start|>assistant\n{assistant_prompt}"
        else:
            input_prompt = f"{system_prompt}\n\n{detox_prefix}{detox_shots}{detox_tail}{example[text_field]}\n\n{assistant_prompt}"
            
        tokenized_inputs = tok(
            input_prompt, add_special_tokens=False, padding="max_length", return_tensors="pt",
            truncation=True, max_length=max_length, pad_to_multiple_of=8
        )
        return {
            "raw_text": example[text_field],
            "input_prompt": input_prompt,
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0)
        }

    dataset = datasets.load_dataset("json", data_files=data_args["input_file"], split="train")
    tokenized_dataset = dataset.map(
        example_process,
        fn_kwargs={"model_type": model_args["model_type"], "tok": tokenizer, "max_length": gen_args["max_prompt_length"], "text_field": data_args["text_field"], "shots_num": detox_args.get("shots_num", 1)},
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
            "raw_texts": [item["raw_text"] for item in batch]
        }

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=gen_args["batch_size"], collate_fn=collate_fn,
        shuffle=False, pin_memory=True, num_workers=data_args.get("data_workers", 4), persistent_workers=True
    )

    print(">>> Loading Toxic Model...")
    final_toxic_model_path = model_args["toxic_model_path"] if not kwargs.get("use_new_toxic_model", False) else kwargs.get("toxic_model_path_refresh", model_args["toxic_model_path"])
    toxic_model = transformers.AutoModelForCausalLM.from_pretrained(
        final_toxic_model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).eval().to(device)
    
    print(">>> Loading Base Model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args["model_path"], dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).eval().to(device)

    detox_mode_name = gen_args["detox_mode_name"]
    diffs_fn_name = detox_args.get("diffs_fn", {}).get("diffs_fn_name", "js")
    detox_processor = DetoxifyTokenProcessor(
        toxic_model=toxic_model, vocab_size=tokenizer.vocab_size, max_gen_repeat_times=gen_args["max_gen_repeat_times"],
        max_cache_len=gen_args["max_new_tokens"] + gen_args["max_prompt_length"] + 32,
        detox_mode_name=detox_mode_name, detox_args=detox_args.get(detox_mode_name, {}),
        diffs_fn=DiffsFuncs.get(diffs_fn_name), diffs_fn_args=detox_args.get("diffs_fn", {}).get("diffs_fn_args", {})
    )
    logits_processor = transformers.LogitsProcessorList([detox_processor])

    generation_config = transformers.GenerationConfig.from_pretrained(model_args["model_path"])
    generation_config.max_new_tokens = gen_args["max_new_tokens"]
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.do_sample = gen_args["do_sample"]
    generation_config.num_return_sequences = gen_args["max_gen_repeat_times"]
    generation_config.use_cache = True
    generation_config.cache_implementation = "static"

    output_dir = final_output_dir_prefix
    output_file = os.path.join(output_dir, f"detox_{detox_mode_name}_{model_args['model_type']}_{model_args['model_size_info']}.jsonl")

    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Detoxifying text"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        bs = input_ids.size(0)
        detoxified_texts_list = [[] for _ in range(bs)]
        
        for temperature in gen_args["temperature_list"]:
            generation_config.temperature = temperature
            detox_processor.reset()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    generation_config=generation_config, logits_processor=logits_processor
                )

            prompt_lens = input_ids.size(1)
            for i in range(bs):
                start = outputs.size(1) - prompt_lens
                seq_block = outputs[i * gen_args["max_gen_repeat_times"] : (i + 1) * gen_args["max_gen_repeat_times"], -start - 3:]
                texts = tokenizer.batch_decode(seq_block, skip_special_tokens=True)
                for t_idx, text in enumerate(texts, start=1):
                    detoxified_texts_list[i].append({"temperature": temperature, "current_time": t_idx, "text": text})
        
        with open(output_file, "a", encoding="utf-8") as f:
            for i, detoxified_text_list in enumerate(detoxified_texts_list):
                gen_result = {
                    "raw_text": batch["raw_texts"][i],
                    "detoxified_texts": [t["text"] for t in detoxified_text_list],
                    "output": detoxified_text_list
                }
                f.write(json.dumps(gen_result, ensure_ascii=False) + "\n")
                
    print(f">>> Completed. Results saved to: {output_file}")
    
    return {
        "loop_idx": loop_idx,
        "output_file": output_file
    }


def toxic_rank(seed: int, loop_idx: int, final_output_dir_prefix: str, data_args: dict, embeddings_args: dict, detoxify_args: dict, rank_mode_args: dict) -> Dict[str, str | int]:
    print(f"\n>>> Starting Fusion Ranking")
    transformers.set_seed(seed)

    def example_extractor(example: Dict[str, Any]) -> Dict[str, Any]:
        extracted = []
        for text in example["detoxified_texts"]:
            match = re.match(r"<answer>(.*?)</answer>", text, re.DOTALL)
            extracted.append(match.group(1) if match else text.split("<answer>")[0].strip())
        return {"raw_text": example["raw_text"], "extracted_answers": list(set(extracted))}

    def example_toxicity_eval(example: Dict[str, Any], model: Detoxify) -> Dict[str, Any]:
        answers = example["extracted_answers"]
        origin_scores = model.predict(example["raw_text"])
        fused_origin = origin_scores["toxicity"] * (1 + origin_scores["severe_toxicity"])
        
        score_dicts = []
        ans_scores = model.predict(answers)
        for ans_idx, ans_txt in enumerate(answers):
            fused_score = ans_scores["toxicity"][ans_idx] * (1 + ans_scores["severe_toxicity"][ans_idx])
            score_dicts.append({
                "answer": ans_txt, "fused_toxicity_score": fused_score,
                "toxicity_scores": {k: v[ans_idx] for k, v in ans_scores.items()},
            })
        return {"toxicity_scores": score_dicts, "fused_origin_toxicity_score": fused_origin, "origin_toxicity_scores": origin_scores}

    def example_semantic_rank_cos(example: Dict[str, Any], embed_model) -> Dict[str, Any]:
        answers = example["extracted_answers"]
        emb_ans = embed_model.encode(answers, convert_to_tensor=True)
        emb_raw = embed_model.encode(example["raw_text"], convert_to_tensor=True)
        return {"semantic_scores": [{"answer": txt, "cosine_similarity": torch.nn.functional.cosine_similarity(emb_raw, emb, dim=-1).item()} for txt, emb in zip(answers, emb_ans)]}

    def example_get_final_text(example: Dict[str, Any], lambda_value: float) -> Dict[str, Any]:
        tox_scores, sem_scores = example["toxicity_scores"], example["semantic_scores"]
        final_scores = [-lambda_value * tox["fused_toxicity_score"] + (1 - lambda_value) * sem["cosine_similarity"] for tox, sem in zip(tox_scores, sem_scores)]
        # score = -tox["fused_toxicity_score"] # lambda = 1
        # score = sem["cosine_similarity"] # lambda = 0
        best_idx = int(torch.tensor(final_scores).argmax().item())
        return {
            "final_detoxified_text": tox_scores[best_idx]["answer"],
            "fused_toxicity_score": tox_scores[best_idx]["fused_toxicity_score"],
            "all_toxicity_scores": tox_scores[best_idx]["toxicity_scores"],
            "semantic_score": sem_scores[best_idx]["cosine_similarity"]
        }

    dataset = datasets.load_dataset("json", data_files=data_args["input_file"])["train"]
    dataset = dataset.map(example_extractor, remove_columns=dataset.column_names, desc="Extracting answers")
    
    print(">>> Initializing Detoxify Model...")
    detoxify.get_model_and_tokenizer = get_model_and_tokenizer_offline
    detox_model = Detoxify(model_type=detoxify_args["model_type"], checkpoint=detoxify_args["ckpt_path"], huggingface_config_path=detoxify_args["huggingface_config_path"], device="cuda:0")
    dataset = dataset.map(example_toxicity_eval, fn_kwargs={"model": detox_model}, desc="Eval toxicity scores")
    
    print(">>> Initializing Embedding Model...")
    embed_model = sentence_transformers.SentenceTransformer(embeddings_args["embedding_model_path"], device="cuda:0", local_files_only=True, model_kwargs={"dtype": torch.bfloat16})
    
    if rank_mode_args["rank_mode_name"] == "cosine_similarity":
        dataset = dataset.map(example_semantic_rank_cos, fn_kwargs={"embed_model": embed_model}, desc="Eval semantic similarity")
        
    dataset = dataset.map(example_get_final_text, fn_kwargs={"lambda_value": rank_mode_args["lambda_value"]}, desc="Get final detoxified text")

    # 计算去毒前后的宏观分数差值
    direct_score = {
        "after": {k: sum(ex["all_toxicity_scores"][k] for ex in dataset)/len(dataset) for k in dataset[0]["all_toxicity_scores"].keys()},
        "before": {k: sum(ex["origin_toxicity_scores"][k] for ex in dataset)/len(dataset) for k in dataset[0]["origin_toxicity_scores"].keys()}
    }

    output_dir = final_output_dir_prefix
    base_name = os.path.splitext(os.path.basename(data_args["input_file"]))[0]
    out_file = os.path.join(output_dir, f"ranked_{rank_mode_args['rank_mode_name']}_{base_name}.jsonl")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in tqdm.tqdm(dataset, desc="Saving results"):
            f.write(json.dumps({
                "raw_text": ex["raw_text"], 
                "detoxified_texts": ex["final_detoxified_text"],
                "eval_scores": {
                    "fused_toxicity_score": ex["fused_toxicity_score"],
                    "semantic_score": ex["semantic_score"],
                    "direct_score": direct_score
                }
            }, ensure_ascii=False) + "\n")
            
    print(f">>> Fusion Ranking completed. Results saved to: {out_file}")
    
    return {
        "loop_idx": loop_idx,
        "output_file": out_file
    }


def do_sft(seed: int, loop_idx: int, final_output_dir_prefix: str, step1_config: dict) -> Dict[str, str | int]:
    print(f"\n>>> Starting Toxic Model SFT")
    toxic_dataset = datasets.load_dataset("json", data_files=step1_config["dataset"])["train"]
    swift_dataset_path = os.path.join(final_output_dir_prefix, "swift_formatted_data.jsonl")
    swift_jsonl_path = prepare_swift_dataset(toxic_dataset, swift_dataset_path, text_field=step1_config["text_field"])
    
    sft_output_model_path = os.path.join(final_output_dir_prefix, "sft_toxic_model")
    sft_output = finetune_model_with_swift(
        seed=seed,
        model_id_or_path=step1_config["model_path"],
        dataset_path=swift_jsonl_path,
        output_dir=sft_output_model_path,
        **step1_config["train_args"]
    )
    
    return {
        "loop_idx": loop_idx,
        "best_model_checkpoint": sft_output["best_model_checkpoint"]
    }
    

def run_detox_pipeline(pipeline_config: Dict[str, Any]):
    """
    统一入口方法，用于外部调用去毒流水线。
    :param pipeline_config: 包含各步骤配置及运行开关的嵌套字典。
    """
    seed = pipeline_config["seed"]
    loop_idx = pipeline_config["loop_idx"]
    output_dir_prefix = pipeline_config["output_dir_prefix"]
    run_step_1 = pipeline_config.get("run_step_1", False)
    run_step_2 = pipeline_config.get("run_step_2", True)
    run_step_3 = pipeline_config.get("run_step_3", True)
    
    final_output_dir_prefix = f"{output_dir_prefix}_loop_{loop_idx}"
    detox_output_file_path = None

    if run_step_1:
        step1_config = pipeline_config.get("step1_config", {})
        sft_output_dict = do_sft(seed=seed, loop_idx=loop_idx, final_output_dir_prefix=final_output_dir_prefix, step1_config=step1_config)

    if run_step_2:
        step2_config = pipeline_config.get("step2_config", {})
        detox_output_dict = text_detox(
            seed=seed,
            loop_idx=loop_idx,
            final_output_dir_prefix=final_output_dir_prefix,
            data_args=step2_config.get("data", {}),
            model_args=step2_config.get("model", {}),
            gen_args=step2_config.get("gen", {}),
            detox_args=step2_config.get("detox", {}),
            use_new_toxic_model=step2_config.get("use_new_toxic_model", False),
            toxic_model_path_refresh=sft_output_dict["best_model_checkpoint"] if run_step_1 else None
        )

    if run_step_3:
        step3_config = pipeline_config.get("step3_config", {})
        detox_output_file_path = step3_config.get("output_file")
        if detox_output_file_path and run_step_2:
            if "data" not in step3_config:
                step3_config["data"] = {}
            step3_config["data"]["input_file"] = detox_output_file_path

        rank_output_dict = toxic_rank(
            seed=seed,
            loop_idx=loop_idx,
            final_output_dir_prefix=final_output_dir_prefix,
            data_args=step3_config.get("data", {}),
            embeddings_args=step3_config.get("embeddings", {}),
            detoxify_args=step3_config.get("detoxify", {}),
            rank_mode_args=step3_config.get("rank_mode", {})
        )
    
    print("\n>>> Detox Pipeline completed.")


if __name__ == "__main__":
    external_config = {
        "loop_idx": 1,
        "seed": 1337,
        "output_dir_prefix": "output/detox",
        
        # toxic model sft
        "run_step_1": False,
        # detox
        "run_step_2": False,
        # rank
        "run_step_3": True,
        
        "step1_config": {
            "model_path": "models/Qwen2.5-0.5B-Instruct",
            "dataset": "data/dghs_to_detox.jsonl",
            "text_field": "text",
            "output_dir": "output/toxic_model_sft",
            "train_args": {
                "train_type": "full",
                "max_steps": 6,
                "split_dataset_ratio": 0.2,
                "save_steps": 2,
                "eval_steps": 2,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "logging_steps": 1,
                "metric_for_best_model": "eval_token_acc",
                "greater_is_better": True,
                "load_best_model_at_end": True,
                "save_total_limit": 1,
                "use_chat_template": True
            }
        },
        
        "step2_config": {
            "use_new_toxic_model": False,
            "data": {
                "input_file": "data/dghs_to_detox.jsonl",
                "text_field": "text",
                "data_workers": 4
            },
            "model": {
                "model_type": "qwen2",
                "model_size_info": "3-0.5",
                "model_path": "models/Qwen2.5-3B-Instruct",
                "toxic_model_path": "output/detox_loop_1/sft_toxic_model/v0-20260313-004151/checkpoint-4"
            },
            "gen": {
                "batch_size": 2,
                "max_prompt_length": 512,
                "max_new_tokens": 512,
                "max_gen_repeat_times": 2,
                "do_sample": True,
                "temperature_list": [0.7],
                "detox_mode_name": "prompt_only"   # "prompt_only", "vanilla_cd", "top_k_cd", "socd"
            },
            "detox": {
                "diffs_fn": {"diffs_fn_name": "js", "diffs_fn_args": {"wsd_p": 1, "wsd_blur": 0.01}},
                # diffs_fn_name: none, fkl, rkl, js, tvd, wsd
                "shots_num": 1,
                "vanilla_cd": {"alpha": 0.1, "beta1": 0.5, "beta2": 0.5},
                "top_k_cd": {"top_k_num": 10, "beta1": 0.5, "beta2": 0.5},
                "socd": {"k_setting": "dynamic", "k_max": 0.5, "min_vocab": 10, "beta1": 0.5, "beta2": 0.5}
            }
        },
        
        "step3_config": {
            "data": {
                "input_file": "output/detox_loop_1/detox_socd_qwen2_3-0.5.jsonl"
            },
            "embeddings": {
                "embedding_model": "Qwen3-Embedding-0.6B",
                "embedding_model_path": "models/Qwen3-Embedding-0.6B",
                "dim_num": 1024
            },
            "detoxify": {
                "model_type": "original",
                "ckpt_path": "models/detoxify/toxic_original-c1212f89.ckpt",
                "huggingface_config_path": "models/bert-base-uncased"
            },
            "rank_mode": {
                "rank_mode_name": "cosine_similarity",
                "lambda_value": 0.5
            },
        }
    }

    run_detox_pipeline(external_config)