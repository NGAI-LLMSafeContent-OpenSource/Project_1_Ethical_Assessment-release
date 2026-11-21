
import os
import json
import time
import enum
import random

import torch
import datasets
import omegaconf
import accelerate
import transformers
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class ModelType(str, enum.Enum):
    GPT2 = "gpt2"
    QWEN2 = "qwen2"
    

def get_model_and_tokenizer_offline(
    model_type: str, 
    model_name: str, 
    tokenizer_name: str, 
    num_classes: int, 
    state_dict: torch.nn.Module.state_dict, 
    huggingface_config_path: Optional[str] = None
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    model_class = getattr(transformers, model_name)
    config = model_class.config_class.from_pretrained(huggingface_config_path or model_type, num_labels=num_classes)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
    )
    return model, tokenizer


def save_args(args_dict: omegaconf.OmegaConf, arg_output_dir: str, accelerator: accelerate.Accelerator = None) -> str:
    if accelerator is None:
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_dir = os.path.join(arg_output_dir, time_str)
        os.makedirs(output_dir, exist_ok=True)
        current_args = omegaconf.OmegaConf.to_container(args_dict, resolve=True)
        with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(current_args, f, ensure_ascii=False, indent=4)
        return output_dir
    
    else:
        if accelerator.is_main_process:
            timestamp_float = time.time()
            timestamp_tensor = torch.tensor([timestamp_float], dtype=torch.float32, device=accelerator.device)
        else:
            timestamp_tensor = torch.empty(1, dtype=torch.float32, device=accelerator.device)
        timestamp_tensor = accelerate.utils.broadcast(timestamp_tensor)
        shared_timestamp = timestamp_tensor.item()
        shared_timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(shared_timestamp))
        
        output_dir = os.path.join(arg_output_dir, shared_timestamp_str)
        
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            current_args = omegaconf.OmegaConf.to_container(args_dict, resolve=True)
            with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(current_args, f, ensure_ascii=False, indent=4)
            
        return output_dir
    

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finetune_model_with_swift(
    model_id_or_path: str,
    dataset_path: str,
    output_dir: str,
    task_type: str = "causal_lm",
    train_type: str = "full",
    use_chat_template: bool = False,
    torch_dtype: str = 'bfloat16',
    split_dataset_ratio: float = 0.01,
    warmup_ratio: float = 0.05,
    max_length: int = 512,
    max_steps: Optional[int] = None,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    loss_scale: str = "all",
    eval_steps: int = 50,
    save_steps: int = 50,
    logging_steps: int = 5,
    dataset_num_proc: int = 4,
    save_only_model: bool = True,
    save_safetensors: bool = True,
    check_model: bool = False,
    seed: int = 1337,
    **kwargs
) -> Dict[str, Any]:
    
    set_seed(seed)
    
    from swift.llm import sft_main, TrainArguments
    if sft_main is None:
        raise ImportError("ms-swift is not installed. Please install it to use this fine-tuning method.")
    
    # 解决 GPT2 的 hidden_size 属性读取问题
    from swift.llm.model.utils import HfConfigFactory
    @staticmethod
    def compat_zero3(config: transformers.PretrainedConfig) -> None:
        value = HfConfigFactory.get_config_attr(config, 'hidden_size') or config.hidden_size
        try:
            config.hidden_size = value
        except AttributeError:
            pass
    HfConfigFactory.compat_zero3 = compat_zero3
    
    sft_args_dict = dict(
        seed=seed,
        data_seed=seed,
        model=model_id_or_path, 
        dataset=[dataset_path],
        split_dataset_ratio=split_dataset_ratio,
        output_dir=output_dir,
        task_type=task_type,
        use_chat_template=use_chat_template,
        train_type=train_type,
        torch_dtype=torch_dtype,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        dataset_num_proc=dataset_num_proc,
        loss_scale=loss_scale,
        max_steps=max_steps,
        save_only_model=save_only_model,
        save_safetensors=save_safetensors,
        check_model=check_model,
    )
    
    sft_args_dict.update(kwargs)
    sft_args = TrainArguments(**sft_args_dict)
    
    print(f"Starting MS-Swift fine-tuning ...")
    result = sft_main(sft_args)
    return result


def prepare_swift_dataset(dataset: datasets.Dataset, output_jsonl_path: str, text_field: str = "text") -> str:
    """
    将原始数据集处理为 ms-swift 所需的 messages 格式:
    {"messages": [{"role": "assistant", "content": "..."}]}
    """
    print(f"Formatting dataset for ms-swift to {output_jsonl_path}")
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            text_content = item.get(text_field, "")
            if text_content:
                record = {"messages": [{"role": "assistant", "content": text_content}]}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
    return output_jsonl_path


if __name__ == "__main__":
    pass
    # finetune_model_with_swift(
    #     model_id_or_path="/home/sw/pt_models/GPT2-125M",
    #     dataset_path="output/detox_loop_1/swift_formatted_data.jsonl",
    #     output_dir="/home/sw/gd-detox-sys/test/output/sft",
    #     task_type="causal_lm",
    #     train_type="full",
    #     use_chat_template=False,
    #     torch_dtype='bfloat16',
    #     split_dataset_ratio=0.01,
    #     warmup_ratio=0.05,
    #     max_length=512,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     learning_rate=1e-5,
    #     gradient_accumulation_steps=1,
    #     eval_steps=50,
    #     save_steps=50,
    #     logging_steps=5,
    #     dataset_num_proc=4,
    #     save_only_model=True,
    #     save_safetensors=True,
    #     check_model=False,
    # )