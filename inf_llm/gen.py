"""
This code is based on modifications to the FastChat code originally developed by the LMSYS team. 
The original project is licensed under the Apache License 2.0. 

## Changes Made

 - This code incorporates support for the InfLLM patch.

"""


"""Inference for FastChat models."""
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings
import torch
import argparse

from fastchat.serve.inference import (
    ChatIO, GptqConfig, AWQConfig, 
    ExllamaConfig, XftConfig, 
    load_model, 
)

from fastchat.serve.cli import (
    SimpleChatIO, 
    RichChatIO,
    ProgrammaticChatIO,
    str_to_torch_dtype,
    add_model_args
)

from inf_llm.utils import patch_hf

import transformers
transformers.logging.set_verbosity(transformers.logging.CRITICAL)

@torch.inference_mode()
def gen(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    dtype: Optional[torch.dtype],
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    exllama_config: Optional[ExllamaConfig] = None,
    xft_config: Optional[XftConfig] = None,
    inf_llm_config: Optional[dict] = None,
    revision: str = "main",
    judge_sent_end: bool = True,
    debug: bool = True,
    history: bool = True,
    clear_kv_cache = False,
    top_k: int = -1,
    top_p: float = 1.0,
    load_kv_cache_file: str = None,
    store_kv_cache_file: str = None,
    prompt_file: str = None,
):
    # Model
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        revision=revision,
        debug=debug,
    )
    if inf_llm_config is not None:
        model = patch_hf(model, inf_llm_config.type,  **inf_llm_config)
    
    with open(prompt_file) as f:
        prompt = f.read()
    # print("prompt:",prompt)
    input_ids = tokenizer(prompt).input_ids

    past_key_values = None

    # print("past_key_values:",past_key_values)
        
    start_length = 0

    assert len(input_ids) > start_length
    input_ids = input_ids[start_length:]
    if model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    out = model(input_ids=start_ids, use_cache=True, past_key_values=past_key_values)
    past_key_values = out.past_key_values
    if store_kv_cache_file is not None:
        # print(past_key_values)
        print(f"Storing past_key_values to{store_kv_cache_file}")
        for past_key_value in past_key_values:
            for block in past_key_value.global_blocks:
                for unit in block:
                    if unit.event is not None:
                        unit.event.wait()
                        unit.event = None
        torch.save(past_key_values, store_kv_cache_file)
def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None

    if args.inf_llm_config_path is not None:
        from omegaconf import OmegaConf
        inf_llm_config = OmegaConf.load(args.inf_llm_config_path)
        if inf_llm_config.conv_type == "llama-3-inst":
            args.conv_template = "llama-3-inst"

        inf_llm_config = inf_llm_config["model"]
    else:
        inf_llm_config = None

    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    print(f"device:{args.device}, {args.num_gpus}")
    try:
        gen(
            args.model_path,
            "cuda",
            args.num_gpus,
            args.max_gpu_memory,
            str_to_torch_dtype(args.dtype),
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.conv_system_msg,
            args.temperature,
            args.repetition_penalty,
            args.max_new_tokens,
            chatio,
            gptq_config=GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ),
            awq_config=AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            ),
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=args.revision,
            judge_sent_end=args.judge_sent_end,
            debug=args.debug,
            history=not args.no_history,
            inf_llm_config=inf_llm_config,
            clear_kv_cache=args.clear_kv_cache,
            load_kv_cache_file=args.load_kv_cache_file,
            store_kv_cache_file=args.store_kv_cache_file,
            prompt_file=args.prompt_file
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    parser.add_argument(
        "--inf-llm-config-path",
        type=str, help="Inf LLM patch config",
        default=None
    )
    parser.add_argument(
        "--clear-kv-cache",
        action="store_true"
    )
    parser.add_argument(
        "--load-kv-cache-file",
        type=str, help="load kv cache from",
        default=None
    )
    parser.add_argument(
        "--store-kv-cache-file",
        type=str, help="store kv cache to",
        default=None
    )
    parser.add_argument(
        "--prompt-file",
        type=str, help="load prompt from",
        default=None
    )
    args = parser.parse_args()
    main(args)
