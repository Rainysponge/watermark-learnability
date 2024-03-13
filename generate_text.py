import json
import os
import argparse
from argparse import Namespace
from transformers import pipeline
from tqdm import tqdm
from pprint import pprint
from functools import partial

# import numpy  # for gradio hot reload
# import gradio as gr

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LogitsProcessorList,
)


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API"
    )

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=True,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--ouput_file",
        type=str,
        default="./output.txt",
        help="Path to save the result.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=42,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--gpu_rank",
        type=str,
        default=None,
        help="The RANK of GPU.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args


def format_names(s):
    """Format names for the gradio demo interface"""
    s = s.replace("num_tokens_scored", "Tokens Counted (T)")
    s = s.replace("num_green_tokens", "# Tokens in Greenlist")
    s = s.replace("green_fraction", "Fraction of T in Greenlist")
    s = s.replace("z_score", "z-score")
    s = s.replace("p_value", "p value")
    s = s.replace("prediction", "Prediction")
    s = s.replace("confidence", "Confidence")
    return s


def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k, v in score_dict.items():
        if k == "green_fraction":
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k == "confidence":
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float):
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append(
                [format_names(k), ("Watermarked" if v else "Human/Unwatermarked")]
            )
        else:
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2, ["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1, ["z-score Threshold", f"{detection_threshold}"])
    return lst_2d


def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
    )
    args.is_decoder_only_model = any(
        [
            (model_type in args.model_name_or_path)
            for model_type in ["pythia", "gpt", "opt", "bloom"]
        ]
    )
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        if args.gpu_rank is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.gpu_rank
        if args.load_fp16:
            pass
        else:
            model = model.to(device)

    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="left"
    )

    return model, tokenizer, device


def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
    and generate watermarked text by passing it to the generate method of the model
    as a logits processor."""

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))

    generate_without_watermark = partial(model.generate, **gen_kwargs)
    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = (
            model.config.max_position_embeddings - args.max_new_tokens
        )
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens

    tokd_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=args.prompt_max_length,
    ).to(device)
    truncation_warning = (
        True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    )
    redecoded_input = tokenizer.batch_decode(
        tokd_input["input_ids"], skip_special_tokens=True
    )[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[
            :, tokd_input["input_ids"].shape[-1] :
        ]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )[0]

    return (
        redecoded_input,
        int(truncation_warning),
        decoded_output_without_watermark,
        args,
    )


def generate_batch(prompts, args, model=None, device=None, tokenizer=None):
    """Generate text for multiple prompts using the provided model and tokenizer.

    Args:
        prompts (list[str]): A list of prompts to generate text for.
        args (argparse.Namespace): A namespace containing generation arguments.
        model (torch.nn.Module): The model to use for generation.
        device (torch.device): The device to use for computation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding and decoding.

    Returns:
        tuple: A tuple containing the re-decoded prompts, truncation warnings, decoded outputs, and the original args.
    """

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(do_sample=True, top_k=0, temperature=args.sampling_temp))
    else:
        gen_kwargs.update(dict(num_beams=args.n_beams))

    generate_without_watermark = partial(model.generate, **gen_kwargs)

    # Determine the prompt max length based on the model configuration or a default value
    if args.prompt_max_length:
        max_length = args.prompt_max_length
    elif hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        max_length = 2048 - args.max_new_tokens

    # Encode the prompts using the tokenizer
    encoded_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)

    # Prepare the inputs for generation
    input_ids = encoded_prompts["input_ids"]
    attention_mask = encoded_prompts.get("attention_mask")

    # Generate text for each prompt
    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(
        input_ids=input_ids, attention_mask=attention_mask
    )

    # Decode the generated text
    decoded_outputs_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    ######################### PLEASE REWRITE HERE ############################
    if args.task_name.lower() == "boolq":
        decoded_outputs_without_watermark = [
            "\n".join(item.split("\n")[11:])
            for item in decoded_outputs_without_watermark
        ]
    if args.task_name.lower() == "alpaca":
        pass
    ######################### PLEASE REWRITE HERE ############################
    # Prepare the outputs
    redecoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    truncation_warnings = [
        True if len(seq) == max_length else False for seq in redecoded_inputs
    ]

    return (
        redecoded_inputs,
        truncation_warnings,
        decoded_outputs_without_watermark,
        args,
    )


def learn_watwemark_alpaca():
    file_name = "./datasets/alpaca_data.json"
    with open(file_name, "r+") as f:
        content = json.load(f)

    data = {}
    data["sample"] = [
        "Human:\n" + item["instruction"] + item["input"] + "\nAssistant:\n"
        for item in content
    ]
    return data["sample"]


def alpaca_dataset(batch=8):
    file_name = "./datasets/alpaca_data.json"
    with open(file_name, "r+") as f:
        content = json.load(f)

    data = {}
    data["sample"] = [
        "Human:\n" + item["instruction"] + item["input"] + "\nAssistant:\n"
        for item in content
    ]
    grouped_data = [
        data["sample"][i : i + batch] for i in range(0, len(data["sample"]), batch)
    ]
    return grouped_data


def boolQ(batch=8):
    file_name = "./datasets/boolq/dev.jsonl"

    data = list()
    with open(file_name, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
                # 如果加载成功，您可以在这里对 obj 进行处理
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")
    print(len(data))
    data_ = [
        "Human:\nis there a now you see me 3 coming out?\nAssistant:\nyes.\nHuman:\nis jersey currency legal tender in the uk?\nAssistant:\nno.\n"
        + "Human:\n"
        + item["question"]
        + "?\nAssistant:\n"
        for item in data
    ]
    grouped_data = [data_[i : i + batch] for i in range(0, len(data_), batch)]

    return grouped_data


def CB(batch=8):
    file_name = "./datasets/CB/test.jsonl"
    data = []
    with open(file_name, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
                # 如果加载成功，您可以在这里对 obj 进行处理
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")
    data_ = ["Human:\n" + item["question"] + "?\nAssistant:\n" for item in data]
    grouped_data = [data_[i : i + batch] for i in range(0, len(data_), batch)]

    return grouped_data


def c4_data():
    file_name = "./datasets/c4/c4-train.00001-of-00512.json"
    data = []
    with open(file_name, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
                # 如果加载成功，您可以在这里对 obj 进行处理
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")

    return [item["text"] for item in data]


def WSC_dataset():
    file_name = "./datasets/WSC/val.jsonl"
    data = []
    with open(file_name, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
                # 如果加载成功，您可以在这里对 obj 进行处理
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")
    data_list = []
    for sample in data:
        text = sample["text"]
        span1 = sample["target"]["span1_text"]
        span2 = sample["target"]["span2_text"]
        # Does "A" refer to "B" in the sentence "ABSHDJKS"?
        data_list.append(
            "Human:\n"
            + 'Does "{}" refer to "{}" in the sentence "{}"?'.format(span2, span1, text)
            + "?\nAssistant:\n"
        )
    return data_list


if __name__ == "__main__":

    data = alpaca_dataset()
    data = learn_watwemark_alpaca()

    args = parse_args()
    model_name = "opt_kgw"
    args.normalizers = args.normalizers.split(",") if args.normalizers else []
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    outputs_file = {
        "samples": {
            model_name: {
                "watermark_config": [
                    {
                        "vocab_size": 50265,
                        "gamma": 0.5,
                        "delta": 2.0,
                        "seeding_scheme": "simple_1",
                        "hash_key": 15485863,
                        "select_green_tokens": True,
                    }
                ],
                "model_text": [],
            },
        }
    }
    for inputs in tqdm(data[:], desc="Generating outputs"):
        _, _, decoded_output_without_watermark, _ = generate(
            inputs, args, model=model, device=device, tokenizer=tokenizer
        )

        outputs_file["samples"][model_name]["model_text"].append(
            decoded_output_without_watermark
        )
        # IF USE GENERATE_BATCH()
        # outputs_file["samples"][model_name]["model_text"].extend(decoded_output_without_watermark)
    with open(args.ouput_file, "w") as f:
        json.dump(outputs_file, f)
