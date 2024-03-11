# On the Learnability of Watermarks for Language Models

This repository contains code for the paper [On the Learnability of Watermarks for Language Models](https://arxiv.org/abs/2312.04469) by Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto.

The `kgw_watermarking` directory is from [github.com/jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking). In the `kth_watermarking` directory, `detect.py`, `levenshtein.pyx`, and `mersenne.py` are from [github.com/jthickstun/watermark](https://github.com/jthickstun/watermark). `train_logits_distill.py` and `train_sampling_distill.py` are adapted from [github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py).

The links to trained model weights from the paper's experiments are shown in [Origin Repo](https://github.com/chenchenygu/watermark-learnability).

# How To Run

## Setup

The code runs on Python 3.11.8 with PyTorch 2.0.1.

```python
conda create -n watermark_learnability python=3.11
conda activate watermark_learnability
pip install -r requirements.txt
```

## Logits Distill

### torchrun
```cmd
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_logits_distill.py --train_file ./datasets/alpaca_data.json - --model_name opt --model_name_or_path facebook/opt-1.3b     --do_train   --fp16     --per_device_train_batch_size 4     --learning_rate 2e-5     --num_train_epochs 1     --output_dir ./output/    --overwrite_output_dir     --save_steps 0     --save_strategy "no" --watermark_type kgw --argmax_watermark false --do_eval False
```
- model_name The name of the model in Huggingface or the path on local path.
### deepspeed

For more details on Deepspeed, see [DeepseedExample](https://github.com/microsoft/DeepSpeedExamples).
```cmd
# zero 2
deepspeed --num_nodes=1 --num_gpus=2 train_logits_distill.py --train_file ./datasets/alpaca_data.json --deepspeed ./ds_config_fp16_z2.json    --model_name_or_path /mnt/workspace/huzhanyi/pythia_/Models/OPT/1.3B     --do_train     --do_eval     --fp16     --per_device_train_batch_size 4     --learning_rate 2e-5     --num_train_epochs 1     --output_dir ./output/opt_kgw     --overwrite_output_dir     --save_steps 0     --save_strategy "no" --watermark_type kgw --argmax_watermark false --do_eval False
```

- watermark_type 

The type of watermark is bound to the argmax_watermark. If you are using kgw, please set argmax_watermark as false.
- output

Directory used to store the trained model. In order to better use the subsequent watermark detection code, it is recommended to set the directory name as "\{model_name\}_\{watermark_type\}".

Using train_sampling_distill.py is similar to using train_logits_distill.py.

## Detector

### Generate text

Create your own train data loading code as follows:
```python
def c4_data():
    file_name = "./datasets/c4-train.00001-of-00512.json"
    data = []
    with open(file_name, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")
    
    return [
        item["text"]
        for item in data
    ]
```
And then, rewrite the following codes.
```python
    outputs_file = {
        "samples": {
            model_name: {
                "watermark_config": [{"vocab_size": 50265, "gamma": 0.5, "delta": 2.0, "seeding_scheme": "simple_1", "hash_key": 15485863, "select_green_tokens": True}],
                "model_text": [],
            },
        }
    }
```

And then, utilize the dataset and your own model to generate answers.

```cmd
torchrun generate_text.py --model_name_or_path the_path_to_your_model --output_file ./output.txt
```

### Detection

Taking compute_watermark_scores.py as an example.
```cmd
torchrun compute_watermark_scores.py --tokenizer_name the_path_to_your_model --input_file the_file_generated_by_your_own_model --output_file the_file_for_saving_the_score
```

## Layer Split

Rewrite the code in train_logits_distill.py
```python
for name, param in model.named_parameters():
    if len(name.split(".")) > 4 and name.split(".")[3].isdigit():
        if int(name.split(".")[3]) < 20:
            # Only train the last 4 layers
            param.requires_grad = False

all_params = model.parameters()
total_params = sum(p.numel() for p in all_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", total_params)
print("Total trainable parameters:", trainable_params)
```
