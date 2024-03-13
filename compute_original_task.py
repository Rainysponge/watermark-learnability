import re
import json
import argparse


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of The Task",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="The model output",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Whether to expose the gradio demo to the internet.",
    )

    args = parser.parse_args()
    return args


def score_factory(args):
    if args.task.lower() == "boolq":
        compute_boolq_score(args)
    elif args.task.lower() == "cb":
        compute_CB_score(args)
    else:
        print("Error")


def compute_boolq_score(args):

    with open(args.result_file, "r") as f:
        result_list = json.load(f)
    ans_list = []
    # print(len(result['samples']['opt_kgw']['model_text']))
    for result in result_list["samples"]["opt_kgw"]["model_text"]:
        ans = result.split("\n")[0]
        if "yes" in ans.lower():
            ans_list.append(True)
        else:
            ans_list.append(False)
    print(len(result_list["samples"]["opt_kgw"]["model_text"]))

    data = list()
    with open(args.data_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
                # 如果加载成功，您可以在这里对 obj 进行处理
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")
    label = [item["answer"] for item in data]
    print(len(data))
    print(len(label), len(ans_list))
    assert len(label) == len(ans_list)
    hit = 0

    for i in range(len(label)):
        hit += 1 if label[i] == ans_list[i] else 0
    print(hit / len(label))


def compute_CB_score(args):
    with open(args.result_file, "r") as f:
        result_list = json.load(f)
    ans_list = []
    # print(len(result['samples']['opt_kgw']['model_text']))
    for result in result_list["samples"]["opt_kgw"]["model_text"]:
        ans = result.split("\n")[0]

        ans_list.append(ans.lower())

    print(len(result_list["samples"]["opt_kgw"]["model_text"]))

    data = list()
    with open(args.data_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
                data.append(obj)
                # 如果加载成功，您可以在这里对 obj 进行处理
            except json.JSONDecodeError as e:
                print(f"Error in line {line_num}: {e}")
    label = [item["label"] for item in data]
    hit = 0
    for i in range(len(label)):
        hit += 1 if label[i] == ans_list[i] else 0
    print(hit / len(label))


if __name__ == "__main__":
    args = parse_args()
    score_factory(args)
