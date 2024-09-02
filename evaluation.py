import numpy as np
from torchmetrics.text.rouge import ROUGEScore
from argparse import ArgumentParser

parser = ArgumentParser(description='Evaluation')
parser.add_argument('--model', default="bert-base-uncased", type=str, help='large language model')
parser.add_argument('--dataset', default="cola", type=str, help='dataset for task')
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--setting", default="benchmark", type=str, help="setting for evaluation")
parser.add_argument("--ablation", default="no-ablation", type=str, help="ablation or not")
parser.add_argument("--dropout", default=0.1, type=float, help="dropout rate")
parser.add_argument("--defense", default="no-defense", type=str, help="defense or not")
parser.add_argument("--noise", default=0.001, type=float, help="dropout rate")
parser.add_argument("--counter_noise", default="DL_NN", type=str, help="defense or not")
parser.add_argument("--prune", default=0.75, type=float, help="dropout rate")
parser.add_argument("--counter_prune", default="DL_PM", type=str, help="defense or not")
args = parser.parse_args()

model = args.model
setting = args.setting
batch_size = args.batch_size
dataset = args.dataset
ablation = args.ablation
dropout = args.dropout
defense = args.defense
noise = args.noise
counter_noise = args.counter_noise
prune = args.prune
counter_prune = args.counter_prune

R1s = []
R2s = []
RLs = []

for run in ["first", "second", "third"]:
    if ablation == "no-ablation" and defense == "no-defense":
        result_file = f"results/{setting}/{dataset}/{model}_{run}_run_b_{batch_size}.txt"

    elif ablation == "bert_tiny":
        result_file = f"results/ablation/bert_tiny/{setting}/{run}_run_b_{batch_size}.txt"

    elif ablation == "bert_large":
        result_file = f"results/ablation/bert_large/{setting}/{run}_run_b_{batch_size}.txt"

    elif ablation == "roberta_tiny":
        result_file = f"results/ablation/roberta_tiny/{setting}/{run}_run_b_{batch_size}.txt"

    elif ablation == "roberta_large":
        result_file = f"results/ablation/roberta_large/{setting}/{run}_run_b_{batch_size}.txt"

    elif ablation == "roberta_base":
        result_file = f"results/ablation/roberta_base/{setting}/{run}_run_b_{batch_size}.txt"

    elif ablation == "bert_large_grad_clip":
        result_file = f"results/ablation/bert_large/{setting}/grad_clip/{run}_run_b_{batch_size}.txt"

    elif ablation == "roberta_large_grad_clip":
        result_file = f"results/ablation/roberta_large/{setting}/grad_clip/{run}_run_b_{batch_size}.txt"

    elif ablation == "dropout":
        result_file = f"results/ablation/dropout/{dropout}/{setting}/{run}_run_b_{batch_size}.txt"

    elif ablation == "label":
        result_file = f"results/ablation/label/{run}_run_b_{batch_size}.txt"

    elif ablation == "longest_length":
        result_file = f"results/ablation/longest_length/{run}_run_b_{batch_size}.txt"

    elif ablation == "label_longest_length":
        result_file = f"results/ablation/label_longest_length/{run}_run_b_{batch_size}.txt"

    elif ablation == "no-ablation" and defense == "noise":
        result_file = f"results/defense/noise/{noise}/{counter_noise}/{run}_run_b_{batch_size}.txt"

    elif ablation == "no-ablation" and defense == "prune":
        result_file = f"results/defense/prune/{prune}/{counter_prune}/{run}_run_b_{batch_size}.txt"



    with open(result_file, "r") as file:
        lines = file.readlines()

    model_out = []
    reference = []
    for i in range(len(lines)):
        if lines[i] == "Current Reference:\n":
            current_reference = []
            for j in range(batch_size):
                current_reference.append(lines[i + j + 1].strip())
            reference.append(current_reference)
        elif "solution is better" in lines[i]:
            current_solution = []
            current_solution.append(lines[i + 1].split("solution:")[1].strip())
            for j in range(batch_size - 1):
                current_solution.append(lines[i + j + 2].strip())
            model_out.append(current_solution)

    if len(model_out) > len(reference):
        model_out = model_out[:len(reference)]
    elif len(model_out) < len(reference):
        reference = reference[:len(model_out)]

    overall_scores = {"r1": [], "r2": [], "rl": []}
    rouge = ROUGEScore(accumulate="best")
    for i in range(len(model_out)):
        references = reference[i]
        solutions = model_out[i]
        batch_scores = {"r1": [], "r2": [], "rl": []}
        for ref in references:
            temp_scores = []
            for solution in solutions:
                temp_scores.append(rouge(preds=solution, target=ref))
            temp_r1 = []
            temp_r2 = []
            temp_rl = []
            for score in temp_scores:
                temp_r1.append(score["rouge1_fmeasure"].item())
                temp_r2.append(score["rouge2_fmeasure"].item())
                temp_rl.append(score["rougeL_fmeasure"].item())
            batch_scores["r1"].append(max(temp_r1))
            batch_scores["r2"].append(max(temp_r2))
            batch_scores["rl"].append(max(temp_rl))

        overall_scores["r1"].append(np.mean(batch_scores["r1"]))
        overall_scores["r2"].append(np.mean(batch_scores["r2"]))
        overall_scores["rl"].append(np.mean(batch_scores["rl"]))

    R1s.append(np.mean(overall_scores["r1"]))
    R2s.append(np.mean(overall_scores["r2"]))
    RLs.append(np.mean(overall_scores["rl"]))

# calculate the mean and variance
print(f"R1: {np.mean(R1s)} ± {np.std(R1s)}")
print(f"R2: {np.mean(R2s)} ± {np.std(R2s)}")
print(f"RL: {np.mean(RLs)} ± {np.std(RLs)}")
