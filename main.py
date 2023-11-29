import argparse
import os
import random
import numpy as np

from modify_data import convert_dataset
from data_store import *
from train import *

random.seed(42)
np.random.seed(42)

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action="store_true", help = "Verbose")

    parser.add_argument("--train", action="store_true", help = "Add this when you need to train a model and save it")

    parser.add_argument("--evaluate", action="store_true", help = "Add this when you need to test a model and save the results")
    parser.add_argument("--model_evaluate_task", default="acos", choices=["acos", "asqp", "aste", "tasd"], type=str, help="Name of the task whose model is to be tested")
    parser.add_argument("--model_evaluate_subtask", default="rest16", type=str, help="Name of the subtask whose model is to be tested")

    parser.add_argument('--train_epochs', default="10")

    parser.add_argument("--task", default="acos", choices=["acos", "asqp", "aste", "tasd"], type=str, help="Name of the task to be performed")
    parser.add_argument("--subtask", default="rest16", type=str, help="Name of the subtask to be used as dataset.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = init_args()
    if not os.path.exists("modified_data"):
        for task in task_subtasks_list:
            if not os.path.exists('modified_data/{}'.format(task)):
                os.makedirs('modified_data/{}'.format(task))
            for subtask in task_subtasks_list[task]:
                if not os.path.exists('modified_data/{}/{}'.format(task,subtask)):
                    os.makedirs('modified_data/{}/{}'.format(task,subtask))
                args.task = task
                args.subtask = subtask
                print(args.task, args.subtask)
                lines = []
                with open("data/{}/{}/train.txt".format(args.task, args.subtask),"r",encoding="utf-8") as f:
                    print("Dataset used: data/{}/{}/train.txt".format(args.task, args.subtask))
                    lines = f.readlines()
                dataset = convert_dataset(lines, args.task, args.subtask)
                max_len_input = 0
                max_len_output = 0
                with open("modified_data/{}/{}/train.txt".format(args.task, args.subtask),"w",encoding="utf-8") as f:
                    for row in dataset:
                        # max_len_input = max(max_len_input, len(row[0].split(' ')))
                        max_len_input += len(row[0].split(' '))
                        max_len_output += len(row[1].split(' '))
                        # max_len_output = max(max_len_output, len(row[1].split(' ')))
                        f.write(row[0]+"\t"+row[1]+"\n")
                # print("Task: {} Subtask: {} Training Max input length {}, Max output length {}".format(task, subtask, max_len_input, max_len_output))
                if args.verbose:
                    print("Task: {} Subtask: {} Training avg input length {}, avg output length {}".format(task, subtask, max_len_input/len(dataset), max_len_output/len(dataset)))

                lines = []
                with open("data/{}/{}/dev.txt".format(args.task, args.subtask),"r",encoding="utf-8") as f:
                    print("Dataset used: data/{}/{}/dev.txt".format(args.task, args.subtask))
                    lines = f.readlines()
                dataset = convert_dataset(lines, args.task, args.subtask)
                max_len_input = 0
                max_len_output = 0
                with open("modified_data/{}/{}/dev.txt".format(args.task, args.subtask),"w",encoding="utf-8") as f:
                    for row in dataset:
                        # max_len_input = max(max_len_input, len(row[0].split(' ')))
                        max_len_input += len(row[0].split(' '))
                        max_len_output += len(row[1].split(' '))
                        # max_len_output = max(max_len_output, len(row[1].split(' ')))
                        f.write(row[0]+"\t"+row[1]+"\n")
                # print("Task: {} Subtask: {} Validation Max input length {}, Max output length {}".format(task, subtask, max_len_input, max_len_output))
                if args.verbose:
                    print("Task: {} Subtask: {} Validation avg input length {}, avg output length {}".format(task, subtask, max_len_input/len(dataset), max_len_output/len(dataset)))

                lines = []
                with open("data/{}/{}/test.txt".format(args.task, args.subtask),"r",encoding="utf-8") as f:
                    print("Dataset used: data/{}/{}/test.txt".format(args.task, args.subtask))
                    lines = f.readlines()
                dataset = convert_dataset(lines, args.task, args.subtask)
                max_len_input = 0
                max_len_output = 0
                with open("modified_data/{}/{}/test.txt".format(args.task, args.subtask),"w",encoding="utf-8") as f:
                    for row in dataset:
                        # max_len_input = max(max_len_input, len(row[0].split(' ')))
                        max_len_input += len(row[0].split(' '))
                        max_len_output += len(row[1].split(' '))
                        # max_len_output = max(max_len_output, len(row[1].split(' ')))
                        f.write(row[0]+"\t"+row[1]+"\n")
                # print("Task: {} Subtask: {} Testing Max input length {}, Max output length {}".format(task, subtask, max_len_input, max_len_output))
                if args.verbose:
                    print("Task: {} Subtask: {} Testing avg input length {}, avg output length {}".format(task, subtask, max_len_input/len(dataset), max_len_output/len(dataset)))

    if args.train:
        train_function(args.task, args.subtask, int(args.train_epochs))

    if args.evaluate:
        model_name = "model_{}_{}".format(args.model_evaluate_task,args.model_evaluate_subtask)
        if not os.path.exists("models/{}".format(model_name)):
            train_function(args.model_evaluate_task, args.model_evaluate_subtask, int(args.train_epochs))
        evaluate_function(args.task, args.subtask, model_name) 

        