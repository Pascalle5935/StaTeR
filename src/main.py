import concurrent.futures
from datetime import datetime
from os.path import exists
import time
import argparse
from tabulate import tabulate
from threading import Lock
import sys

from blazegraph_functions import (
    load_data_to_blazegraph,
)
from calculate_confidence import *
from convert_dataset_to_turtle import (
    convert_dataset_to_turle_main
)
from evaluation import *
from generate_file_names import *
from prediction import *
from evaluation import *


# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--learning_time', type=int)
parser.add_argument('--threshold', type=float)
parser.add_argument('--quality_measure_confidence', type=str)
parser.add_argument('--calculate_confidence', action='store_true')
parser.add_argument('--link_prediction', action='store_true')
parser.add_argument('--time_prediction', action='store_true')
parser.add_argument('--evaluation', action='store_true')

# extract parameters
args = parser.parse_args()
dataset = args.dataset
learning_time = str(args.learning_time)
threshold = args.threshold
quality_measure_confidence = args.quality_measure_confidence
calculate_confidence = args.calculate_confidence
link_prediction = args.link_prediction
time_prediction = args.time_prediction
evaluation = args.evaluation


def write_running_time_to_file(task: str, running_time: float):
    headers = ["Date time", "Task", "Running time"]
    # get current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    new_row = [[formatted_datetime, task, f"{running_time:.2f} seconds"]]

    # check if the file exists
    file_path = '../running_times.txt'
    file_exists = exists(file_path)

    # read existing rows if the file exists
    if file_exists:
        with open(file_path, "r") as file:
            existing_content = file.readlines()

        # extract rows after the headers
        existing_rows = [
            line.strip().split("|")[1:-1]
            for line in existing_content if "|" in line and "Task" not in line
        ]
        existing_rows = [[cell.strip() for cell in row] for row in existing_rows]
    else:
        existing_rows = []

    # add the new row to the existing rows
    all_rows = existing_rows + new_row

    # write everything back with headers
    with open(file_path, "w") as file:
        table = tabulate(all_rows, headers=headers, tablefmt="grid")
        file.write(table + "\n")


# code for evaluation
if evaluation:
    if link_prediction:
        basis_file_path = f'../link_predictions/{dataset}-{learning_time}-link-predictions'
        quality_measures_confidence = ["naive", "naive_overlap", "same_interval", "intersection_over_union", "temporal_alignment_coefficient"]
        metrics = ["aeIOU", "TAC"]

        input_files = [
            f"{basis_file_path}-{quality_measure}-{threshold}-{metric}"
            for quality_measure in quality_measures_confidence
            for metric in metrics
        ]
        output_file = f'../evaluation/{dataset}-{learning_time}-link-predictions-{threshold}'
        write_link_predictions_evaluation_to_file(input_files, output_file)
        print(f"Evaluation link prediction written in file {output_file}")
        sys.exit()

    if time_prediction:
        quality_measures_confidence = ["naive", "naive_overlap", "same_interval", "intersection_over_union", "temporal_alignment_coefficient"]
        input_files = [
            f'../time_predictions/{dataset}-{learning_time}-time-predictions-{quality_measure}-{str(threshold)}'
            for quality_measure in quality_measures_confidence
        ]
        output_file = f'../evaluation/{dataset}-{learning_time}-time-predictions-{threshold}'
        write_time_predictions_evaluation_to_file(input_files, output_file)
        print(f"Evaluation time prediction written in file {output_file}")
        sys.exit()


# create file names for the dataset files
basis_dataset_test_file_name = generate_file_name_for_dataset(dataset, 'test_clean')
basis_dataset_train_file_name = generate_file_name_for_dataset(dataset, 'train_clean')
basis_dataset_valid_file_name = generate_file_name_for_dataset(dataset, 'valid_clean')

# create temporal knowledge graph of train set and load it to blazegraph
# convert_dataset_to_turle_main(dataset)
train_set_turtle_file_name = generate_file_name_output_turtle(dataset)
load_data_to_blazegraph(train_set_turtle_file_name)


# code for calculating the confidence for the specified dataset, learning time, rule length, and quality_measure
if calculate_confidence:
    # start program
    start_time = time.time()

    # get function based on selected quality_measure
    if quality_measure_confidence == 'naive':
        function = process_rule_naive
    elif quality_measure_confidence == 'naive_overlap':
        function = process_rule_naive_overlap
    elif quality_measure_confidence == 'same_interval':
        function = process_rule_same_interval
    elif quality_measure_confidence == 'intersection_over_union':
        function = process_rule_intersection_over_union
    elif quality_measure_confidence == 'temporal_alignment_coefficient':
        function = process_rule_temporal_alignment_coefficient
    else:
        raise ValueError(f"Unsupported quality measure for calculate confidence: {quality_measure_confidence}")

    # get correct files based on settings
    input_file_calculate_confidence = generate_file_name_anyburl_rules(dataset, learning_time)
    output_file_calculate_confidence = generate_file_name_calculate_confidence(dataset, learning_time, quality_measure_confidence, str(threshold))
    output_file_calculate_confidence = f'../confidence_calculations/{dataset}-{learning_time}-{quality_measure_confidence}-{threshold}'

    # exctract rules
    rules = []
    with open(input_file_calculate_confidence, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            confidence = float(parts[2])
            # only add the rule if confidence is greater than the threshold
            if confidence > threshold:
                rule = parts[3]
                # check if rule has both head and body
                rule_parts = rule.split(' <= ')
                if len(rule_parts) != 2:
                    print(f"Skipping invalid rule: '{rule}'")
                    continue
                # add valid rules
                rules.append(parts[3])

    # compute confidence for correct quality measure, input rules, and dataset prefixes
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_rule = {executor.submit(function, rule, dataset): rule for rule in rules}

        # initialize a counter and a lock for thread-safe updates
        processed_count = 0
        lock = Lock()

        # monitor progress as tasks are completed
        results = []
        for future in concurrent.futures.as_completed(future_to_rule):
            try:
                rule, body_coverage, support, confidence = future.result(timeout=2)
                # increment the counter in a thread-safe manner
                with lock:
                    results.append((rule, body_coverage, support, confidence))
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"Completed processing rules: {processed_count} / {len(rules)}")
            except Exception as e:
                print(f"Error processing rule: {rule}: {e}")

    # write results to output file
    print("Write temporal confidences to file.")
    with open(output_file_calculate_confidence, 'w') as f:
        for (rule, body_coverage, support, confidence) in results:
            f.write(f"{body_coverage}\t{support}\t{confidence}\t{rule}\n")

    # write task and running time to running_times.txt
    end_time = time.time()
    write_running_time_to_file(f'{dataset}-{learning_time} calculate confidence {quality_measure_confidence} {threshold}', end_time - start_time)

    print(f"Confidence scores written in file {output_file_calculate_confidence}, and running time in ../running_times.txt")


# code for link prediction
if link_prediction:
    # start program
    start_time = time.time()

    basis_rules_file_name = generate_file_name_calculate_confidence(dataset, learning_time, quality_measure_confidence, threshold)
    basis_dataset_train_file_name = generate_file_name_training_set(dataset)
    basis_dataset_test_file_name = generate_file_name_test_set(dataset)

    # convert text rules and quintuples to objects
    rules = convert_text_rules_to_objects_of_class_rule(basis_rules_file_name, threshold)
    test_set = convert_text_relations_to_object_of_class_quintuple(basis_dataset_test_file_name)
    train_set = convert_text_relations_to_object_of_class_quintuple(basis_dataset_train_file_name)

    triple_object_predictions_aeIOU, triple_subject_predictions_aeIOU, triple_object_predictions_TAC, triple_subject_predictions_TAC = make_predictions_for_link_prediction(test_set, train_set, rules, dataset)

    output_file = generate_file_name_output_link_predictions(dataset, learning_time, quality_measure_confidence, str(threshold))
    output_file_aeIOU = output_file + '-aeIOU'
    write_link_predictions_to_file(output_file_aeIOU, test_set, triple_subject_predictions_aeIOU, triple_object_predictions_aeIOU)
    output_file_TAC = output_file + '-TAC'
    write_link_predictions_to_file(output_file_TAC, test_set, triple_subject_predictions_TAC, triple_object_predictions_TAC)
    
    end_time = time.time()
    write_running_time_to_file(f'{dataset}-{learning_time}-{quality_measure_confidence} link prediction {str(threshold)}', end_time - start_time)
    print(f"Predictions written in files {output_file_aeIOU} and {output_file_TAC}, and running time in ../running_times.txt")


# code for time prediction
if time_prediction:
    # start program
    start_time = time.time()

    basis_rules_file_name = generate_file_name_calculate_confidence(dataset, learning_time, quality_measure_confidence, threshold)
    basis_dataset_train_file_name = generate_file_name_training_set(dataset)
    basis_dataset_test_file_name = generate_file_name_test_set(dataset)

    # convert text rules and quintuples to objects
    rules = convert_text_rules_to_objects_of_class_rule(basis_rules_file_name, threshold)
    test_set = convert_text_relations_to_object_of_class_quintuple(basis_dataset_test_file_name)
    train_set = convert_text_relations_to_object_of_class_quintuple(basis_dataset_train_file_name)

    time_predictions = make_predictions_for_time_prediction(test_set, train_set, rules, dataset)
    output_file = generate_file_name_output_time_predictions(dataset, learning_time, quality_measure_confidence, str(threshold))
    write_time_predictions_to_file(output_file, test_set, time_predictions)

    end_time = time.time()
    write_running_time_to_file(f'{dataset}-{learning_time}-{quality_measure_confidence} time prediction', end_time - start_time)
    print(f"Predictions written in files {output_file}, and running time in ../running_times.txt")

