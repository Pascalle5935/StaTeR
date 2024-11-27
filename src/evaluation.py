from typing import List
from prediction import *


def parse_link_predictions(file_path):
    triples = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            # parse the actual triple line
            triple_line = lines[i].strip()
            head_entity, _, tail_entity = triple_line.split()
            head_entity, tail_entity = int(head_entity), int(tail_entity)
            i += 1

            # parse heads line
            heads_line = lines[i].strip().split(":")[1].strip().split()
            head_predictions = [(int(heads_line[j]), float(heads_line[j+1])) for j in range(0, len(heads_line), 2)]
            i += 1

            # parse tails line
            tails_line = lines[i].strip().split(":")[1].strip().split()
            tail_predictions = [(int(tails_line[j]), float(tails_line[j+1])) for j in range(0, len(tails_line), 2)]
            i += 1

            triples.append((head_entity, tail_entity, head_predictions, tail_predictions))
    
    return triples


def calculate_hits_and_mrr(triples):
    hits_at_1, hits_at_10 = 0, 0
    reciprocal_ranks = []

    for subject_entity, object_entity, subject_predictions, object_predictions in triples:
        # calculate subject metrics
        subject_ranks = [entity for entity, _ in sorted(subject_predictions, key=lambda x: -x[1])]
        if subject_entity in subject_ranks:
            rank = subject_ranks.index(subject_entity) + 1
            if rank == 1:
                hits_at_1 += 1
            if rank <= 10:
                hits_at_10 += 1
            reciprocal_ranks.append(1 / rank)
        
        # calculate object metrics
        object_ranks = [entity for entity, _ in sorted(object_predictions, key=lambda x: -x[1])]
        if object_entity in object_ranks:
            rank = object_ranks.index(object_entity) + 1
            if rank == 1:
                hits_at_1 += 1
            if rank <= 10:
                hits_at_10 += 1
            reciprocal_ranks.append(1 / rank)
    
    total_predictions = 2 * len(triples)
    hits_at_1 /= total_predictions
    hits_at_10 /= total_predictions
    mrr = sum(reciprocal_ranks) / total_predictions
    
    return hits_at_1, hits_at_10, mrr


def write_link_predictions_evaluation_to_file(input_files: List[str], output_file: str):
    with open(output_file, 'w') as file:
        for input_file in input_files:
            predictions = parse_link_predictions(input_file)
            hits_at_1, hits_at_10, mrr = calculate_hits_and_mrr(predictions)
            file.write(f'{input_file}\n')
            file.write(f'Hits@1: {hits_at_1:.4f}\n')
            file.write(f'Hits@10: {hits_at_10:.4f}\n')
            file.write(f'MRR: {mrr:.4f}\n')


def calculate_aeIOU_and_TAC(input_file: str):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        number_of_test_facs = int(lines[0].split('\t')[1])
        aeIOU_factor = 0
        TAC_factor = 0
        i = 1

        while i < len(lines):
            test_fact = lines[i].strip()
            _, _, _, fact_start_date, fact_end_date = test_fact.split()
            fact_start_date = int(fact_start_date)
            fact_end_date = int(fact_end_date)
            i += 1

            prediction = lines[i].strip()
            _, _, predicted_start_date, predicted_end_date = prediction.split()
            predicted_start_date = int(predicted_start_date)
            predicted_end_date = int(predicted_end_date)
            i += 1

            aeIOU_factor += calculate_aeIOU(fact_start_date, fact_end_date, predicted_start_date, predicted_end_date)
            TAC_factor += calculate_TAC(fact_start_date, fact_end_date, predicted_start_date, predicted_end_date)
    
    return (aeIOU_factor / number_of_test_facs), (TAC_factor / number_of_test_facs)



def write_time_predictions_evaluation_to_file(input_files: List[str], output_file: str):
    with open(output_file, 'w') as file:
        for input_file in input_files:
            aeIOU_factor, TAC_factor = calculate_aeIOU_and_TAC(input_file)
            file.write(f'{input_file}\n')
            file.write(f'aeIOU: {aeIOU_factor:.4f}\n')
            file.write(f'TAC: {TAC_factor:.4f}\n')

            
