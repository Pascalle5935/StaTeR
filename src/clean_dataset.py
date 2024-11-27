from collections import defaultdict
import argparse
from generate_file_names import *
import os


def find_earliest_and_latest_years(input_file: str):
    """
    Extracts the earliest and latest year of a dataset.
    """

    # set default earliest and latest years
    earliest_year = float('inf')
    latest_year = float('-inf')

    # extract earliest and latest years of input file
    with open(input_file, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            
            start_year = columns[3][:4] if columns[3][:4] != '####' else None
            end_year = columns[4][:4] if columns[4][:4] != '####' else None
            
            if start_year and (start_year.lstrip('-').isdigit()):
                start_year = int(start_year)
                if start_year < earliest_year:
                    earliest_year = start_year

            if end_year and (end_year.lstrip('-').isdigit()):
                end_year = int(end_year)
                if end_year > latest_year:
                    latest_year = end_year
    
    return str(earliest_year), str(latest_year)


def merge_intervals(intervals):
    """
    Merges overlapping intervals.
    """

    intervals.sort()
    merged = [intervals[0]]
    original_overlaps = []
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # if the current interval overlaps or touches the last, merge them
        if last[1] >= current[0]:
            new_end = max(last[1], current[1])
            # update the last interval with the merged one
            merged[-1] = (last[0], new_end)
            # save the overlap
            original_overlaps.append((last, current))
        else:
            merged.append(current)
    
    return merged, original_overlaps


def clean_dataset(input_file: str, output_file: str, invalid_file: str):
    earliest_year, latest_year = find_earliest_and_latest_years(input_file)

    temporal_data = defaultdict(list)
    with open(input_file, 'r') as infile:
        invalidfile = open(invalid_file, 'w')
        
        for line in infile:
            subject, predicate, object, start_date, end_date = line.strip().split('\t')

            # if either start or end date is negative, mark as invalid
            invalid = start_date.startswith('-') or end_date.startswith('-')

            # take only the year of the start and end dates
            start_date = start_date.split('-')[0]
            end_date = end_date.split('-')[0]

            # write line to invalid file if invalid
            if invalid:
                invalidfile.write(f"{subject}\t{predicate}\t{object}\t{start_date}\t{end_date}\n")
                continue

            # replace empty years with ealiest/latest
            if start_date == "####":
                start_date = earliest_year
            if end_date == "####":
                end_date = latest_year

            start_date = start_date.replace('#', '').zfill(4)
            end_date = end_date.replace('#', '').zfill(4)

            temporal_data[(subject, predicate, object)].append((start_date, end_date))

    # create new list with subject, predicate, object relation, with respective merged intervals, and a list for the overlapping intervals
    cleaned_data = []
    overlapping_intervals = []
    for key, intervals in temporal_data.items():
        merged_intervals, original_overlapping_intervals = merge_intervals(intervals)
        for start_date, end_date in merged_intervals:
            cleaned_data.append((*key, start_date, end_date))

        if original_overlapping_intervals:
            for overlap in original_overlapping_intervals:
                overlapping_intervals.append((*key, overlap[0], overlap[1]))

    # write merged intervals to output file
    with open(output_file, 'w') as outfile:
        for subject, predicate, object, start_date, end_date in cleaned_data:
            outfile.write(f"{subject}\t{predicate}\t{object}\t{start_date}-01-01\t{end_date}-01-01\n")

    # write original overlapping intervals to overlap file
    with open(invalid_file, 'w') as invalidfile:
        for subject, predicate, object, start_date, end_date in overlapping_intervals:
            invalidfile.write(f"{subject}\t{predicate}\t{object}\t{start_date}\t{end_date}\n")


def clean_dataset_main(dataset: str):

    # run function with correct parameters
    file_names = ['train', 'test', 'valid']
    for file_name in file_names:
        invalid_file_dir = os.path.dirname(f'../data_clean/{dataset}_clean/{file_name}_invalid.txt')
        os.makedirs(invalid_file_dir, exist_ok=True)

        input_file = f'../data_original/{dataset}/{file_name}.txt'
        invalid_file = f'../data_clean/{dataset}_clean/{file_name}_invalid.txt'
        output_file = f'../data_clean/{dataset}_clean/{file_name}_clean.txt'
        clean_dataset(input_file, output_file, invalid_file)


def get_unique_triples(dataset: str, output_file: str):
    """
    Gets the unique triples for the static graph and writes them to a new.
    """
    input_file = generate_file_name_orignal_training_set(dataset)
    unique_triples = set()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            subject, predicate, object, start_date, end_date = line.strip().split('\t')
            triple = (subject, predicate, object)

            if triple not in unique_triples:
                unique_triples.add(triple)
                outfile.write(f"{subject}\t{predicate}\t{object}\n")


# add parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)

# extract parameters
args = parser.parse_args()
dataset = args.dataset

clean_dataset_main(dataset)
print('Datasets cleaned.')

output_file = generate_file_name_unique_static_training_set(dataset)
get_unique_triples(dataset, output_file)
print(f"Unique static triples written in {output_file}")
