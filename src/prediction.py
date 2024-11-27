from typing import List, Tuple, Dict, Set, Optional
from model import *
from blazegraph_functions import *
from collections import defaultdict
import concurrent.futures
from threading import Lock
import random


## general functions
def convert_text_rules_to_objects_of_class_rule(file_path_rules: str, threshold: float) -> List[Rule]:
    """
    Reads rules of the form: predicate(subject,object) <= predicate(subject,object), predicate(subject,object), ...,
    and converts them to a list of Rule objects.

    Parameters:
    - file_path_rules (str): String indicating which file with rules has to be read.
    - threshold (float): Threshold for the confidence value, all rules with confidence below threshold are filtered.
    """

    model_rules: List[Rule] = []
    with open(file_path_rules, 'r') as file:
        for line in file:
            body_coverage, support, confidence, rule = line.strip().split('\t')

            # check if valid rule
            if len(rule.split(' <= ')) != 2:
                print(f"Invalid rule: '{rule}'")
                continue

            # only create model if confidence greater than threshold
            if float(confidence) > threshold:
                model_rule: Rule = convert_text_rule_to_object_of_class_rule(rule, body_coverage, support, confidence)
                model_rules.append(model_rule)

    return model_rules


def convert_text_rule_to_object_of_class_rule(rule: str, body_coverage: str, support: str, confidence: str) -> Rule:
    """
    Creates a Rule object.

    Parameters:
    - rule (str): The rule to be split into a head and multiple body atoms, of the form:
                  predicate(subject,object) <= predicate(subject,object), predicate(subject,object), ...
    - body_coverage (str): The body coverage of the rule.
    - support (str): The support of the rule.
    - confidence (str): The confidence of the rule.
    """

    parts = rule.split(' <= ')
    head_str, body_str = parts

    # extract head atom information and create Triple object
    head_p, head_rest = head_str.split('(')
    head_s, head_o = head_rest.strip(')').split(',')
    head_triple = Triple(head_s.strip(), head_p.strip(), head_o.strip())

    # extract body atom information and create list of Triple object
    body_triples: List[Triple] = []
    body_atoms = body_str.split(', ')
    for body_atom in body_atoms:
        body_p, body_rest = body_atom.split('(')
        body_s, body_o = body_rest.strip(')').split(',')
        body_triple = Triple(body_s.strip(), body_p.strip(), body_o.strip())
        body_triples.append(body_triple)

    return Rule(head_triple, body_triples, int(float(body_coverage)), int(float(support)), float(confidence))


def convert_text_relations_to_object_of_class_quintuple(file_path_data_set: str) -> List[Quintuple]:
    """
    Reads text relations from a file and converts them to Quintuple objects.

    Parameters:
    - file_path_data_set (str): String indicating the train set with relations.
    """

    objects: List[Quintuple] = []
    with open(file_path_data_set, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            subject, predicate, object = parts[:3]
            start_date, end_date = parts[3], parts[4]

            # extract the year from the start and end date
            start_date = start_date.split('-')[0]
            end_date = end_date.split('-')[0]

            # create a Quintuple object and add it to the list
            quintuple = Quintuple(subject, predicate, object, start_date, end_date)
            objects.append(quintuple)

    return objects


def find_rules_for_test_relation_in_head_link_prediction(rules: dict, test_triple: Quintuple):
    matched_rules_predict_object: List[Rule] = []
    matched_rules_predict_subject: List[Rule] = []

    # Retrieve rules for the predicate of the test triple
    relevant_rules: List[Rule] = rules.get(test_triple.get_predicate(), [])

    for rule in relevant_rules:
        rule_head = rule.get_head()
        if rule_head.get_object().isalpha():
            # swapped variable filled in
            filled_in_object = substitute_variable_in_rule(rule, test_triple.get_object(), 'Y')
            matched_rules_predict_subject.append(filled_in_object)

            # head: predicate(subject, Y)
            if rule_head.get_subject() == test_triple.get_subject():
                matched_rules_predict_object.append(rule)

        elif rule_head.get_subject().isalpha():
            # swapped variable filled in
            filled_in_subject = substitute_variable_in_rule(rule, test_triple.get_subject(), 'X')
            matched_rules_predict_object.append(filled_in_subject)

            # head: predicate(X, object)
            if rule_head.get_object() == test_triple.get_object():
                matched_rules_predict_subject.append(rule)

        # head: predicate(X, Y)
        elif rule_head.get_subject().isalpha() and rule_head.get_object().isalpha():
            matched_rules_predict_object.append(rule)
            matched_rules_predict_subject.append(rule)        

    # sort for confidence in descending order
    matched_rules_predict_object.sort(key=lambda rule: rule.get_confidence(), reverse=True)
    matched_rules_predict_subject.sort(key=lambda rule: rule.get_confidence(), reverse=True)

    return matched_rules_predict_object, matched_rules_predict_subject


def find_rules_for_test_relation_in_head_time_prediction(rules: dict, test_triple: Quintuple) -> List[Rule]:
    matched_rules: List[Rule] = []

    # Retrieve rules for the predicate of the test triple
    relevant_rules: List[Rule] = rules.get(test_triple.get_predicate(), [])
    for rule in relevant_rules:
        rule_head = rule.get_head()
        if rule_head.get_subject().isalpha() and rule_head.get_object().isalpha():
            substitute_variable_in_rule(rule, test_triple.get_subject(), 'X')
            substitute_variable_in_rule(rule, test_triple.get_object(), 'Y')
            matched_rules.append(rule)
        elif rule_head.get_object().isalpha() and rule_head.get_subject() == test_triple.get_subject():
            substitute_variable_in_rule(rule, test_triple.get_object(), 'Y')
            matched_rules.append(rule)
        elif rule_head.get_subject().isalpha() and rule_head.get_object() == test_triple.get_object():
            substitute_variable_in_rule(rule, test_triple.get_subject(), 'X')
            matched_rules.append(rule)

    # sort for confidence in descending order
    matched_rules.sort(key=lambda rule: rule.get_confidence(), reverse=True)

    return matched_rules


head_in_training_set: Set[Tuple[str, str, str]] = set()
head_not_in_training_set: Set[Tuple[str, str, str]] = set()
def filter_rules_head_in_training_set(rules: List[Rule], training_set_intervals_with_triple_index: dict) -> List[Rule]:
    """
    Filters rules by checking if their head exists in the training set of quintuples.
    Uses a cache to speed up subsequent checks for the same rule head.

    Parameters:
    - rules (List[Rule]): List of rules to filter.
    - train_set (Set[Quintuple]): Training set of quintuples.

    Returns:
        List[Rule]: Filtered list of rules whose heads are not in the training set.
    """
    global head_in_training_set, head_not_in_training_set
    filtered_rules = []
    for rule in rules:
        head = rule.get_head()
        head_tuple = (head.subject, head.predicate, head.object)

        # Check the cache first
        if head_tuple in head_in_training_set:
            continue
        if head_tuple in head_not_in_training_set:
            filtered_rules.append(rule)
            continue

        # Check the precomputed set of training triples
        if head_tuple in training_set_intervals_with_triple_index:
            head_in_training_set.add(head_tuple)
        else:
            head_not_in_training_set.add(head_tuple)
            filtered_rules.append(rule)

    return filtered_rules


def check_if_body_in_training_set(rule: Rule, training_set_intervals_with_triple_index: dict) -> bool:
    for body_atom in rule.get_body():
        if (body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object()) not in training_set_intervals_with_triple_index:
            return False

    return True

## link prediction
def process_test_relation_link_prediction(test_relation: Quintuple, rules_with_predicate_index: dict, training_set_intervals_with_triple_index: dict, dataset: str):    
    rules_predict_object, rules_predict_subject = find_rules_for_test_relation_in_head_link_prediction(rules_with_predicate_index, test_relation)
    filtered_rules_predict_object = filter_rules_head_in_training_set(rules_predict_object, training_set_intervals_with_triple_index)
    filtered_rules_predict_subject = filter_rules_head_in_training_set(rules_predict_subject, training_set_intervals_with_triple_index)
    
    # print("Rules")
    # for rule in filtered_rules_predict_object:
    #     print(rule.__str__())

    object_predictions_aeIOU, object_predictions_TAC = get_link_prediction_predictions(
        filtered_rules_predict_object, test_relation, training_set_intervals_with_triple_index, dataset, True, 10
    )
    subject_predictions_aeIOU, subject_predictions_TAC = get_link_prediction_predictions(
        filtered_rules_predict_subject, test_relation, training_set_intervals_with_triple_index, dataset, False, 10
    )

    return (
        test_relation,
        object_predictions_aeIOU, subject_predictions_aeIOU,
        object_predictions_TAC, subject_predictions_TAC
    )


def make_predictions_for_link_prediction(test_set: List[Quintuple], train_set: List[Quintuple], rules: List[Rule], dataset: str):
    # index rules by predicate
    rules_with_predicate_index = defaultdict(list)
    for rule in rules:
        rules_with_predicate_index[rule.get_head().get_predicate()].append(rule)

    # index training set by triple
    training_set_intervals_with_triple_index = defaultdict(list)
    for quintuple in train_set:
        triple_key = (quintuple.get_subject(), quintuple.get_predicate(), quintuple.get_object())
        start_time, end_time = quintuple.get_interval()
        training_set_intervals_with_triple_index[triple_key].append((start_time, end_time))

    triple_object_predictions_aeIOU = {}
    triple_subject_predictions_aeIOU = {}
    triple_object_predictions_TAC = {}
    triple_subject_predictions_TAC = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Prepare tasks
        future_to_test_relation = {
            executor.submit(
                process_test_relation_link_prediction, 
                test_relation,
                rules_with_predicate_index,
                training_set_intervals_with_triple_index, 
                dataset
            ): test_relation
            for test_relation in test_set
        }

        # initialize a counter and a lock for thread-safe updates
        processed_count = 0
        lock = Lock()
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_test_relation):
            try:
                test_relation, object_aeIOU, subject_aeIOU, object_TAC, subject_TAC = future.result()

                with lock:
                    triple_object_predictions_aeIOU[test_relation] = object_aeIOU
                    triple_subject_predictions_aeIOU[test_relation] = subject_aeIOU
                    triple_object_predictions_TAC[test_relation] = object_TAC
                    triple_subject_predictions_TAC[test_relation] = subject_TAC
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"Completed processing test facts: {processed_count} / {len(test_set)}")
            except Exception as e:
                print(f"Error processing test relation: {e}, test relation: {test_relation.__str__()}")

    return (
        triple_object_predictions_aeIOU,
        triple_subject_predictions_aeIOU,
        triple_object_predictions_TAC,
        triple_subject_predictions_TAC,
    )


def get_link_prediction_predictions(rules: List[Rule], test_relation: Quintuple, training_set_intervals_with_triple_index: dict, dataset: str, predict_for_object: bool, max: int):
    predictions_rules_list_aeIOU: Dict[str, List[Rule]] = {}
    predictions_rules_list_TAC: Dict[str, List[Rule]] = {}

    def add_to_predictions_aeIOU(key: str, rule: Rule):
        """Helper to add a rule to the prediction list."""
        if key not in predictions_rules_list_aeIOU:
            predictions_rules_list_aeIOU[key] = []  # initialize if key is absent
        predictions_rules_list_aeIOU[key].append(rule)

    def add_to_predictions_TAC(key: str, rule: Rule):
        """Helper to add a rule to the prediction list."""
        if key not in predictions_rules_list_TAC:
            predictions_rules_list_TAC[key] = []  # initialize if key is absent
        predictions_rules_list_TAC[key].append(rule)

    def add_to_precitions(head_key: str, rule: Rule):
        aeIOU_factor = get_aeIOU_factor(rule, test_relation, training_set_intervals_with_triple_index)
        added_factor_aeIOU_in_confidence_rule = Rule(rule.get_head(), rule.get_body(), rule.get_body_coverage(), rule.get_support(), aeIOU_factor * rule.get_confidence())
        add_to_predictions_aeIOU(head_key, added_factor_aeIOU_in_confidence_rule)

        TAC_factor = get_TAC_factor(rule, test_relation, training_set_intervals_with_triple_index)
        added_factor_TAC_in_confidence_rule = Rule(rule.get_head(), rule.get_body(), rule.get_body_coverage(), rule.get_support(), TAC_factor * rule.get_confidence())
        add_to_predictions_TAC(head_key, added_factor_TAC_in_confidence_rule)


    for rule in rules:
        if rule_has_variable(rule):
            groundings = get_grounding_for_variables_link_prediction(rule, dataset, predict_for_object)
            for grounding in groundings:
                newRule = Rule(rule.get_head(), rule.get_body(), rule.get_body_coverage(), rule.get_support(), rule.get_confidence())
                for variable, value in grounding.items():
                    newRule = substitute_variable_in_rule(newRule, value, variable)
                if check_if_body_in_training_set(newRule, training_set_intervals_with_triple_index):
                    head_key = newRule.get_head().get_object() if predict_for_object else newRule.get_head().get_subject()
                    add_to_precitions(head_key, newRule)                
        else:
            if check_if_body_in_training_set(rule, training_set_intervals_with_triple_index):
                head_key = rule.get_head().get_object() if predict_for_object else rule.get_head().get_subject()
                add_to_precitions(head_key, rule)

    # sort the order of the rules for each prediction based on the new confidence values
    for prediction, rules in predictions_rules_list_aeIOU.items():
        rules.sort(key=lambda r: r.get_confidence(), reverse=True)
    for prediction, rules in predictions_rules_list_TAC.items():
        rules.sort(key=lambda r: r.get_confidence(), reverse=True)

    # sort the list of predictions based on the new confidence values
    sorted_predictions_aeIOU = sorted(
        predictions_rules_list_aeIOU.items(),
        # create a sorted list of rule confidences for each prediction
        key=lambda item: [
            rule.get_confidence() for rule in sorted(item[1], key=lambda r: r.get_confidence(), reverse=True)
        ],  
        # higher confidences come first
        reverse=True
    )

    sorted_predictions_TAC = sorted(
        predictions_rules_list_TAC.items(),
        # create a sorted list of rule confidences for each prediction
        key=lambda item: [
            rule.get_confidence() for rule in sorted(item[1], key=lambda r: r.get_confidence(), reverse=True)
        ],  
        # higher confidences come first
        reverse=True
    )

    return sorted_predictions_aeIOU[:max], sorted_predictions_TAC[:max]


## time prediction
def process_test_relation_time_prediction(test_relation: Quintuple, rules_with_predicate_index: dict, training_set_intervals_with_triple_index: dict, dataset: str) -> Optional[Tuple[Quintuple, int, int]]:
    rules = find_rules_for_test_relation_in_head_time_prediction(rules_with_predicate_index, test_relation)
    filtered_rules = filter_rules_head_in_training_set(rules, training_set_intervals_with_triple_index)

    result = get_time_prediction_predictions(filtered_rules, training_set_intervals_with_triple_index, dataset)
    if result is not None:
        prediction_start, prediction_end = result    
        return (
            test_relation, prediction_start, prediction_end
        )
    else:
        return None


def make_predictions_for_time_prediction(test_set: List[Quintuple], train_set: List[Quintuple], rules: List[Rule], dataset: str):
    # index rules by predicate
    rules_with_predicate_index = defaultdict(list)
    for rule in rules:
        rules_with_predicate_index[rule.get_head().get_predicate()].append(rule)

    # index training set by triple
    training_set_intervals_with_triple_index = defaultdict(list)
    for quintuple in train_set:
        triple_key = (quintuple.get_subject(), quintuple.get_predicate(), quintuple.get_object())
        start_time, end_time = quintuple.get_interval()
        training_set_intervals_with_triple_index[triple_key].append((start_time, end_time))

    time_interval_predictions = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Prepare tasks
        future_to_test_relation = {
            executor.submit(
                process_test_relation_time_prediction, 
                test_relation,
                rules_with_predicate_index,
                training_set_intervals_with_triple_index,
                dataset
            ): test_relation
            for test_relation in test_set
        }

        # initialize a counter and a lock for thread-safe updates
        processed_count = 0
        lock = Lock()
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_test_relation):
            try:
                result = future.result()
                if result is not None:
                    test_relation, time_prediction_start, time_prediction_end = result

                    with lock:
                        time_interval_predictions[test_relation] = (time_prediction_start, time_prediction_end)
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"Completed processing test facts: {processed_count} / {len(test_set)}")
            except Exception as e:
                print(f"Error processing test relation: {e}")

    return time_interval_predictions


def get_time_prediction_predictions(rules: List[Rule], training_set_intervals_with_triple_index: dict, dataset: str) -> Optional[Tuple[int, int]]:
    for rule in rules:
        groundings = get_grounding_for_variables_time_prediction(rule, dataset)
        for grounding in groundings:
            newRule = Rule(rule.get_head(), rule.get_body(), rule.get_body_coverage(), rule.get_support(), rule.get_confidence())
            for variable, value in grounding.items():
                newRule = substitute_variable_in_rule(newRule, value, variable)

            if check_if_body_in_training_set(newRule, training_set_intervals_with_triple_index):
                intervals = []
                for body_atom in newRule.get_body():
                    triple = (body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object())
                    intervals.append(training_set_intervals_with_triple_index[triple])
                intervals = [[(int(start), int(end)) for start, end in sublist] for sublist in intervals]
                result = find_predictions(intervals)

                # Find overlap or handle no overlap
                if result is not None:
                    prediction_start, prediction_end = result
                    return prediction_start, prediction_end

    return None


def find_predictions(intervals: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    if len(intervals) == 1:
        if len(intervals[0]) == 1:
            return intervals[0][0]
        else:
            return random.choice(intervals[0])
    
    # Start with the first list as the initial common overlap
    common_overlap = intervals[0]
    # Iterate through the rest of the interval lists
    for i in range(1, len(intervals)):
        result = find_overlap_between_two(common_overlap, intervals[i])
        if result is None:
            return None
        else:
            common_overlap = [result]
    
    return common_overlap[0]


def find_overlap_between_two(intervals1: List[Tuple[int, int]], intervals2: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for start1, end1 in intervals1:
        for start2, end2 in intervals2:
            # print(start1, end1, start2, end2)
            # Find the overlap between the two intervals
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            # If there is an overlap
            if overlap_start <= overlap_end:
                return overlap_start, overlap_end

    return None


## temporal interval measures
def get_aeIOU_factor(rule: Rule, test_relation: Quintuple, training_set_intervals_with_triple_index: dict) -> float:
    aeIOU_factor = 0
    number_of_intervals = 0
    start1_str, end1_str = test_relation.get_interval()
    start1 = int(start1_str)
    end1 = int(end1_str)
    for body_atom in rule.get_body():
        # if number_of_intervals == 0:
        #     print(training_set_intervals_with_triple_index[(body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object())], body_atom.__str__())
        for (start2_str, end2_str) in training_set_intervals_with_triple_index[(body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object())]:
            aeIOU_factor += calculate_aeIOU(start1, end1, int(start2_str), int(end2_str))
            number_of_intervals += 1

    if number_of_intervals == 0:
        print(training_set_intervals_with_triple_index[(body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object())], body_atom.__str__())

    return aeIOU_factor / number_of_intervals


def calculate_aeIOU(start1: int, end1: int, start2: int, end2: int) -> float:
    # calculate intersection and union volumes
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_vol = max(0, intersection_end - intersection_start + 1)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_vol = union_end - union_start + 1


    if union_vol == 0:
        print(start1, end1, start2, end2)

    # calculate aeIOU
    return max(1, intersection_vol) / union_vol


def get_TAC_factor(rule: Rule, test_relation: Quintuple, training_set_intervals_with_triple_index: dict) -> float:
    TAC_factor = 0
    number_of_intervals = 0
    start1_str, end1_str = test_relation.get_interval()
    start1 = int(start1_str)
    end1 = int(end1_str)

    for body_atom in rule.get_body():
        for (start2_str, end2_str) in training_set_intervals_with_triple_index[(body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object())]:
            TAC_factor += calculate_TAC(start1, end1, int(start2_str), int(end2_str))
            number_of_intervals += 1

    if number_of_intervals == 0:
        print(training_set_intervals_with_triple_index[(body_atom.get_subject(), body_atom.get_predicate(), body_atom.get_object())], body_atom.__str__())
    
    return TAC_factor / number_of_intervals


def calculate_TAC(start1: int, end1: int, start2: int, end2: int) -> float:
    # calculate absolute differences start and end times
    start_diff = abs(start1 - start2)
    end_diff = abs(end1 - end2)
    
    # calculate TAC
    return 0.5 * (1 / (1 + start_diff) + 1 / (1 + end_diff))


def write_link_predictions_to_file(output_file: str, test_set: List[Quintuple], triple_subject_groundings: Dict[str, List[Rule]], triple_object_groundings: Dict[str, List[Rule]]):
    with open(output_file, 'w') as file:
        for test_relation in test_set:
            file.write(f"{test_relation.get_subject()} {test_relation.get_predicate()} {test_relation.get_object()}\n")

            subject_groundings = triple_subject_groundings[test_relation]
            subject_line = "Subjects: "
            for entity_id, rules in subject_groundings:
                confidence = get_confidence(rules)
                subject_line += f"{entity_id} {confidence} \t"
            file.write(subject_line + "\n")
            
            object_groundings = triple_object_groundings[test_relation]
            object_line = "Objects: "
            for entity_id, rules in object_groundings:
                confidence = get_confidence(rules)
                object_line += f"{entity_id} {confidence} \t"
            file.write(object_line + "\n")


def write_time_predictions_to_file(output_file: str, test_set: List[Quintuple], time_predictions: Dict[str, Tuple[int, int]]):
    with open(output_file, 'w') as file:
        file.write(f"Number of test facts:\t{str(len(time_predictions))}\n")
        for test_relation in test_set:
            if test_relation in time_predictions:
                file.write(f"{test_relation.get_subject()} {test_relation.get_predicate()} {test_relation.get_object()} \t {int(test_relation.get_interval()[0])} \t {int(test_relation.get_interval()[1])}\n")
                start_time, end_time = time_predictions[test_relation]
                subject_line = f"Time prediction:\t{str(start_time)}\t {str(end_time)}\n"
                file.write(subject_line)
