from blazegraph_functions import *
from model import *
from typing import List, Tuple

def convert_text_rule_to_objects_of_class_triple(rule: str) -> Tuple[Triple, List[Triple]]:
    """
    Reads a rule of the form: predicate(subject,object) <= predicate(subject,object), predicate(subject,object), ...,
    and converts it to a head Triple and body List[Triple] objects.

    Paramters:
    - rule (str): The rule string to be parsed.

    Returns:
    - Tuple[Triple, List[Triple]]: Tuple of the head Triple and the body List[Triple].
    """

    def convert_text_triple_to_object_of_class_triple(triple: str) -> Triple:
        """
        Converts a triple string into a Triple object.

        Parameters:
        - triple (str): The triple string of the form predicate(subject, object)
        """
        predicate, rest = triple.split('(')
        subject, object = rest.strip(')').split(',')

        return Triple(subject, predicate.strip(), object)
    
    parts = rule.split(' <= ')
    head_str, body_str = parts
    head = convert_text_triple_to_object_of_class_triple(head_str)
    body_triples_str = body_str.split(', ')
    body = [convert_text_triple_to_object_of_class_triple(triple_str) for triple_str in body_triples_str]

    return head, body


def process_rule(rule: str, dataset: str, function_computation_support):
    """
    General function that computes the support, body coverage, and confidence based on the provided function for support computation.

    Parameters:
    - rule (str): The textual rule.
    - dataset (str): The dataset to be used.
    - function_computation_support: Function to computate support of one of the temporal confidence quality measures.
    """
    head, body = convert_text_rule_to_objects_of_class_triple(rule)
    body_coverage = compute_temporal_body_coverage(body, dataset)
    support = function_computation_support(head, body, dataset)
    confidence = float(support) / body_coverage if body_coverage != 0 else 0
    # print(f"THREAD ID: {threading.get_ident()}, HEAD: {head.__str__()},  BODY COVERAGE: {body_coverage}, SUPPORT: {support}, CONFIDENCE: {confidence}" )

    return rule, body_coverage, support, confidence

# Wrapper functions for different support computation methods
def process_rule_naive(rule_with_index, dataset: str):
    return process_rule(rule_with_index, dataset, compute_temporal_support_naive)

def process_rule_naive_overlap(rule_with_index, dataset: str):
    return process_rule(rule_with_index, dataset, compute_temporal_support_naive_overlap)

def process_rule_same_interval(rule_with_index, dataset: str):
    return process_rule(rule_with_index, dataset, compute_temporal_support_same_interval)

def process_rule_intersection_over_union(rule_with_index, dataset: str):
    return process_rule(rule_with_index, dataset, compute_temporal_support_intersection_over_union)

def process_rule_temporal_alignment_coefficient(rule_with_index, dataset: str):
    return process_rule(rule_with_index, dataset, compute_temporal_temporal_alignment_coefficient)
