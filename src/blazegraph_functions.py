from SPARQLWrapper import SPARQLWrapper, JSON, POST
from model import *
import requests
from itertools import combinations

SPARQL_ENDPOINT = "http://localhost:9999/blazegraph/namespace/kb/sparql"
sparql = SPARQLWrapper(SPARQL_ENDPOINT)


## confidence functions
def compute_temporal_body_coverage(body: List[Triple], dataset: str) -> int:
    """
    Computes the temporal body coverage for a rule.

    Parameters:
    - body (List[Triple]): The body of the triple.
    - dataset (str): The dataset used.

    Returns:
    - int: The count of instances where all body triples in the rule exist.
    """
    # generate triple statements
    triple_statements = get_triple_statements(body, 'withoutTime')

    # build the complete SPARQL query
    query = f"""
    {get_prefixes(dataset)}
    
    SELECT (COUNT(DISTINCT *) AS ?count) WHERE {{
        {' '.join(triple_statements)}
    }}
    """

    return execute_sparql_count_query(query)


def compute_temporal_support_naive(head: Triple, body: List[Triple], dataset: str) -> int:
    """
    Computes the temporal support for a rule using naive temporal confidence quality measure.
    
    Parameters:
    - head (Triple): The head of the rule.
    - body (List[Triple]): A list of Triple objects representing the rule body.
    - dataset (str): The dataset used.

    Returns:
    - int: The naive support of the rule.
    """
    # generate triple statements
    triple_statements = get_triple_statements([head] + body, 'withoutTime')

    # build the complete SPARQL query
    query = f"""
    {get_prefixes(dataset)}
    
    SELECT (COUNT(DISTINCT *) AS ?count) WHERE {{
        {' '.join(triple_statements)}
    }}
    """

    return execute_sparql_count_query(query)


def compute_temporal_support_naive_overlap(head: Triple, body: List[Triple], dataset: str) -> int:
    """
    Computes the temporal support for a rule using naive temporal overlap confidence quality measure.

    Parameters:
    - head (Triple): The head of the rule.
    - body (List[Triple]): A list of Triple objects representing the rule body.
    - dataset (str): The dataset used.

    Returns:
    - int: The naive overlap support of the rule.
    """
    # generate triple statements
    all_triples = [head] + body
    triple_statements = get_triple_statements(all_triples, 'withTime')
    temporal_constraints = []
    
    # generate pairwise temporal constraints
    n = len(all_triples)
    for i in range(n):
        for j in range(i + 1, n):
            temporal_constraints.append(f"(?s{i} <= ?e{j} && ?s{j} <= ?e{i})")

    # combine the temporal constraints
    temporal_filter = "FILTER (" + " && ".join(temporal_constraints) + ")"

    # build the complete SPARQL query
    query = f"""
    {get_prefixes(dataset)}
    
    SELECT (COUNT(DISTINCT *) AS ?count) WHERE {{
        {' '.join(triple_statements)}
        {temporal_filter}
    }}
    """

    return execute_sparql_count_query(query)


def compute_temporal_support_same_interval(head: Triple, body: Triple, dataset: str) -> int:
    """
    Computes the temporal support for a rule using same interval confidence quality measure.

    Parameters:
    - head (Triple): The head of the rule.
    - body (List[Triple]): A list of Triple objects representing the rule body.
    - dataset (str): The dataset used.

    Returns:
    - int: The same interval support of the rule.
    """
    # generate triple statements
    triple_statements = get_triple_statements([head] + body, 'sameTime')

    # build the complete SPARQL query
    query = f"""
    {get_prefixes(dataset)}
    
    SELECT (COUNT(DISTINCT *) AS ?count) WHERE {{
        {' '.join(triple_statements)}
    }}
    """

    return execute_sparql_count_query(query)


def compute_temporal_support_intersection_over_union(head, body: List[Triple], dataset: str) -> float:
    """
    Computes the temporal support for a rule using intersection over union confidence quality measure.
    
    Parameters:
    - head (Triple): The head of the rule.
    - body (List[Triple]): A list of Triple objects representing the body of the rule.
    - dataset (str): The dataset used.

    Returns:
    - float: The intersection over union support of the rule.
    """
    # generate triple statements
    all_triples = [head] + body
    triple_statements = get_triple_statements(all_triples, 'withTime')
    
    # build the complete SPARQL query
    query = f"""
    {get_prefixes(dataset)}
    
    SELECT {' '.join([f'?s{i} ?e{i}' for i in range(len(all_triples))])} WHERE {{
        {' '.join(triple_statements)}
    }}
    """

    results = execute_sparql_query(query)
    if not results:
        return 0.0
    
    # generate all pair combinations
    pairs = list(combinations(range(len(all_triples)), 2))  
    
    overlap_sum = 0
    for result in results:
        for i, j in pairs:
            # get the start and end times for the current pair of instances
            s1, e1 = extract_year_from_date(result[f"s{i}"]['value']), extract_year_from_date(result[f"e{i}"]['value'])
            s2, e2 = extract_year_from_date(result[f"s{j}"]['value']), extract_year_from_date(result[f"e{j}"]['value'])
            
            # calculate the union
            latest_end = max(e1, e2)
            earliest_start = min(s1, s2)
            union = latest_end - earliest_start + 1
            
            # calculate the overlap
            beginning_overlap = max(s1, s2)
            ending_overlap = min(e1, e2)
            overlap = max(0, ending_overlap - beginning_overlap + 1)
            
            # calculate the overlap fraction
            overlap_fraction = overlap / union if union > 0 else 0
            overlap_sum += overlap_fraction

    # normalize by the number of pairs
    temporal_support = float(overlap_sum / len(pairs))
    
    return temporal_support


def compute_temporal_temporal_alignment_coefficient(head, body: List[Triple], dataset: str) -> float:
    """
    Computes the temporal support for a rule using temporal alignment coefficient confidence quality measure.
    
    Parameters:
    - head (Triple): The head of the rule.
    - body (List[Triple]): A list of Triple objects representing the body of the rule.
    - dataset (str): The dataset used.

    Returns:
    - float: The temporal alignment coefficient support of the rule.
    """
    # generate triple statements
    all_triples = [head] + body
    triple_statements = get_triple_statements(all_triples, 'withTime')
    
    # build the complete SPARQL query
    query = f"""
    {get_prefixes(dataset)}
    
    SELECT {' '.join([f'?s{i} ?e{i}' for i in range(len(all_triples))])} WHERE {{
        {' '.join(triple_statements)}
    }}
    """

    results = execute_sparql_query(query)
    if not results:
        return 0.0

    # generate all pair combinations
    pairs = list(combinations(range(len(all_triples)), 2))  

    tac_sum = 0
    for result in results:
        for i, j in pairs:
            # get the start and end times for the current pair of instances
            s1, e1 = extract_year_from_date(result[f"s{i}"]['value']), extract_year_from_date(result[f"e{i}"]['value'])
            s2, e2 = extract_year_from_date(result[f"s{j}"]['value']), extract_year_from_date(result[f"e{j}"]['value'])
            
            # calculate the TAC contribution for the pair (I_i, I_j)
            start_diff = abs(s1 - s2)
            end_diff = abs(e1 - e2)
            
            # calculate the TAC for this pair
            tac = 0.5 * (1 / (1 + start_diff) + 1 / (1 + end_diff))
            tac_sum += tac
    
    # normalize by the number of pairs
    temporal_support = float(tac_sum / len(pairs))

    return temporal_support


## prediction functions
def get_grounding_for_variables_link_prediction(rule: Rule, dataset: str, predict_for_object: bool):
    head: Quintuple = rule.get_head()
    body: List[Quintuple] = rule.get_body()
    variables: List[str] = retrieve_all_variables_rule(rule)

    select_vars = ' '.join([f"?{var.lower()}" for var in variables])

    query = f"""
    {get_prefixes(dataset)} \n
    """
    triple_statements = []

    # generate SPARQL statements for each triple
    for i, body_atom in enumerate(body):
        alias = f"?a{i}"
        triple_statements.append(
            f"""
            {alias} rdf:subject {convert_to_sql_query_part(body_atom.get_subject())} ;
                  rdf:predicate p:{body_atom.get_predicate()} ;
                  rdf:object {convert_to_sql_query_part(body_atom.get_object())} ;
                  time:hasStart ?s{i} ;
                  time:hasEnd ?e{i} .
            """
        )

    filter_not_exists = ""
    if atom_has_variable(head):
        # add the MINUS clause to exclude cases where the head exists in the training graph
        if predict_for_object:
            # Exclude cases where head.subject() head.predicate() ?y exists
            filter_not_exists = f"""
            FILTER NOT EXISTS {{
            {convert_to_sql_query_part(head.get_subject())} p:{head.get_predicate()} ?y .
            }}
            """
        else:
            # Exclude cases where ?x head.predicate() head.object() exists
            filter_not_exists = f"""
            FILTER NOT EXISTS {{
            ?x p:{head.get_predicate()} {convert_to_sql_query_part(head.get_object())} .
            }}
            """

    query += f"""SELECT DISTINCT {select_vars} WHERE {{
        {' '.join(triple_statements)}
        {filter_not_exists}
    }}
    """

    return execute_sparql_groundings_query(query, variables)


def get_grounding_for_variables_time_prediction(rule: Rule, dataset: str):
    body: List[Quintuple] = rule.get_body()
    variables: List[str] = retrieve_all_variables_rule(rule)

    select_vars = ' '.join([f"?{var.lower()}" for var in variables])

    query = f"""
    {get_prefixes(dataset)} \n
    """
    triple_statements = []

    # generate SPARQL statements for each triple
    for i, body_atom in enumerate(body):
        alias = f"?a{i}"
        triple_statements.append(
            f"""
            {alias} rdf:subject {convert_to_sql_query_part(body_atom.get_subject())} ;
                  rdf:predicate p:{body_atom.get_predicate()} ;
                  rdf:object {convert_to_sql_query_part(body_atom.get_object())} ;
                  time:hasStart ?s{i} ;
                  time:hasEnd ?e{i} .
            """
        )

    query += f"""SELECT DISTINCT {select_vars} WHERE {{
        {' '.join(triple_statements)}
    }}
    """

    return execute_sparql_groundings_query(query, variables)


## general functions
def clear_default_graph():
    """
    Clears the default graph in Blazegraph using SPARQLWrapper.
    """
    sparql.setQuery("CLEAR DEFAULT")
    sparql.setMethod(POST)
    sparql.setReturnFormat(JSON)

    try:
        sparql.query()
        print("Default graph cleared successfully.")
    except Exception as e:
        print(f"Failed to clear the default graph: {e}")


def load_data_to_blazegraph(train_set_turtle_file_name: str):
    """
    Loads the specified turtle file to blazegraph.
    """
    clear_default_graph()

    headers = {
        'Content-Type': 'application/x-turtle',
        "Cache-Control": "max-age=3600"
    }
    with open(train_set_turtle_file_name, 'rb') as f:
        response = requests.post(SPARQL_ENDPOINT, headers=headers, data=f)
        if response.status_code == 200:
            print(f"Turtle file {train_set_turtle_file_name} loaded successfully to blazegraph")
        else:
            print("Failed to load data", response.text)


def get_triple_statements(triples: List[Triple], option: str) -> str:
    """
    Generates the triple statements based on the provided option.

    Parameters:
    - triples (List[Triple]): The triples of the rule.
    - option (str): Parameter which decides what triple statements will be generated.

    Returns:
    - str: The triple statements for the query.
    """
    triple_statements = []

    if option == 'withoutTime':
        for i, triple in enumerate(triples):
            alias = f"?a{i}"
            triple_statements.append(
                f"""
                {alias} rdf:subject {convert_to_sql_query_part(triple.get_subject())} ;
                    rdf:predicate p:{triple.get_predicate()} ;
                    rdf:object {convert_to_sql_query_part(triple.get_object())} .
                """
            )
    elif option == 'withTime':
        for i, triple in enumerate(triples):
            alias = f"?a{i}"
            triple_statements.append(
                f"""
                {alias} rdf:subject {convert_to_sql_query_part(triple.get_subject())} ;
                    rdf:predicate p:{triple.get_predicate()} ;
                    rdf:object {convert_to_sql_query_part(triple.get_object())} ;
                    time:hasStart ?s{i} ;
                    time:hasEnd ?e{i} .
                """
            )
    elif option == 'sameTime':
        for i, triple in enumerate(triples):
            alias = f"?a{i}"
            triple_statements.append(
                f"""
                {alias} rdf:subject {convert_to_sql_query_part(triple.get_subject())} ;
                    rdf:predicate p:{triple.get_predicate()} ;
                    rdf:object {convert_to_sql_query_part(triple.get_object())} ;
                    time:hasStart ?s ;
                    time:hasEnd ?e .
                """
            )
    
    return triple_statements


def convert_to_sql_query_part(entity_id):
    """
    Function that converts the entity id to the correct SQL query part.
    """
    if entity_id.isalpha():
        return f"?{entity_id.lower()}"
    else:
        return f"e:{entity_id}"


def get_prefixes(dataset: str) -> str:
    """
    Function to get the prefixes based on the dataset.
    
    Parameters:
    - dataset (str): The name of the dataset (either "wikidata12k" or "yago11k").
    
    Returns:
    - str: The SPARQL prefixes for the given dataset.
    
    Raises:
    - ValueError: If an unsupported dataset name is provided.
    """
    if dataset == "wikidata12k":
        dataset_prefixes = """PREFIX p: <http://wikidata.org/prop/direct/>\nPREFIX e: <http://wikidata.org/entity/>"""
    elif dataset == "yago11k":
        dataset_prefixes = """PREFIX p: <http://yago-knowledge.org/prop/direct/>\nPREFIX e: <http://yago-knowledge.org/resource/>"""
    else:
        raise ValueError("Unsupported dataset. Choose 'wikidata12k' or 'yago11k'.")

    general_prefixes = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX time: <http://www.w3.org/2006/time#>"""
    
    return f"{dataset_prefixes}\n{general_prefixes}"


def execute_sparql_query(query):
    """
    Helper function that returns the result after a sparql query.
    """
    # sparql = get_sparql_wrapper()
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results and "results" in results and "bindings" in results["results"]:
            return results["results"]["bindings"]
        else:
            return []
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return []


def execute_sparql_count_query(query):
    """
    Returns the result after a count sparql query.
    """    
    # sparql = get_sparql_wrapper()
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results and "results" in results and "bindings" in results["results"]:
            count = int(results["results"]["bindings"][0]["count"]["value"])
            return count
        else:
            return 0
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return []
    

def execute_sparql_groundings_query(query: str, variables: List[str]):
    """
    Returns the result after a grounding sparql query.
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        # Execute the query and convert the result into a JSON format
        results = sparql.query().convert()
        
        # Check if the results contain the necessary data
        if results and "results" in results and "bindings" in results["results"]:
            groundings = []
            
            # Extract the grounding values for the specified variables
            for binding in results["results"]["bindings"]:
                grounding = {}
                for var in variables:
                    var_key = var.lower()
                    if var_key in binding:
                        # Extract the value after the last '/'
                        grounding[var] = binding[var_key]["value"].split("/")[-1]
                groundings.append(grounding)
            
            return groundings
        
        return []
    
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return []


def extract_year_from_date(date_string: str) -> int:
    """
    Extracts the year from a date string (e.g., '1977-01-01' -> 1977).

    Parameters:
    - date_string (str): The date string in 'YYYY-MM-DD' format.

    Returns:
    - int: The year extracted from the date string.
    """
    # Split the date by the hyphen and return the first part which is the year
    return int(date_string.split('-')[0])
