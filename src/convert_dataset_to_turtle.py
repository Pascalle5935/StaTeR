from generate_file_names import (
    generate_file_name_training_set,
    generate_file_name_output_turtle
)

import rdflib

WIKIDATA = rdflib.Namespace("http://wikidata.org/entity/")
PROP_WIKI = rdflib.Namespace("http://wikidata.org/prop/direct/")
YAGO = rdflib.Namespace("http://yago-knowledge.org/resource/")
PROP_YAGO = rdflib.Namespace("http://yago-knowledge.org/prop/direct/")
TIME = rdflib.Namespace("http://www.w3.org/2006/time#")


def create_rdf_graph_wikidata(file_path):
    """
    Creates an RDF graph for static wikidata12k data.
    """

    g = rdflib.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            subject, predicate, object = line.strip().split('\t')
            s_uri = WIKIDATA[subject]
            p_uri = PROP_WIKI[predicate]
            o_uri = WIKIDATA[object]
            g.add((s_uri, p_uri, o_uri))

    return g


def create_rdf_graph_wikidata_temporal(file_path):
    """
    Creates an RDF graph for temporal wikidata12k data.
    """

    g = rdflib.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            subject, predicate, object, start_date, end_date = line.strip().split('\t')
            s_uri = WIKIDATA[subject]
            p_uri = PROP_WIKI[predicate]
            o_uri = WIKIDATA[object]

            reified_statement = rdflib.URIRef(f"urn:uuid:{str(rdflib.BNode())}")
            g.add((reified_statement, rdflib.RDF.subject, s_uri))
            g.add((reified_statement, rdflib.RDF.predicate, p_uri))
            g.add((reified_statement, rdflib.RDF.object, o_uri))
            
            start_literal = rdflib.Literal(start_date, datatype=rdflib.XSD.date)
            g.add((reified_statement, TIME.hasStart, start_literal))

            end_literal = rdflib.Literal(end_date, datatype=rdflib.XSD.date)
            g.add((reified_statement, TIME.hasEnd, end_literal))

    return g


def create_rdf_graph_yago(file_path):
    """
    Creates an RDF graph for static yago11k data.
    """

    g = rdflib.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            subject, predicate, object = line.strip().split('\t')
            s_uri = YAGO[subject]
            p_uri = PROP_YAGO[predicate]
            o_uri = YAGO[object]
            g.add((s_uri, p_uri, o_uri))

    return g


def create_rdf_graph_yago_temporal(file_path):
    """
    Creates an RDF graph for temporal yago11k data.
    """
    g = rdflib.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            subject, predicate, object, start_date, end_date = line.strip().split('\t')
            s_uri = YAGO[subject]
            p_uri = PROP_YAGO[predicate]
            o_uri = YAGO[object]

            reified_statement = rdflib.URIRef(f"urn:uuid:{str(rdflib.BNode())}")
            g.add((reified_statement, rdflib.RDF.subject, s_uri))
            g.add((reified_statement, rdflib.RDF.predicate, p_uri))
            g.add((reified_statement, rdflib.RDF.object, o_uri))

            start_literal = rdflib.Literal(start_date, datatype=rdflib.XSD.date)
            g.add((reified_statement, TIME.hasStart, start_literal))
         
            end_literal = rdflib.Literal(end_date, datatype=rdflib.XSD.date)
            g.add((reified_statement, TIME.hasEnd, end_literal))

    return g


def convert_dataset_to_turle_main(dataset: str):
    input_file_train_set = generate_file_name_training_set(dataset)
    output_file_train_set_turtle = generate_file_name_output_turtle(dataset)

    if dataset == 'wikidata12k':
        graph = create_rdf_graph_wikidata_temporal(input_file_train_set)
    elif dataset == 'yago11k':
        graph = create_rdf_graph_yago_temporal(input_file_train_set)
    else:
        raise ValueError("Unsupported dataset")

    graph.serialize(destination=output_file_train_set_turtle, format='turtle')
    print(f"RDF data saved to {output_file_train_set_turtle}")
