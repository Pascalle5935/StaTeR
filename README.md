# StaTeR: Static to Temporal Rule learning
- created for a Master Thesis at the University Utrecht, 2024.

# Before running the pipeline of main.py:
- set up a blazegraph server
    - download the blazegraph.jar file from https://github.com/blazegraph/database/releases/tag/BLAZEGRAPH_2_1_6_RC
    - run the file in the terminal where the jar file is located
    - this will set up a connection with blazegraph on local host
- create a turtle file of the dataset you would like to use and put it in /turtle_files. Wikidata12k and YAGO11k are provided.
- run a symbolic static rule learner to generate static rules for the dataset you would like to use
    - ensure that this file contains unique static files
    - output should be of the form:
        {body_caoverage} {support} {confidence} {h(X,Y) <= b(X,A), ... b(A,Y)} (or partially grounded)
    - AnyBURL rules are provided.

# A pipeline is created in the main.py file:
First, ensure that:
- datasets are present in the data_clean file, subfolders and files will need the '_clean' extension for the pipeline to work. Wikidata12k clean and YAGO11k clean are provided.
- static rules are present. AnyBURL rules are provided.
- blazegraph connection is open.

For confidence calculation, run:
python main.py --dataset {dataset} --learning_time {learning_time} --threshold {threshold} --quality_measure_confidence {quality_measure_confidence} --calculate_confidence
- ensure that:
    - there are files with static rules for the combination of {dataset} and {learning_time}.
    - threshold between 0 and 1.
    - quality_measure_confidence one of: naive, naive_overlap, same_interval, intersection_over_union, temporal_alignment_coefficient.
- output will be the file /confidence_calcualtions/{dataset}-{learning_time}-{quality_measure_confidene}-{threshold} with temporal confidences.

For link prediction, run:
python main.py --dataset {dataset} --learning_time {learning_time} --threshold {threshold} --quality_measure_confidence {quality_measure_confidence} --link_prediction
- ensure that:
    - there are confidence calculation files for the combination of {dataset}, {learning_time}, {quality_measure_confidence}, and {threshold}.
- output will be the files:
    - /link-predictions/{dataset}-{learning_time}-link-predictions-{quality_measure_confidene}-{threshold}-{aeIOU}
    - /link-predictions/{dataset}-{learning_time}-link-predictions-{quality_measure_confidene}-{threshold}-{TAC}
    - with the top k predictions for each link prediction task.

For time prediction, run:
python main.py --dataset {dataset} --learning_time {learning_time} --threshold {threshold} --quality_measure_confidence {quality_measure_confidence} --time_prediction
- ensure that:
    - there are confidence calculation files for the combination of {dataset}, {learning_time}, {quality_measure_confidence}, and {threshold}.
- output will be the file /time-predictions/{dataset}-{learning_time}-time-predictions-{quality_measure_confidene}-{threshold} with a predicted time interval for each time prediction task.

For link prediction evaluation, run:
python main.py --dataset {dataset} --learning_time {learning_time} --threshold {threshold} --link_prediction
- ensure that there are link prediction files for every quality measure.
- output will be the file /evaluation/{dataset}-{learning_time}-link-predictions-{threshold} containing the evaluation metrics hits@1, hits@10, MRR for all scenarios.

For time prediction evaluation, run:
python main.py --dataset {dataset} --learning_time {learning_time} --threshold {threshold} --time_prediction
- output will be the file /evaluation/{dataset}-{learning_time}-time-predictions-{threshold} containing the evaluation metrics aeIOU and TAC for all scenarios.