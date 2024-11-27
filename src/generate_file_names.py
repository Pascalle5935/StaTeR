def generate_file_name_anyburl_rules(dataset: str, learning_time: str) -> str:
        return f'../anyBURL_rules/{dataset}-{learning_time}'


def generate_file_name_calculate_confidence(dataset: str, learning_time: str, quality_measure_confidence: str, threshold: str) -> str:
    return f'../confidence_calculations/{dataset}-{learning_time}-{quality_measure_confidence}-{threshold}'


def generate_file_name_output_link_predictions(dataset: str, learning_time: str, quality_measure_confidence: str, threshold: str) -> str:
    return f'../link_predictions/{dataset}-{learning_time}-link-predictions-{quality_measure_confidence}-{threshold}'


def generate_file_name_output_time_predictions(dataset: str, learning_time: str, quality_measure_confidence: str, threshold: str) -> str:
    return f'../time_predictions/{dataset}-{learning_time}-time-predictions-{quality_measure_confidence}-{threshold}'     


def generate_file_name_for_dataset(dataset: str, file_type: str) -> str:
    return f'../data_clean/{dataset}_clean/{file_type}.txt'


def generate_file_name_training_set(dataset: str) -> str:
    return f'../data_clean/{dataset}_clean/train_clean.txt'


def generate_file_name_test_set(dataset: str) -> str:
    return f'../data_clean/{dataset}_clean/test_clean.txt'     


def generate_file_name_output_turtle(dataset: str) -> str:
    return f'../turtle_files/{dataset}.ttl'


def generate_file_name_orignal_training_set(dataset: str) -> str:
    return f'../data_original/{dataset}/train.txt'


def generate_file_name_unique_static_training_set(dataset: str) -> str:
    return f'../data_clean/{dataset}_clean/train_unique.txt'
