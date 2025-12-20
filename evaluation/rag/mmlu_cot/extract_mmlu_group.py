import os
import yaml


def get_mmlu_group_subjects():
    task_dir = '/home/rulin/RAG-evaluation-harnesses/lm_eval/tasks/mmlu/default'

    groups = {
        'humanities': [],
        'other': [],
        'social_sciences': [],
        'stem': [],
    }

    subject_to_group = {}

    for filename in os.listdir(task_dir):
        if filename.endswith(".yaml") and filename.startswith('mmlu'):
            filepath = os.path.join(task_dir, filename)
            with open(filepath, 'r') as fin:
                data = yaml.safe_load(fin)
            
            group = data['group'].replace('mmlu_', '')
            task = data['task'].replace('mmlu_', '')

            groups[group].append(task)

            subject_to_group[task] = group

    print(groups)

    return groups, subject_to_group
