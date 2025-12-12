import os
import json

def get_classifier_lines(classifier_path, enabled_classifiers='ALL'):
    with open(classifier_path, 'r') as f:
        data = json.load(f)
    is_enabled = lambda _: True
    if enabled_classifiers != 'ALL':
        classifiers_enabled_list = tuple(map(lambda x: x.lower().strip(), enabled_classifiers.split(',')))
        is_enabled = lambda classifier: classifier.lower().strip() in classifiers_enabled_list
    return "\n".join([f"{classifier['Classifier']}: {classifier['Definition']}{(' - Specific Items of Interest: ' + classifier['Items of Interest']) if classifier['Items of Interest'] and len(classifier['Items of Interest']) > 0 else ''}" for classifier in data if is_enabled(classifier['Classifier'])])