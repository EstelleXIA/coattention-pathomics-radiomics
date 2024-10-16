from sksurv.metrics import concordance_index_censored
import numpy as np
from typing import List
import os
import json


def generate_patient_task_codebook(tasks: List) -> dict:
    all_patients_dict = dict()
    for task_name in sorted(tasks):
        # dataset split, combining patients in all tasks
        json_path = f"./survival_modeling/splits/json_{task_name.lower()}/"
        with open(os.path.join(json_path, f"fold_0.json"), "r") as f:
            task_patients_json = json.load(f)
        patient_names = task_patients_json["train"] + task_patients_json["val"]
        task_dict = dict(zip(patient_names, [task_name] * len(patient_names)))
        all_patients_dict.update(task_dict)
    return all_patients_dict


def cal_task_metrics(tasks: List, patient_id: List, censorships: np.ndarray,
                     event_times: np.ndarray, risk_scores: np.ndarray, codebook_dict: dict):
    all_c_indexes = []
    all_c_indexes_dict = {}
    for task in tasks:
        patient_indicator = []
        for patient in patient_id:
            task_patient = codebook_dict[patient]
            if task_patient == task:
                patient_indicator.append(True)
            else:
                patient_indicator.append(False)

        selected_censorship = censorships[np.array(patient_indicator)]
        selected_event_times = event_times[np.array(patient_indicator)]
        selected_risk_scores = risk_scores[np.array(patient_indicator)]

        c_index_task = concordance_index_censored((1 - selected_censorship).astype(bool),
                                                  selected_event_times, selected_risk_scores,
                                                  tied_tol=1e-08)[0]
        all_c_indexes_dict[task] = c_index_task
        all_c_indexes.append(c_index_task)
    average_c_index = np.mean(np.array(all_c_indexes))

    return all_c_indexes_dict, average_c_index

