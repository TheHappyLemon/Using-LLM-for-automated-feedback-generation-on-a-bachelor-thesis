from __future__ import annotations
from src.code.parsing.old.EvaluationRow import EvaluationRow
import csv
import operator
from itertools import groupby
from collections import Counter
import logging

logger = logging.getLogger(__name__)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

class EvaluationDataset:

    HEADER_data = ['Nr', 'Significance', 'State-of-the-art', 'Gap', 'Problem', 'References', 'Purpose', 'Intention', 'Structure_goal', 'Congruence', 'Outlook', 'Quantity', 'Completeness', 'Format', 'Structure_tasks', 'Clarity', 'Relevance', 'Chapters', 'Description', 'Structure_aftertasks']
    HEADER_feedback = ['Nr', 'Question'] # dynamicaly appended with 'author_value', 'author_feedback'.

    questions = [
        ("Significance",     "beforeGoal"),
        ("State_of_the_art", "beforeGoal"),
        ("Gap",              "beforeGoal"),
        ("Problem",          "beforeGoal"),
        ("References",       "beforeGoal"),
        ("Purpose",          "goal"),
        ("Intention",        "goal"),
        ("Structure",        "goal"),
        ("Congruence",       "goal"),
        ("Outlook",          "tasks"),
        ("Quantity",         "tasks"),
        ("Completeness",     "tasks"),
        ("Format", 	         "tasks"),
        ("Structure",        "tasks"),
        ("Clarity",          "tasks"),
        ("Relevance",        "tasks"),
        ("Chapters",         "afterTasks"),
        ("Description",      "afterTasks"),
        ("Structure",        "afterTasks"),
    ]

    def __init__(self, author : str, iteration : int | None = None):
        self.rows: list[EvaluationRow] = []
        self.author = author
        self.iteration = iteration

    def append(self, row: EvaluationRow):
        self.rows.append(row)

    def load_from_csv(self, path : str):
        with open(path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader) # skip header
            
            for line_num, row in enumerate(reader, start=2): # start=2 as we skip header
                if not row or len(row) < 20:
                    logger.error(f"Skipping incomplete row {line_num}: {row}")
                    continue
                
                # Convert all values to int
                try:
                    values = list(map(int, row))
                except ValueError:
                    logger.error(f"Skipping invalid row {line_num} (non-integer values): {row}")
                    continue

                (
                    Nr,
                    Significance, State_of_the_art, Gap, Problem, References,
                    Purpose, Intention, Structure_goal, Congruence,
                    Outlook, Quantity, Completeness, Format, Structure_tasks, Clarity, Relevance,
                    Chapters, Description, Structure_after_tasks
                ) = values

                evaluation_row = EvaluationRow(self.author)
                evaluation_row.load_1(
                    Nr,
                    Significance, State_of_the_art, Gap, Problem, References,
                    Purpose, Intention, Structure_goal, Congruence,
                    Outlook, Quantity, Completeness, Format, Structure_tasks, Clarity, Relevance,
                    Chapters, Description, Structure_after_tasks
                )
                self.append(evaluation_row)

    def dump_to_csv(self, path : str):
        with open(path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=EvaluationDataset.HEADER_data)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "Nr": row.Nr,
                        "Significance": row.beforeGoal.Significance.value,
                        "State-of-the-art": row.beforeGoal.State_of_the_art.value,
                        "Gap": row.beforeGoal.Gap.value,
                        "Problem": row.beforeGoal.Problem.value,
                        "References": row.beforeGoal.References.value,
                        "Purpose": row.goal.Purpose.value,
                        "Intention": row.goal.Intention.value,
                        "Structure_goal": row.goal.Structure.value,
                        "Congruence": row.goal.Congruence.value,
                        "Outlook": row.tasks.Outlook.value,
                        "Quantity": row.tasks.Quantity.value, # type: ignore
                        "Completeness": row.tasks.Completeness.value,
                        "Format": row.tasks.Format.value,
                        "Structure_tasks": row.tasks.Structure.value,
                        "Clarity": row.tasks.Clarity.value,
                        "Relevance": row.tasks.Relevance.value,
                        "Chapters": row.afterTasks.Chapters.value,
                        "Description": row.afterTasks.Description.value,
                        "Structure_aftertasks": row.afterTasks.Structure.value
                    }
                )

    def to_bool(self, quantity_already_bool : bool = False):
        for r in self.rows:
            r.to_bool(quantity_already_bool)

    def print(self):
        for row in self.rows:
            print(row)
            input("...")

    @staticmethod
    def compute_metrics(
        baseline_ds: EvaluationDataset,
        predicted_ds: EvaluationDataset,
        path: str | None = None,
        base_average: str = 'macro'
    ):
        """
        Compute Precision, Recall, F1, Cohen's kappa,
        and MCC for given dataset against baseline dataset.
        """
        logger.info(f"Comparing {baseline_ds.author} to {predicted_ds.author}")

        header = [
            'Question',
            'Precision_macro',
            'Precision_weighted',
            'Precision_positive',
            'Precision_negative',
            'Recall_macro',
            'Recall_weighted',
            'Recall_positive',
            'Recall_negative',
            'F1_macro',
            'F1_weighted',
            'F1_positive',
            'F1_negative',
            'CohenKappa',
            'MCC'
        ]
        rows = []

        for label, attr_path in EvaluationDataset.questions:
            export_label = label
            if label == "Structure":
                export_label = "Structure_" + attr_path

            getter = operator.attrgetter(f"{attr_path}.{label}")

            y_true = [getter(row).value for row in baseline_ds.rows]
            y_pred = [getter(row).value for row in predicted_ds.rows]

            # precision
            precision_macro    = precision_score(y_true, y_pred, average='macro')
            precision_weighted = precision_score(y_true, y_pred, average='weighted')
            precision_positive = precision_score(y_true, y_pred, average='binary', pos_label=1)
            precision_negative = precision_score(y_true, y_pred, average='binary', pos_label=0)

            # recall
            recall_macro    = recall_score(y_true, y_pred, average='macro')
            recall_weighted = recall_score(y_true, y_pred, average='weighted')
            recall_positive = recall_score(y_true, y_pred, average='binary', pos_label=1)
            recall_negative = recall_score(y_true, y_pred, average='binary', pos_label=0)

            # F1
            F1_macro    = f1_score(y_true, y_pred, average='macro')
            F1_weighted = f1_score(y_true, y_pred, average='weighted')
            F1_positive = f1_score(y_true, y_pred, average='binary', pos_label=1)
            F1_negative = f1_score(y_true, y_pred, average='binary', pos_label=0)

            # others
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            rows.append({
                'Question': export_label,
                'Precision_macro': precision_macro,
                'Precision_weighted': precision_weighted,
                'Precision_positive': precision_positive,
                'Precision_negative': precision_negative,
                'Recall_macro': recall_macro,
                'Recall_weighted': recall_weighted,
                'Recall_positive': recall_positive,
                'Recall_negative': recall_negative,
                'F1_macro': F1_macro,
                'F1_weighted': F1_weighted,
                'F1_positive': F1_positive,
                'F1_negative': F1_negative,
                'CohenKappa': kappa,
                'MCC': mcc
            })

        # write CSV only if path is provided
        if path:
            with open(path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)

        return rows[0]

    @staticmethod
    def compute_metrics_total_average(baseline_ds : EvaluationDataset, predicted_datasets : list[EvaluationDataset], path : str, base_average : str = 'macro'):
        """
        Compute Precision, Recall, F1, Cohen's kappa,
        and MCC for given datasets against baseline dataset.
        Returns average value for all questions in all datasets
        """
        with open(path, mode='w', newline='', encoding='utf-8') as file:

            header = [
                'Model',
                'Precision_macro',
                'Precision_weighted',
                'Precision_positive',
                'Precision_negative',
                'Recall_macro',
                'Recall_weighted',
                'Recall_positive',
                'Recall_negative',
                'F1_macro',
                'F1_weighted',
                'F1_positive',
                'F1_negative',
                'CohenKappa',
                'MCC'
            ]

            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            rows = []

            for predicted_ds in predicted_datasets:

                y_true_total = []
                y_pred_total = []

                for label, attr_path in EvaluationDataset.questions:
                    getter = operator.attrgetter(f"{attr_path}.{label}")
                    
                    y_true = [getter(row).value for row in baseline_ds.rows]
                    y_pred = [getter(row).value for row in predicted_ds.rows]
                    y_true_total.extend(y_true)
                    y_pred_total.extend(y_pred)

                # precision
                precision_macro    = precision_score(y_true_total, y_pred_total, average='macro')
                precision_weighted = precision_score(y_true_total, y_pred_total, average='weighted')
                precision_positive = precision_score(y_true_total, y_pred_total, average='binary', pos_label=1)
                precision_negative = precision_score(y_true_total, y_pred_total, average='binary', pos_label=0)
                # recall
                recall_macro    = recall_score(y_true_total, y_pred_total, average='macro')
                recall_weighted = recall_score(y_true_total, y_pred_total, average='weighted')
                recall_positive = recall_score(y_true_total, y_pred_total, average='binary', pos_label=1)
                recall_negative = recall_score(y_true_total, y_pred_total, average='binary', pos_label=0)
                # F1
                F1_macro    = f1_score(y_true_total, y_pred_total, average='macro')
                F1_weighted = f1_score(y_true_total, y_pred_total, average='weighted')
                F1_positive = f1_score(y_true_total, y_pred_total, average='binary', pos_label=1)
                F1_negative = f1_score(y_true_total, y_pred_total, average='binary', pos_label=0)
                # others
                kappa = cohen_kappa_score(y_true_total, y_pred_total)
                mcc = matthews_corrcoef(y_true_total, y_pred_total)

                rows.append({
                    'Model' : predicted_ds.author,
                    'Precision_macro' : precision_macro,
                    'Precision_weighted' : precision_weighted,
                    'Precision_positive' : precision_positive,
                    'Precision_negative' : precision_negative,
                    'Recall_macro' : recall_macro,
                    'Recall_weighted' : recall_weighted,
                    'Recall_positive' : recall_positive,
                    'Recall_negative' : recall_negative,
                    'F1_macro' : F1_macro,
                    'F1_weighted' : F1_weighted,
                    'F1_positive' : F1_positive,
                    'F1_negative' : F1_negative,
                    'CohenKappa' : kappa,
                    'MCC' : mcc
                })
            writer.writerows(rows)

    @staticmethod 
    def compute_metrics_total_average_by_iterations(
        baseline_ds: EvaluationDataset,
        predicted_datasets: list[EvaluationDataset],
        path: str,
        base_average: str = 'macro'
    ):
        """
        Compute Precision, Recall, F1, Cohen's kappa, and MCC
        for predicted datasets against baseline dataset.

        New behavior:
        - Datasets with the same author are treated as multiple iterations
        of the same predictor.
        - All iterations for the same predictor are merged into one pooled
        prediction vector.
        - Metrics are computed once on the pooled vectors.
        """
        with open(path, mode='w', newline='', encoding='utf-8') as file:

            header = [
                'Model',
                'Iterations',
                'Precision_macro',
                'Precision_weighted',
                'Precision_positive',
                'Precision_negative',
                'Recall_macro',
                'Recall_weighted',
                'Recall_positive',
                'Recall_negative',
                'F1_macro',
                'F1_weighted',
                'F1_positive',
                'F1_negative',
                'CohenKappa',
                'MCC'
            ]

            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            rows = []

            # Group datasets by author
            grouped_datasets = {}
            for predicted_ds in predicted_datasets:
                if grouped_datasets.get(predicted_ds.author) is None:
                    grouped_datasets[predicted_ds.author] = []
                grouped_datasets[predicted_ds.author].append(predicted_ds)

            for author, datasets in grouped_datasets.items():
                y_true_total = []
                y_pred_total = []

                for predicted_ds in datasets:
                    for label, attr_path in EvaluationDataset.questions:
                        getter = operator.attrgetter(f"{attr_path}.{label}")

                        y_true = [getter(row).value for row in baseline_ds.rows]
                        y_pred = [getter(row).value for row in predicted_ds.rows]

                        y_true_total.extend(y_true)
                        y_pred_total.extend(y_pred)

                # precision
                precision_macro    = precision_score(y_true_total, y_pred_total, average='macro')
                precision_weighted = precision_score(y_true_total, y_pred_total, average='weighted')
                precision_positive = precision_score(y_true_total, y_pred_total, average='binary', pos_label=1)
                precision_negative = precision_score(y_true_total, y_pred_total, average='binary', pos_label=0)

                # recall
                recall_macro    = recall_score(y_true_total, y_pred_total, average='macro')
                recall_weighted = recall_score(y_true_total, y_pred_total, average='weighted')
                recall_positive = recall_score(y_true_total, y_pred_total, average='binary', pos_label=1)
                recall_negative = recall_score(y_true_total, y_pred_total, average='binary', pos_label=0)

                # F1
                F1_macro    = f1_score(y_true_total, y_pred_total, average='macro')
                F1_weighted = f1_score(y_true_total, y_pred_total, average='weighted')
                F1_positive = f1_score(y_true_total, y_pred_total, average='binary', pos_label=1)
                F1_negative = f1_score(y_true_total, y_pred_total, average='binary', pos_label=0)

                # others
                kappa = cohen_kappa_score(y_true_total, y_pred_total)
                mcc = matthews_corrcoef(y_true_total, y_pred_total)

                rows.append({
                    'Model': author,
                    'Iterations': len(datasets),
                    'Precision_macro': precision_macro,
                    'Precision_weighted': precision_weighted,
                    'Precision_positive': precision_positive,
                    'Precision_negative': precision_negative,
                    'Recall_macro': recall_macro,
                    'Recall_weighted': recall_weighted,
                    'Recall_positive': recall_positive,
                    'Recall_negative': recall_negative,
                    'F1_macro': F1_macro,
                    'F1_weighted': F1_weighted,
                    'F1_positive': F1_positive,
                    'F1_negative': F1_negative,
                    'CohenKappa': kappa,
                    'MCC': mcc
                })

            writer.writerows(rows)

    @staticmethod
    def compute_metrics_by_question_pooled(
        baseline_ds: EvaluationDataset,
        predicted_datasets: list[EvaluationDataset],
        path: str,
        base_average: str = 'macro'
    ):
        """
        Compute Precision, Recall, F1, Cohen's kappa, and MCC
        for predicted datasets against baseline dataset.

        Behavior:
        - Datasets with the same author are treated as multiple iterations
        of the same predictor.
        - For each question, all iterations for the same predictor are pooled
        into one large prediction vector.
        - Metrics are computed once per question on the pooled vectors.
        """
        with open(path, mode='w', newline='', encoding='utf-8') as file:

            header = [
                'Model',
                'Question',
                'Section',
                'Iterations',
                'Precision_macro',
                'Precision_weighted',
                'Precision_positive',
                'Precision_negative',
                'Recall_macro',
                'Recall_weighted',
                'Recall_positive',
                'Recall_negative',
                'F1_macro',
                'F1_weighted',
                'F1_positive',
                'F1_negative',
                'CohenKappa',
                'MCC'
            ]

            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            rows = []

            # Group datasets by author
            grouped_datasets = {}
            for predicted_ds in predicted_datasets:
                if grouped_datasets.get(predicted_ds.author) is None:
                    grouped_datasets[predicted_ds.author] = []
                grouped_datasets[predicted_ds.author].append(predicted_ds)

            for author, datasets in grouped_datasets.items():
                for label, attr_path in EvaluationDataset.questions:
                    export_label = label
                    if label == "Structure":
                        export_label = "Structure_" + attr_path
                    getter = operator.attrgetter(f"{attr_path}.{label}")

                    y_true_total = []
                    y_pred_total = []

                    for predicted_ds in datasets:
                        y_true = [getter(row).value for row in baseline_ds.rows]
                        y_pred = [getter(row).value for row in predicted_ds.rows]

                        y_true_total.extend(y_true)
                        y_pred_total.extend(y_pred)

                    # precision
                    precision_macro = precision_score(y_true_total, y_pred_total, average='macro', zero_division=0)
                    precision_weighted = precision_score(y_true_total, y_pred_total, average='weighted', zero_division=0)
                    precision_positive = precision_score(y_true_total, y_pred_total, average='binary', pos_label=1, zero_division=0)
                    precision_negative = precision_score(y_true_total, y_pred_total, average='binary', pos_label=0, zero_division=0)
                    # recall
                    recall_macro = recall_score(y_true_total, y_pred_total, average='macro', zero_division=0)
                    recall_weighted = recall_score(y_true_total, y_pred_total, average='weighted', zero_division=0)
                    recall_positive = recall_score(y_true_total, y_pred_total, average='binary', pos_label=1, zero_division=0)
                    recall_negative = recall_score(y_true_total, y_pred_total, average='binary', pos_label=0, zero_division=0)
                    # F1
                    F1_macro = f1_score(y_true_total, y_pred_total, average='macro', zero_division=0)
                    F1_weighted = f1_score(y_true_total, y_pred_total, average='weighted', zero_division=0)
                    F1_positive = f1_score(y_true_total, y_pred_total, average='binary', pos_label=1, zero_division=0)
                    F1_negative = f1_score(y_true_total, y_pred_total, average='binary', pos_label=0, zero_division=0)

                    # others
                    kappa = cohen_kappa_score(y_true_total, y_pred_total)
                    mcc = matthews_corrcoef(y_true_total, y_pred_total)

                    rows.append({
                        'Model': author,
                        'Question': export_label,
                        'Section': attr_path,
                        'Iterations': len(datasets),
                        'Precision_macro': precision_macro,
                        'Precision_weighted': precision_weighted,
                        'Precision_positive': precision_positive,
                        'Precision_negative': precision_negative,
                        'Recall_macro': recall_macro,
                        'Recall_weighted': recall_weighted,
                        'Recall_positive': recall_positive,
                        'Recall_negative': recall_negative,
                        'F1_macro': F1_macro,
                        'F1_weighted': F1_weighted,
                        'F1_positive': F1_positive,
                        'F1_negative': F1_negative,
                        'CohenKappa': kappa,
                        'MCC': mcc
                    })

            writer.writerows(rows)

    @staticmethod
    def dump_to_csv_feedback(path : str, datasets : list[EvaluationDataset]):

        with open(path, mode='w', newline='', encoding='utf-8') as file:
            header = EvaluationDataset.HEADER_feedback.copy()
            for ds in datasets:
                header.append(ds.author)
                if ds.author != "human":
                    header.append(f"{ds.author}_feedback")
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

            # Here we have a loop for 64 entries (64 texts)
            for i in range(len(datasets[0].rows)):
                Nr = datasets[0].rows[i].Nr
                # Here we have a loop for 20 questions.
                for question, path in EvaluationDataset.questions:
                    row = {} # entry to write to CSV
                    for ds in datasets:
                        row['Nr'] = Nr
                        row['Question'] = question
                        
                        evaluation_row = None
                        for r in ds.rows:
                            if r.Nr == Nr:
                                evaluation_row = r
                                break 
                       
                        row[ds.author] =  getattr(getattr(getattr(evaluation_row, path), question), "value")
                        if ds.author != "human":
                            row[f"{ds.author}_feedback"] = getattr(getattr(getattr(evaluation_row, path), question), "feedback")
                    writer.writerow(row)
    
