import gzip
import numpy as np
import pickle

from collections import defaultdict
from sklearn.metrics import average_precision_score

CLUTTER_RANGES = [(0, 18), (18, 33), (33, 53), (53, float("inf"))]
SCALE_RANGES = [
    (0, 0.06880213760855043),
    (0.06880213760855043, 0.13526567164179104),
    (0.13526567164179104, 0.2729876723163842),
    (0.2729876723163842, float("inf")),
]

class ILIAS:
    def __init__(self, dataset_dir, check_ids=False):
        """
        Initializes the ILIAS dataset.
        Args:
            dataset_dir (str): Path to the directory containing the dataset files.
            check_ids (bool): Whether to load distractor image IDs from the file.
        """
        self.dataset_dir = dataset_dir
        self.check_ids = check_ids

        image_query_info = np.loadtxt(
            f"{dataset_dir}/image_ids/image_query_ids.txt", dtype=str, delimiter=","
        )
        self.image_query_ids = sorted(image_query_info[:, 0].tolist())

        self.text_query_ids = sorted(
            np.loadtxt(
                f"{dataset_dir}/image_ids/text_query_ids.txt", dtype=str
            ).tolist()
        )

        positive_info = np.loadtxt(
            f"{dataset_dir}/image_ids/positive_ids.txt", dtype=str, delimiter=","
        )
        self.positive_ids = sorted(positive_info[:, 0].tolist())

        self.positive_info = {}
        for q, s, c in positive_info:
            self.positive_info[q] = (float(s), int(c))

        self.distractor_ids = []
        if self.check_ids:
            with gzip.open(f"{dataset_dir}/image_ids/distractor_ids.txt.gz", "rt") as f:
                for line in f:
                    self.distractor_ids.append(line.strip())

        self.gt = {img.split("/")[0]: set() for img in self.text_query_ids}
        for img in self.positive_ids:
            self.gt[img.split("/")[0]].add(img)

    def get_image_queries(self):
        return self.image_query_ids

    def get_text_queries(self):
        return self.text_query_ids

    def get_queries(self):
        return self.get_image_queries() + self.get_text_queries()

    def get_positives(self):
        return self.positive_ids

    def get_distractors(self):
        return self.distractor_ids

    def _compute_AP(self, q, ids, scores, ranks, k, excluded=None):
        """Computes the Average Precision (AP) for a given query.
        Args:
            q (str): Query ID.
            ids (np.ndarray): Array of database IDs.
            scores (np.ndarray): Array of similarity scores corresponding to the IDs.
            ranks (np.ndarray): Array of ranks corresponding to the IDs.
            k (int): Number of top results to consider.
            excluded (set, optional): Set of positive IDs to exclude from the evaluation.
        Returns:
            tuple: Average Precision (AP) and oracle value.
        """
        if excluded is not None:
            e = [i for i in range(len(ranks)) if ids[ranks[i]] in excluded]
            scores = np.delete(scores.copy(), e, 0)
            ranks = np.delete(ranks.copy(), e, 0)
        else:
            excluded = set()

        # sort by descending score
        order = np.argsort(-scores)
        top_ids = ids[ranks[order][:k]]

        gt = self.gt[q.split("/")[0]] - set(excluded)
        y_true = np.isin(top_ids, list(gt))
        y_score = scores[order][:k]

        if not y_true.any():
            return 0.0, 0.0

        ap = average_precision_score(y_true, y_score)
        ap *= y_true.sum() / min(k, len(gt))
        oracle = y_true.mean() * (k / len(gt))
        return ap, oracle

    def _evaluate_pkl(self, q_ids, db_ids, sims, ranks, k):
        # Check for missing or extra queries
        if self.check_ids:
            extra_q = set(q_ids) - (
                set(self.image_query_ids) | set(self.text_query_ids)
            )
            if extra_q:
                raise ValueError(f"Unknown query IDs: {extra_q}")

            # Check database IDs if loading distractors
            extra_db = set(db_ids) - (set(self.positive_ids) | set(self.distractor_ids))
            if extra_db:
                raise ValueError(f"Unknown database IDs: {extra_db}")

        if not isinstance(db_ids, np.ndarray):
            db_ids = np.asarray(db_ids)

        queries, aps, oracles = [], [], []
        scale, clutter = defaultdict(list), defaultdict(list)
        for q, s, r in zip(q_ids, sims, ranks):
            ap, oracle = self._compute_AP(q, db_ids, s, r, k)
            queries.append(q)
            aps.append(ap * 100)
            oracles.append(oracle * 100)
            for t1, t2 in SCALE_RANGES:
                excluded = set(
                    [
                        i
                        for i in self.gt[q.split("/")[0]]
                        if self.positive_info[i][0] < t1
                        or self.positive_info[i][0] >= t2
                    ]
                )
                if self.gt[q.split("/")[0]] - excluded:
                    ap, _ = self._compute_AP(q, db_ids, s, r, k, excluded)
                    scale[f"{t1}-{t2}"].append(ap * 100)
            for t1, t2 in CLUTTER_RANGES:
                excluded = set(
                    [
                        i
                        for i in self.gt[q.split("/")[0]]
                        if self.positive_info[i][1] < t1
                        or self.positive_info[i][1] >= t2
                    ]
                )
                if self.gt[q.split("/")[0]] - excluded:
                    ap, _ = self._compute_AP(q, db_ids, s, r, k, excluded)
                    clutter[f"{t1}-{t2}"].append(ap * 100)

        return {
            "queries": queries,
            "map": np.mean(aps),
            "aps": np.array(aps),
            "oracle": np.mean(oracles),
            "scale": {k: np.mean(v) for k, v in scale.items()},
            "clutter": {k: np.mean(v) for k, v in clutter.items()},
        }

    def _evaluate_json(self, similarities, k):
        # Check for missing or extra queries
        evaluate_n = 1232
        if self.check_ids:
            q_ids = list(similarities.keys())
            db_ids = sum([list(i.keys()) for i in similarities.values()], [])

            extra_q = set(q_ids) - (
                set(self.image_query_ids) | set(self.text_query_ids)
            )
            if extra_q:
                raise ValueError(f"Unknown query IDs: {extra_q}")

            # Check database IDs if loading distractors
            extra_db = set(db_ids) - (set(self.positive_ids) | set(self.distractor_ids))
            if extra_db:
                raise ValueError(f"Unknown database IDs: {extra_db}")

        queries, aps, oracles = [], [], []
        scale, clutter = defaultdict(list), defaultdict(list)
        for q in self.get_queries()[:evaluate_n]:
            if q not in similarities:
                continue
            ids = np.array(list(similarities[q].keys()))
            s = np.array(list(similarities[q].values()))
            r = np.arange(len(s))
            ap, oracle = self._compute_AP(q, ids, s, r, k)

            queries.append(q)
            aps.append(ap * 100)
            oracles.append(oracle * 100)
            for t1, t2 in SCALE_RANGES:
                excluded = set(
                    [
                        i
                        for i in self.gt[q.split("/")[0]]
                        if self.positive_info[i][0] < t1
                        or self.positive_info[i][0] >= t2
                    ]
                )
                if self.gt[q.split("/")[0]] - excluded:
                    ap, _ = self._compute_AP(q, ids, s, r, k, excluded)
                    scale[f"{t1}-{t2}"].append(ap * 100)
            for t1, t2 in CLUTTER_RANGES:
                excluded = set(
                    [
                        i
                        for i in self.gt[q.split("/")[0]]
                        if self.positive_info[i][1] < t1
                        or self.positive_info[i][1] >= t2
                    ]
                )
                if self.gt[q.split("/")[0]] - excluded:
                    ap, _ = self._compute_AP(q, ids, s, r, k, excluded)
                    clutter[f"{t1}-{t2}"].append(ap * 100)

        return {
            "queries": queries,
            "map": np.mean(aps),
            "aps": np.array(aps),
            "oracle": np.mean(oracles),
            "scale": {k: np.mean(v) for k, v in scale.items()},
            "clutter": {k: np.mean(v) for k, v in clutter.items()},
        }

    def evaluate(
        self,
        all_similarities=None,
        query_ids=None,
        db_ids=None,
        similarities=None,
        ranks=None,
        k=1000,
    ):
        if all_similarities is None:
            return self._evaluate_pkl(query_ids, db_ids, similarities, ranks, k)
        else:
            return self._evaluate_json(all_similarities, k)