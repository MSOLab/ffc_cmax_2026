"""
BaseDispatcher class with shared utilities for all dispatchers.
"""

from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


class BaseDispatcher:
    """
    Base class with common functionality for all dispatchers.

    This class provides shared utilities including:
    - Instance reference and parameters
    - Schedule creation
    - Johnson's rule sequence generation
    - CDS, Gupta, Palmer sequence generation
    - NP candidate generation for mixed dispatch
    """

    def __init__(self, instance: HybridFlowshopParameters) -> None:
        """
        Initialize the BaseDispatcher.

        Args:
            instance: The HybridFlowshopParameters instance.
        """
        self.stage_2_job_2_p: dict[str, dict[str, int]] = instance.stage_2_job_2_p_map
        self.job_2_stage_2_p: dict[str, dict[str, int]] = instance.job_2_stage_2_p_map
        self.stage_id_list: list[str] = instance.stage_id_list
        self.job_id_list: list[str] = instance.job_id_list
        self.machines_per_stage: dict[str, list[str]] = instance.stage_2_machines_map
        self.stage_count: int = instance.stage_count
        self.job_count: int = instance.job_count

    def _create_empty_schedule(self) -> HybridFlowshopLiteSchedule:
        """
        Create an empty schedule for the instance.

        Returns:
            An empty HybridFlowshopLiteSchedule.
        """
        return HybridFlowshopLiteSchedule(
            jobs=self.job_id_list,
            stages=self.stage_id_list,
            machines_per_stage=self.machines_per_stage,
        )

    def _prepare_schedule_for_dispatch(
        self,
        schedule: HybridFlowshopLiteSchedule | None,
        in_place: bool = False,
    ) -> HybridFlowshopLiteSchedule:
        """
        Prepare a schedule for dispatch based on in_place logic.

        Args:
            schedule: If provided and in_place=True, modify directly. Otherwise copy.
            in_place: If True, modify the incumbent schedule directly.
                If False, work on a deepcopy and return new schedule.

        Returns:
            A schedule to dispatch to.
        """
        if schedule is None:
            return self._create_empty_schedule()
        if in_place:
            return schedule
        else:
            return schedule.deepcopy()

    @staticmethod
    def get_johnsons_rule_sequence(
        job_name_2_p1_map: dict[str, int], job_name_2_p2_map: dict[str, int]
    ) -> list[str]:
        """
        Apply Johnson's rule to determine the job sequence.

        Args:
            job_name_2_p1_map: job ID -> processing time for the 1st stage
            job_name_2_p2_map: job ID -> processing time for the 2nd stage

        Returns:
            A list of job IDs ordered according to Johnson's rule.
        """
        jobs = list(job_name_2_p1_map.keys())

        l1: list[str] = []
        l2: list[str] = []

        for job in jobs:
            if job_name_2_p1_map[job] <= job_name_2_p2_map[job]:
                l1.append(job)
            else:
                l2.append(job)

        l1.sort(key=lambda j: (job_name_2_p1_map[j], j))
        l2.sort(key=lambda j: (job_name_2_p2_map[j], j), reverse=True)

        return l1 + l2

    def get_cds_sequence(self, k: int) -> list[str]:
        """
        Get Campbell-Dudek-Smith (CDS) sequence given k.

        Args:
            k: Index between 1 and m-1, where m is the number of stages.

        Returns:
            A list of job IDs ordered according to the CDS rule.
        """
        jobs = self.job_id_list
        stages = self.stage_id_list
        p_dict = self.job_2_stage_2_p

        m = self.stage_count
        p1_stages = stages[:k]
        p2_stages = stages[m - k :]
        p1 = {j: sum(p_dict[j][i] for i in p1_stages) for j in jobs}
        p2 = {j: sum(p_dict[j][i] for i in p2_stages) for j in jobs}

        return self.get_johnsons_rule_sequence(p1, p2)

    def get_gupta_sequence(self) -> list[str]:
        """
        Return the job sequence according to Gupta's functional heuristic algorithm.

        Returns:
            A list of job IDs in Gupta heuristic order.
        """
        jobs = self.job_id_list
        stages = self.stage_id_list
        m = len(stages)

        gupta_score = {}
        total_p = {}

        for j in jobs:
            stage_2_p = self.job_2_stage_2_p[j]
            min_sum = min(
                stage_2_p[stages[m1]] + stage_2_p[stages[m1 + 1]] for m1 in range(m - 1)
            )
            A = 1 if stage_2_p[stages[-1]] <= stage_2_p[stages[0]] else -1
            f_j = A / min_sum if min_sum != 0 else float("inf")
            gupta_score[j] = f_j
            total_p[j] = sum(stage_2_p[s] for s in stages)

        sorted_jobs = sorted(jobs, key=lambda j: (gupta_score[j], total_p[j], j))
        return sorted_jobs

    def get_palmer_sequence(self) -> list[str]:
        """
        Return the job sequence according to Palmer's slope index heuristic.

        Returns:
            Job ID list in Palmer slope order (descending s_i).
        """
        jobs = self.job_id_list
        stages = self.stage_id_list
        m = len(stages)

        palmer_score = {}
        for j in jobs:
            stage_2_p = self.job_2_stage_2_p[j]
            s = sum(
                (m - 2 * (stage_idx + 1) + 1) * stage_2_p[stages[stage_idx]]
                for stage_idx in range(m)
            )
            palmer_score[j] = s

        sorted_jobs = sorted(jobs, key=lambda j: (palmer_score[j], j))
        return sorted_jobs
