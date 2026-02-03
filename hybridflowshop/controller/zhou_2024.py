import logging
import random
import time
from dataclasses import dataclass
from typing import Iterable

from mbls.cpsat import ObjValueBoundStore
from routix import ElapsedTimer
from schore.parameters_examples import HybridFlowshopParameters
from schore.schedule_examples.parallel_shop.identical_flow import (
    HybridFlowshopOperation,
    HybridFlowshopSchedule,
)

from .hfs_sched_lite import RapidEvaluatorZhou2024


@dataclass(frozen=True)
class PrTs2024Params:
    time_limit_sec: float

    n: int
    m: int
    population_multiplier: int
    operator_iterations: int
    similarity_threshold: float
    tabu_list_length_multiplier: int
    ts_max_iterations_multiplier: int
    a_hat: float

    # Path relinking (Peng et al., 2015) parameters
    alpha: int
    pr_ts_iterations: int
    # Tabu tenure (Cheng et al., 2016) for path relinking parameters
    tt: int
    d_1: int
    d_2: int

    @property
    def no_improve_limit(self) -> int:
        return int(10 * self.a_hat * self.n)

    @property
    def pop_size(self) -> int:
        return self.population_multiplier * self.n

    @property
    def tabu_tenure(self) -> int:
        return self.tabu_list_length_multiplier * self.n

    @property
    def ts_max_iters(self) -> int:
        return self.ts_max_iterations_multiplier * self.n

    def get_log_string(self) -> str:
        return (
            f"n={self.n}, m={self.m}, A_hat={self.a_hat:.3f}, "
            f"time_limit={self.time_limit_sec:.3f}s, no_improve_limit={self.no_improve_limit}, "
            f"pop_size={self.pop_size}, op_iters={self.operator_iterations}, "
            f"pr_th={self.similarity_threshold:.2f}, tabu_tenure={self.tabu_tenure}, "
            f"ts_max_iters={self.ts_max_iters}"
        )


# Start tabu search helpers


@dataclass(frozen=True)
class OpRef:
    """An operation in a realized schedule (forward semi-active)."""

    stage_idx: int
    machine_idx: int
    pos: int  # position on that machine sequence
    job: str


# --- Complete encoding type helpers ---
# complete_enc[stage_idx][machine_idx] = list of jobs in processing order
CompleteEncoding = tuple[tuple[tuple[str, ...], ...], ...]  # hashable


def _to_hashable(enc: list[list[list[str]]]) -> CompleteEncoding:
    return tuple(tuple(tuple(seq) for seq in stage) for stage in enc)


def _to_mutable(enc: CompleteEncoding) -> list[list[list[str]]]:
    return [[list(seq) for seq in stage] for stage in enc]


# End tabu search helpers


@dataclass
class PrTsRunState:
    timer: ElapsedTimer
    iter_idx: int
    sub_obj_store: ObjValueBoundStore[int]

    population: list[CompleteEncoding]
    fitness_ce: dict[CompleteEncoding, int]

    # Best solution stored as CompleteEncoding
    best_sol: CompleteEncoding | None

    # Cached permutation-view projections for CE
    perm_view_cache: dict[CompleteEncoding, tuple[str, ...]]
    best_fit: int | None

    stagnation: int
    perm_eval_cache: dict[tuple[str, ...], int]
    complete_eval_cache: dict[CompleteEncoding, int]


@dataclass
class PrTsResult:
    schedule: HybridFlowshopSchedule
    sub_obj_store: ObjValueBoundStore[int]
    last_obj_value: float


class PrTs2024Runner:
    """
    Refactored to expose ONLY the parameters explicitly stated in the paper's experimental study,
    plus the fixed A-hat (=0.5) decided by user.

    Exposed params (run arguments):
      - population_multiplier (default=2)   -> pop_size = 2*n
      - operator_iterations (default=20)
      - similarity_threshold (default=0.7)
      - tabu_list_length_multiplier (default=1) -> tenure = n
      - ts_max_iterations_multiplier (default=100) -> TS iters = 100*n
      - a_hat (default=0.5)

    Termination (paper):
      - no improvement for 10*A_hat*n iterations, OR
      - time limit m*A_hat*n seconds
    """

    def __init__(
        self,
        stage_2_job_2_p_dict: dict[str, dict[str, int]],
        instance: HybridFlowshopParameters,
    ):
        self.job_id_list = instance.job_id_list
        self.stage_2_job_2_p_dict: dict[str, dict[str, int]] = stage_2_job_2_p_dict
        self.stage_2_machines_map: dict[str, list[str]] = instance.stage_2_machines_map
        self.instance: HybridFlowshopParameters = instance
        self._st: PrTsRunState | None = None
        self.evaluator = RapidEvaluatorZhou2024(
            job_ids=self.job_id_list,
            stage_ids=instance.stage_id_list,
            stage_2_machines=self.stage_2_machines_map,
            p_time=self.stage_2_job_2_p_dict,
            machine_2_stage={
                mc: idx
                for idx, s in enumerate(self.stage_2_machines_map.keys())
                for mc in self.stage_2_machines_map[s]
            },
        )

    def _require_state(self) -> PrTsRunState:
        if self._st is None:
            raise RuntimeError("Runner is not active.")
        return self._st

    def _time_up(self) -> bool:
        return time.perf_counter() >= self.deadline

    def run(
        self,
        population_multiplier: int = 2,
        operator_iterations: int = 20,
        similarity_threshold: float = 0.7,
        alpha: int = 5,
        pr_ts_iterations: int = 500,
        tt: int = 2,
        d_1: int = 5,
        d_2: int = 5,
        tabu_list_length_multiplier: int = 1,
        ts_max_iterations_multiplier: int = 100,
        a_hat: float = 0.5,
    ) -> PrTsResult:
        # Start algorithm parameter validation

        if population_multiplier <= 0:
            raise ValueError("population_multiplier must be positive.")
        if operator_iterations < 0:
            raise ValueError("operator_iterations must be non-negative.")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0,1].")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if pr_ts_iterations <= 0:
            raise ValueError("pr_ts_iterations must be positive.")
        if tt <= 0:
            raise ValueError("tt must be positive.")
        if d_1 <= 0:
            raise ValueError("d_1 must be positive.")
        if d_2 <= 0:
            raise ValueError("d_2 must be positive.")
        if tabu_list_length_multiplier <= 0:
            raise ValueError("tabu_list_length_multiplier must be positive.")
        if ts_max_iterations_multiplier <= 0:
            raise ValueError("ts_max_iterations_multiplier must be positive.")
        if a_hat <= 0.0:
            raise ValueError("a_hat must be positive.")
        jobs: list[str] = self.job_id_list
        n: int = len(jobs)
        if n <= 0:
            raise ValueError("Instance has no jobs.")
        if n == 1:
            schedule = self._dispatch_perm_to_schedule((jobs[0],))
            last_obj_value = schedule.makespan
            sub_obj_store = ObjValueBoundStore[int]()
            sub_obj_store.add_obj_value(0, last_obj_value, is_maximize=None)
            return PrTsResult(
                schedule=schedule,
                sub_obj_store=sub_obj_store,
                last_obj_value=last_obj_value,
            )

        m = self._compute_machine_count()
        # time_limit_sec = m * a_hat * n
        time_limit_sec = self.instance.stage_count * a_hat * n
        t0 = time.perf_counter()
        self.deadline = t0 + time_limit_sec

        self.params: PrTs2024Params = PrTs2024Params(
            time_limit_sec=time_limit_sec,
            n=n,
            m=m,
            population_multiplier=population_multiplier,
            operator_iterations=operator_iterations,
            similarity_threshold=similarity_threshold,
            tabu_list_length_multiplier=tabu_list_length_multiplier,
            ts_max_iterations_multiplier=ts_max_iterations_multiplier,
            a_hat=a_hat,
            alpha=alpha,
            pr_ts_iterations=pr_ts_iterations,
            tt=tt,
            d_1=d_1,
            d_2=d_2,
        )

        # End algorithm parameter validation

        # Set initial state
        timer = ElapsedTimer()
        sub_obj_store: ObjValueBoundStore[int] = ObjValueBoundStore[int]()
        sub_obj_store.obj_value_series.name = "ObjVal"

        st = PrTsRunState(
            timer=timer,
            iter_idx=0,
            sub_obj_store=sub_obj_store,
            population=[],
            fitness_ce={},
            best_sol=None,
            perm_view_cache={},
            best_fit=None,
            stagnation=0,
            perm_eval_cache={},
            complete_eval_cache={},
        )
        self._st: PrTsRunState = st

        try:
            logging.info("[PRTS] start: %s", self.params.get_log_string())

            # Heartbeat logging: keep the user confident the loop is running
            # even when no improvement happens (or when many iterations are skipped
            # due to the similarity criterion).
            last_heartbeat_iter = 0
            last_heartbeat_sec = 0.0
            sim_rejects = 0
            last_improve_iter = 0
            last_improve_sec = 0.0
            heartbeat_every_sec = 5.0
            heartbeat_every_iters = max(500, 10 * self.params.n)

            # 1) Initialization
            self._initialize_population(
                jobs=jobs,
                pop_size=self.params.pop_size,
                operator_iterations=self.params.operator_iterations,
            )
            self._update_best_and_log(note="init")

            # 2) main loop until termination criterion satisfied
            while True:
                st.iter_idx += 1

                # Progress heartbeat (runs regardless of improvement)
                elapsed = st.timer.elapsed_sec
                if (
                    elapsed - last_heartbeat_sec >= heartbeat_every_sec
                    or st.iter_idx - last_heartbeat_iter >= heartbeat_every_iters
                ):
                    logging.info(
                        "[PRTS] heartbeat: iter=%d, elapsed=%.2fs/%.2fs, best=%s, stagnation=%d, sim_rejects=%d, last_improve_iter=%d (%.2fs)",
                        st.iter_idx,
                        elapsed,
                        self.params.time_limit_sec,
                        st.best_fit if st.best_fit is not None else "NA",
                        st.stagnation,
                        sim_rejects,
                        last_improve_iter,
                        last_improve_sec,
                    )
                    last_heartbeat_iter = st.iter_idx
                    last_heartbeat_sec = elapsed

                # time limit check
                if self._time_up():
                    break

                # stagnation check
                if (
                    self.params.no_improve_limit > 0
                    and st.stagnation >= self.params.no_improve_limit
                ):
                    break

                # 2.1 selection: Xbest is current best (CE), Xselect via roulette wheel (CE)
                x_best_ce: CompleteEncoding = st.population[0]
                x_select_ce: CompleteEncoding = self._roulette_select()

                # 2.2 similarity criterion satisfied? (computed on permutation views)
                x_best = self._perm_view_of(x_best_ce)
                x_select = self._perm_view_of(x_select_ce)
                sim: float = get_kendall_tau_similarity(x_select, x_best)
                if sim >= similarity_threshold:
                    # If not, go back to selection
                    st.stagnation += 1
                    sim_rejects += 1
                    continue

                # 2.3 path relinking
                pr_candidate: CompleteEncoding = self._path_relink_peng_lite(
                    x_select, x_best
                )

                # 2.4 (local) tabu search
                ts_sol: CompleteEncoding = self._tabu_search(
                    start_perm=pr_candidate,
                    iters=self.params.ts_max_iters,
                    tabu_tenure=self.params.tabu_tenure,
                )

                # 2.5 replacement: if new solution is unique and better than worst, insert
                self._try_accept(ts_sol)

                # 2.6 update best & log
                improved = self._update_best_and_log(note=f"iter={st.iter_idx}")
                if improved:
                    st.stagnation = 0
                    last_improve_iter = st.iter_idx
                    last_improve_sec = st.timer.elapsed_sec
                else:
                    st.stagnation += 1

            if st.best_sol is None or st.best_fit is None:
                raise RuntimeError("PRTS finished without any solution.")

            best_ce = st.best_sol
            best_schedule = self._dispatch_complete_encoding_to_schedule(best_ce)
            best_obj = float(best_schedule.makespan)

            return PrTsResult(
                schedule=best_schedule,
                sub_obj_store=st.sub_obj_store,
                last_obj_value=best_obj,
            )

        finally:
            self._st = None

    # -------------------------
    # Initialization (paper: permutation-based + 3 operators * 20)
    # -------------------------
    def _initialize_population(
        self, jobs: list[str], pop_size: int, operator_iterations: int
    ) -> None:
        """Population initialization (permutation-based), but store population as CompleteEncoding.

        Steps (aligned with the paper's description):
        - Generate a random permutation population
        - For each individual, apply insertion / swap / pairwise exchange operators
          for operator_iterations iterations (accept improving move)
        - Evaluate via permutation -> complete encoding -> makespan, but KEEP the CE
          as the population member (permutation remains a view only)
        """
        st = self._require_state()
        n = len(jobs)
        # Temporary permutation population used only during initialization/refinement
        perm_pop: list[tuple[str, ...]] = []

        # We enforce uniqueness on the CE level to avoid storing multiple perms that
        # decode to the same complete encoding (important for CE-centric population).
        ce_set: set[CompleteEncoding] = set()

        # 1) generate initial population (random perms), store fitness keyed by CE
        while len(perm_pop) < pop_size:
            perm = tuple(random.sample(jobs, k=n))

            ce = self._perm_to_complete_encoding(perm)
            if ce in ce_set:
                continue
            ce_set.add(ce)
            perm_pop.append(perm)
            st.fitness_ce[ce] = self._evaluate_complete_encoding(ce)

        # 2) refine each permutation via 3 operators (conditionally replace)
        for ind in range(operator_iterations):
            for idx, cur_perm in enumerate(list(perm_pop)):
                cur_ce = self._perm_to_complete_encoding(cur_perm)

                cur_f = st.fitness_ce[cur_ce]

                c1 = self._op_insertion(cur_perm)
                c2 = self._op_swap(cur_perm)
                c3 = self._op_pairwise_exchange(cur_perm)

                best_perm = cur_perm
                best_ce = cur_ce
                best_f = cur_f
                for cand_perm in (c1, c2, c3):
                    cand_ce = self._perm_to_complete_encoding(cand_perm)
                    f = self._evaluate_complete_encoding(cand_ce)
                    if f < best_f:
                        best_f = f
                        best_ce = cand_ce
                        best_perm = cand_perm

                # If improved and unique at CE level, replace
                if best_ce != cur_ce and best_ce not in ce_set:
                    ce_set.remove(cur_ce)
                    del st.fitness_ce[cur_ce]
                    ce_set.add(best_ce)
                    perm_pop[idx] = best_perm
                    st.fitness_ce[best_ce] = best_f

        # 3) finalize: population is CE (CE-centric), clear perm-view cache
        st.population = [self._perm_to_complete_encoding(p) for p in perm_pop]
        st.perm_view_cache.clear()
        self._sort_population()

    @staticmethod
    def _op_insertion(sol: tuple[str, ...]) -> tuple[str, ...]:
        n = len(sol)
        if n <= 1:
            return sol
        i = random.randrange(0, n)
        j = random.randrange(0, n)
        while j == i and n > 1:
            j = random.randrange(0, n)
        arr = list(sol)
        x = arr.pop(i)
        arr.insert(j, x)
        return tuple(arr)

    @staticmethod
    def _op_swap(sol: tuple[str, ...]) -> tuple[str, ...]:
        n = len(sol)
        if n <= 1:
            return sol
        i = random.randrange(0, n)
        j = random.randrange(0, n)
        while j == i and n > 1:
            j = random.randrange(0, n)
        arr = list(sol)
        arr[i], arr[j] = arr[j], arr[i]
        return tuple(arr)

    @staticmethod
    def _op_pairwise_exchange(sol: tuple[str, ...]) -> tuple[str, ...]:
        """
        Paper description: pick two non-adjacent elements; use midpoint as symmetry axis; swap pairs symmetrically.
        Implementation policy (fixed):
          - choose i < j with gap>=2
          - reverse the segment [i, j] (this matches "pairwise symmetric swaps" around midpoint)
        """
        n = len(sol)
        if n <= 3:
            return sol
        i = random.randrange(0, n - 2)
        j = random.randrange(i + 2, n)
        arr = list(sol)
        arr[i : j + 1] = reversed(arr[i : j + 1])
        return tuple(arr)

    # -------------------------
    # Selection (paper: roulette wheel; fitness = inverse makespan)
    # -------------------------
    def _roulette_select(self) -> CompleteEncoding:
        st = self._require_state()
        pool = st.population[1:]
        if not pool:
            # Population should have at least 2 members; raise error if not
            raise RuntimeError("Roulette selection pool is empty.")

        # fitness = 1 / makespan
        weights = []
        for ce in pool:
            ms = st.fitness_ce[ce]
            w = 1.0 / ms if ms > 0 else 1e9
            weights.append(w)

        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for ce, w in zip(pool, weights):
            acc += w
            if acc >= r:
                return ce
        return pool[-1]

    # -------------------------
    # Path relinking by Peng et al. (2015)
    # -------------------------
    def _path_relink_peng_lite(
        self, initiating: tuple[str, ...], guiding: tuple[str, ...]
    ) -> CompleteEncoding:
        """
        Peng et al. (2015) style Path Relinking, but WITHOUT strong TS.
        - Build PathSet using alpha/beta distance-controlled sampling.
        - Apply slight TS (si iterations) to each solution in PathSet.
        - Return the best improved solution from PathSet as the reference solution.

        Notes:
        - Representation here is permutation (job sequence).
        - Distance dis = number of mismatched positions between current and guiding.
          (Peng's JSP distance is |NCS|; for permutations this is a reasonable analogue.)
        """
        st = self._require_state()
        params = self.params
        n = params.n

        pr_start_sec = st.timer.elapsed_sec
        last_pr_log_sec = pr_start_sec
        pr_log_every_sec = 2.0

        # distance dis: permutation positional mismatch count (practical analogue of |NCS| for permutations)
        def dist(x: tuple[str, ...], y: tuple[str, ...]) -> int:
            return sum(1 for i in range(n) if x[i] != y[i])

        dis0 = dist(initiating, guiding)
        if dis0 == 0:
            raise ValueError("Initiating and guiding solutions are identical.")

        alpha = min(params.alpha, dis0)  # clamp alpha not to exceed path length
        beta = max(dis0 // 10, 2)  # Peng: beta = max(dis/10, 2)

        # logging.info(
        #     "[PRTS][PR] start: iter=%d, dis=%d, alpha=%d, beta=%d",
        #     st.iter_idx,
        #     dis0,
        #     alpha,
        #     beta,
        # )

        # ---- Path construction via position-fixing swaps (permutation PR) ----
        cur = list(initiating)
        pos = {job: idx for idx, job in enumerate(cur)}

        def step_toward_guiding() -> bool:
            # choose an index where cur differs from guiding
            diff_indices = [i for i in range(n) if cur[i] != guiding[i]]
            if not diff_indices:
                return False
            i = random.choice(diff_indices)
            desired = guiding[i]
            j = pos[desired]
            # swap positions i and j
            cur[i], cur[j] = cur[j], cur[i]
            pos[cur[j]] = j
            pos[cur[i]] = i
            return True

        # ---- Build PathSet: move alpha steps, then sample every beta until distance <= alpha ----
        pathset: list[tuple[str, ...]] = []

        # Lines 6–10 in Peng Algorithm 2: move alpha steps, then add to PathSet
        for _ in range(alpha):
            if not step_toward_guiding():
                break
        pathset.append(tuple(cur))

        # Lines 13–20 in Peng Algorithm 2: add every beta steps until distance < alpha
        while dist(tuple(cur), guiding) > alpha:
            for _ in range(beta):
                if not step_toward_guiding():
                    break
            pathset.append(tuple(cur))
            if dist(tuple(cur), guiding) == 0:
                break

            now = st.timer.elapsed_sec
            if now - last_pr_log_sec >= pr_log_every_sec:
                # logging.info(
                #     "[PRTS][PR] building pathset: iter=%d, size=%d, cur_dis=%d",
                #     st.iter_idx,
                #     len(pathset),
                #     dist(tuple(cur), guiding),
                # )
                last_pr_log_sec = now

        # de-duplicate PathSet while preserving order
        seen: set[tuple[str, ...]] = set()
        uniq_pathset: list[tuple[str, ...]] = []
        for s in pathset:
            if s not in seen:
                seen.add(s)
                uniq_pathset.append(s)

        # logging.info(
        #     "[PRTS][PR] pathset ready: iter=%d, uniq_size=%d (raw=%d)",
        #     st.iter_idx,
        #     len(uniq_pathset),
        #     len(pathset),
        # )

        # ---- slight TS on each candidate in PathSet; pick best ----
        # UB = current global best (best_fit) if available, else best among population
        ub = st.best_fit
        if ub is None:
            # fallback: compute best among current population
            if st.population:
                ub = min(st.fitness_ce[ce] for ce in st.population)
            else:
                ub = self._evaluate_perm(initiating)

        best_sol = self._perm_to_complete_encoding(initiating)
        best_fit = self._evaluate_complete_encoding(best_sol)

        last_cand_log_sec = st.timer.elapsed_sec
        cand_log_every_sec = 2.0
        for cand_idx, cand in enumerate(uniq_pathset, start=1):
            if self._time_up():
                break
            now = st.timer.elapsed_sec
            if now - last_cand_log_sec >= cand_log_every_sec:
                # logging.info(
                #     "[PRTS][PR] scanning candidates: iter=%d, %d/%d, best=%s",
                #     st.iter_idx,
                #     cand_idx,
                #     cand_cnt,
                #     best_fit,
                # )
                last_cand_log_sec = now
            # dynamic tenure depends on current cand and UB; we compute tenure once per TS call
            cand_fit = self._evaluate_perm(cand)
            tenure = self._cheng_dynamic_tabu_tenure(current_f=cand_fit, ub=float(ub))
            complete_enc_cand = self._perm_to_complete_encoding(cand)

            improved = self._tabu_search(
                start_perm=complete_enc_cand,
                iters=params.pr_ts_iterations,  # slight TS iterations
                tabu_tenure=tenure,
            )
            fit = self._evaluate_complete_encoding(improved)
            if fit < best_fit:
                best_fit = fit
                best_sol = improved

        return best_sol

    def _cheng_dynamic_tabu_tenure(self, current_f: float, ub: float) -> int:
        """
        Cheng et al. dynamic tabu tenure:
            RL = max((f - UB)/d1, d2)
            tenure = tt + rand(RL)

        Implementation notes:
        - Uses integer arithmetic for RL.
        - rand(RL) is sampled uniformly from [0, RL-1] if RL>0 else 0.
        """
        p = self.params

        gap = current_f - ub
        # if current is already <= ub, use minimum range
        if gap <= 0:
            rl = p.d_2
        else:
            rl = max(int(gap / p.d_1), p.d_2)

        if rl <= 1:
            r = 0
        else:
            r = random.randrange(0, rl)
        return p.tt + r

    # ============================================================
    # 1) permutation -> complete encoding  (transition for TS phase)
    # ============================================================
    def _perm_to_complete_encoding(self, perm: tuple[str, ...]) -> CompleteEncoding:
        """
        Build a complete encoding from a permutation via forward dispatch:
          - process stages in order
          - for each stage, assign each job (in perm order) to the machine that yields
            the earliest completion time, appending it to that machine's sequence.

        This is a reasonable concrete realization of the paper's
        "transition to complete encoding" idea (no extra decoding heuristics).
        """
        jobs = list(perm)
        stage_ids = list(self.stage_2_machines_map.keys())
        stage_machines = [self.stage_2_machines_map[s] for s in stage_ids]

        # job ready time after completing previous stage
        job_ready = {j: 0 for j in jobs}

        enc: list[list[list[str]]] = []
        for s_idx, s in enumerate(stage_ids):
            machines = stage_machines[s_idx]
            m_cnt = len(machines)

            # machine ready times within this stage
            mc_ready = [0] * m_cnt
            stage_seq: list[list[str]] = [[] for _ in range(m_cnt)]

            for j in jobs:
                # choose machine with earliest completion
                best_k = 0
                best_c = None
                p = self.stage_2_job_2_p_dict[s][j]
                for k in range(m_cnt):
                    st = max(job_ready[j], mc_ready[k])
                    c = st + p
                    if best_c is None or c < best_c:
                        best_c = c
                        best_k = k

                stage_seq[best_k].append(j)

                # update times (for next assignment decisions in SAME stage)
                st = max(job_ready[j], mc_ready[best_k])
                c = st + p
                mc_ready[best_k] = c
                job_ready[j] = c  # completion of this stage

            enc.append(stage_seq)

        return _to_hashable(enc)

    # ============================================================
    # 2) forward schedule evaluation from complete encoding
    # ============================================================
    def _eval_complete_encoding(
        self, enc: CompleteEncoding
    ) -> tuple[
        int,
        dict[tuple[int, str], int],
        dict[tuple[int, str], int],
        dict[tuple[int, str], tuple[int, int, int]],
    ]:
        """
        Forward semi-active schedule simulation.

        Returns
          - makespan
          - start_map[(stage_idx, job)] = start time
          - end_map[(stage_idx, job)]   = end time
          - loc_map[(stage_idx, job)]   = (machine_idx, pos, proc_time)
        """
        stage_ids = list(self.stage_2_machines_map.keys())

        job_ready: dict[str, int] = {j: 0 for j in self.job_id_list}
        start_map: dict[tuple[int, str], int] = {}
        end_map: dict[tuple[int, str], int] = {}
        loc_map: dict[tuple[int, str], tuple[int, int, int]] = {}

        cmax = 0
        for s_idx, s in enumerate(stage_ids):
            stage = enc[s_idx]
            for m_idx, seq in enumerate(stage):
                mc_ready = 0
                for pos, j in enumerate(seq):
                    p = int(self.stage_2_job_2_p_dict[s][j])
                    st = max(job_ready[j], mc_ready)
                    en = st + p
                    start_map[(s_idx, j)] = st
                    end_map[(s_idx, j)] = en
                    loc_map[(s_idx, j)] = (m_idx, pos, p)
                    mc_ready = en
                    job_ready[j] = en
                    if en > cmax:
                        cmax = en

        return cmax, start_map, end_map, loc_map

    # ============================================================
    # 3) critical paths (multi) + critical blocks (all paths)
    # ============================================================
    def _extract_critical_blocks_all_paths(
        self,
        enc: CompleteEncoding,
        start_map: dict[tuple[int, str], int],
        end_map: dict[tuple[int, str], int],
        loc_map: dict[tuple[int, str], tuple[int, int, int]],
        cmax: int,
    ) -> list[list[OpRef]]:
        """
        Paper: "takes all critical paths into account" -> we enumerate all critical paths
        by backtracking all zero-slack predecessor choices, then extract all critical blocks
        from the union of operations on those paths.

        Critical predecessor rule (semi-active forward schedule):
          For op (s,j): start = max( end(s-1,j), end(s, prev_on_machine) )
          Any predecessor whose end equals start is a critical predecessor (tie => multiple paths).
        """
        stage_cnt = len(enc)
        # reverse index: for each (s_idx, m_idx, pos) -> job
        job_at: dict[tuple[int, int, int], str] = {}
        for s_idx, stage in enumerate(enc):
            for m_idx, seq in enumerate(stage):
                for pos, j in enumerate(seq):
                    job_at[(s_idx, m_idx, pos)] = j

        # find a terminal op achieving cmax (can be multiple)
        terminals: list[tuple[int, str]] = []
        for (s_idx, j), en in end_map.items():
            if abs(en - cmax) < 1e-9 and s_idx == stage_cnt - 1:
                terminals.append((s_idx, j))
        if not terminals:
            # fallback: any op with end==cmax
            for (s_idx, j), en in end_map.items():
                if abs(en - cmax) < 1e-9:
                    terminals.append((s_idx, j))

        def preds(node: tuple[int, str]) -> list[tuple[int, str]]:
            s_idx, j = node
            st = start_map[(s_idx, j)]
            out: list[tuple[int, str]] = []

            # job predecessor
            if s_idx > 0:
                pj = (s_idx - 1, j)
                if end_map[pj] == st:
                    out.append(pj)

            # machine predecessor (previous in same machine sequence)
            m_idx, pos, _p = loc_map[(s_idx, j)]
            if pos > 0:
                prev_job = job_at[(s_idx, m_idx, pos - 1)]
                pm = (s_idx, prev_job)
                if end_map[pm] == st:
                    out.append(pm)

            return out

        # enumerate all critical paths (could blow up in theory, but usually manageable)
        all_paths: list[list[tuple[int, str]]] = []
        stack: list[tuple[tuple[int, str], list[tuple[int, str]]]] = []
        for t in terminals:
            stack.append((t, [t]))

        visited_cap = (
            20000  # safety to avoid pathological explosion (NOT a tunable param)
        )
        visited_cnt = 0

        while stack:
            node, path = stack.pop()
            visited_cnt += 1
            if visited_cnt > visited_cap:
                # best-effort: stop enumerating further paths
                break

            ps = preds(node)
            if not ps:
                all_paths.append(path)
                continue
            for pnode in ps:
                stack.append((pnode, [pnode] + path))

        # collect critical operations = union of all path nodes
        crit_ops: set[tuple[int, str]] = set()
        for p in all_paths:
            crit_ops.update(p)

        # extract critical blocks: maximal consecutive critical ops on same machine
        blocks: list[list[OpRef]] = []
        for s_idx, stage in enumerate(enc):
            for m_idx, seq in enumerate(stage):
                cur_block: list[OpRef] = []
                for pos, j in enumerate(seq):
                    if (s_idx, j) in crit_ops:
                        cur_block.append(OpRef(s_idx, m_idx, pos, j))
                    else:
                        if len(cur_block) >= 2:
                            blocks.append(cur_block)
                        cur_block = []
                if len(cur_block) >= 2:
                    blocks.append(cur_block)

        return blocks

    # ============================================================
    # 4) Neighborhoods (paper-modified swap versions)
    # ============================================================
    def _moves_n7(self, block: list[OpRef]) -> Iterable[tuple[str, int, int, int, int]]:
        """
        Paper-modified N7 (swap):
          - first and last operations of a critical block
          - swapped with internal operations of the same block

        Move encoding:
          ("N7", stage_idx, machine_idx, pos_a, pos_b)  where pos_a != pos_b on same machine
        """
        if len(block) < 3:
            return []
        s = block[0].stage_idx
        m = block[0].machine_idx
        head = block[0].pos
        tail = block[-1].pos
        internals = [op.pos for op in block[1:-1]]
        out = []
        for p in internals:
            out.append(("N7", s, m, head, p))
            out.append(("N7", s, m, tail, p))
        return out

    def _moves_k_insertion_swap(
        self, block: list[OpRef], enc: CompleteEncoding
    ) -> Iterable[tuple[str, int, int, int, int, int]]:
        """
        Paper-modified k-insertion (swap across machines in same stage):
          - operations within a critical block
          - swapped with operations on other machines at the same stage

        Move encoding:
          ("KSWAP", stage_idx, m_a, pos_a, m_b, pos_b) with m_a != m_b
        """
        if not block:
            return []
        s = block[0].stage_idx
        stage = enc[s]
        out = []
        for op in block:
            m_a, pos_a = op.machine_idx, op.pos
            for m_b, seq_b in enumerate(stage):
                if m_b == m_a or len(seq_b) == 0:
                    continue
                for pos_b in range(len(seq_b)):
                    out.append(("KSWAP", s, m_a, pos_a, m_b, pos_b))
        return out

    def _apply_move_ce(self, enc: CompleteEncoding, mv) -> CompleteEncoding:
        e = _to_mutable(enc)
        if mv[0] == "N7":
            _, s, m, a, b = mv
            e[s][m][a], e[s][m][b] = e[s][m][b], e[s][m][a]
            return _to_hashable(e)
        if mv[0] == "KSWAP":
            _, s, ma, pa, mb, pb = mv
            e[s][ma][pa], e[s][mb][pb] = e[s][mb][pb], e[s][ma][pa]
            return _to_hashable(e)
        raise ValueError(f"Unknown move: {mv}")

    # ============================================================
    # 5) Tabu Search exactly in the paper’s spirit
    # ============================================================
    def _tabu_search(
        self,
        start_perm: CompleteEncoding,
        iters: int,
        tabu_tenure: int,
    ) -> CompleteEncoding:
        """
        Paper TS:
          - start from candidate solution (here: permutation -> complete encoding)
          - compute critical blocks (considering multiple critical paths)
          - generate N7 and k-insertion neighborhoods (swap-enhanced)
          - apply tabu + aspiration:
              aspiration: if improves global best, allow even if tabu
              otherwise choose best non-tabu

        Notes:
          - Paper also proposes a rapid evaluation method, but details aren’t sufficient
            to implement faithfully here, so we do full re-evaluation.
        """
        st = self._require_state()

        ts_start_sec = st.timer.elapsed_sec
        last_ts_log_sec = ts_start_sec
        ts_log_every_sec = 2.0
        ts_log_every_iters = max(200, 5 * self.params.n)

        cur = start_perm
        cur_fit, cur_start, cur_end, cur_loc = self._eval_complete_encoding(cur)
        cache = self.evaluator.build_forward_cache_ce(cur)

        best = cur
        best_fit = cur_fit

        # aspiration threshold: allow tabu moves only if they beat the current global best (paper)
        aspiration_ub = st.best_fit if st.best_fit is not None else best_fit
        # tabu dictionary: move -> expiration iteration index
        tabu: dict[tuple, int] = {}

        for it in range(1, iters + 1):
            if self._time_up():
                break
            blocks = self._extract_critical_blocks_all_paths(
                cur, cur_start, cur_end, cur_loc, cur_fit
            )
            if not blocks:
                # logging.info(
                #     "[PRTS][TS] no critical blocks: iter=%d, ts_it=%d/%d, cur=%d, best=%d",
                #     st.iter_idx,
                #     it,
                #     iters,
                #     cur_fit,
                #     best_fit,
                # )
                break

            # neighborhood from ALL critical blocks (paper: "applied to all critical blocks")
            moves: list[tuple] = []
            for blk in blocks:
                moves.extend(self._moves_n7(blk))
                moves.extend(self._moves_k_insertion_swap(blk, cur))

            # de-duplicate moves (preserve order)
            seen_mv = set()
            uniq_moves = []
            for mv in moves:
                if mv in seen_mv:
                    continue
                seen_mv.add(mv)
                uniq_moves.append(mv)

            # now = st.timer.elapsed_sec
            # if (now - last_ts_log_sec >= ts_log_every_sec) or (
            #     it % ts_log_every_iters == 0
            # ):
            #     logging.info(
            #         "[PRTS][TS] heartbeat: iter=%d, ts_it=%d/%d, cur=%d, best=%d, blocks=%d, moves=%d, tabu=%d",
            #         st.iter_idx,
            #         it,
            #         iters,
            #         cur_fit,
            #         best_fit,
            #         len(blocks),
            #         len(uniq_moves),
            #         len(tabu),
            #     )
            #     last_ts_log_sec = now

            # evaluate all candidate neighbors, select by tabu/aspiration
            best_non_tabu = None  # (fit, enc, mv, start, end, loc)
            best_asp = None  # best (fit < aspiration_ub) even if tabu

            for mv in uniq_moves:
                if self._time_up():
                    break
                s_idx = _mv_stage_idx(mv)
                # 1) stage만 바꾼다 (O(stage size))
                changed_stage = _apply_move_stage_only(cur[s_idx], mv)
                # 2) rapid eval: cur + cache를 재사용해서 cmax만 얻는다 (O(remaining stages))
                new_cache = self.evaluator.rapid_eval_after_stage_change_ce(
                    enc=cur,
                    cache=cache,
                    changed_stage_idx=s_idx,
                    changed_stage=changed_stage,
                )
                fit = new_cache.cmax
                # 3) tabu/aspiration 판단은 fit로 한다
                is_tabu = tabu.get(mv, -1) >= it

                if fit < aspiration_ub and (best_asp is None or fit < best_asp[0]):
                    # 여기서는 nxt_enc를 아직 만들지 않아도 됨. "선택된 mv"만 저장해두면 됨.
                    best_asp = (fit, mv, s_idx, changed_stage, new_cache)

                if (not is_tabu) and (best_non_tabu is None or fit < best_non_tabu[0]):
                    best_non_tabu = (fit, mv, s_idx, changed_stage, new_cache)

            chosen = best_asp if best_asp is not None else best_non_tabu
            if chosen is None:
                break

            nxt_fit, chosen_mv, s_idx, chosen_stage, chosen_cache = chosen
            # cur을 실제로 갱신할 때만 enc를 만든다(=stage replace 1회)
            cur = _replace_stage(cur, s_idx, chosen_stage)
            cache = chosen_cache

            # maps는 next iteration에서 blocks 생성 때문에 필요하니까, 여기서는 full eval 1번은 해야 함.
            cur_fit, cur_start, cur_end, cur_loc = self._eval_complete_encoding(cur)
            assert cur_fit == cache.cmax
            tabu[chosen_mv] = it + tabu_tenure

            if cur_fit < best_fit:
                best, best_fit = cur, cur_fit
                # keep aspiration threshold in sync when we found a new incumbent
                aspiration_ub = min(aspiration_ub, best_fit)
                # logging.info(
                #     "[PRTS][TS] improved local best: iter=%d, ts_it=%d/%d, best=%d",
                #     st.iter_idx,
                #     it,
                #     iters,
                #     best_fit,
                # )

        # store best_fit in state cache for later phases
        st.complete_eval_cache[best] = best_fit
        return best

    # -------------------------
    # Population management
    # -------------------------
    def _pick_bottleneck_stage_idx(self) -> int:
        """
        Deterministic bottleneck stage selector for perm_view projection.

        Policy (stable, cheap, and instance-only):
        - estimate stage load = sum_j p(stage, j) / (#machines at stage)
        - pick argmax load (ties -> smallest stage index)

        This matches the idea that bottleneck stages dominate Cmax behavior,
        so their within-stage order is the best 1D "view" for Kendall–Tau.
        """
        stage_ids = list(self.stage_2_machines_map.keys())

        best_idx = 0
        best_score = None
        for s_idx, s in enumerate(stage_ids):
            m_cnt = max(1, len(self.stage_2_machines_map[s]))
            total_p = 0
            for j in self.job_id_list:
                total_p += int(self.stage_2_job_2_p_dict[s][j])
            score = total_p / m_cnt

            if best_score is None or score > best_score:
                best_score = score
                best_idx = s_idx

        return best_idx

    def _perm_view_of(self, enc: CompleteEncoding) -> tuple[str, ...]:
        """Return cached permutation-view projection of a CE (selection/similarity/PR only)."""
        st = self._require_state()
        pv = st.perm_view_cache.get(enc)
        if pv is None:
            pv = self._complete_to_perm_view(enc)
            st.perm_view_cache[enc] = pv
        return pv

    def _complete_to_perm_view(self, enc: CompleteEncoding) -> tuple[str, ...]:
        """
        Project a complete encoding to a permutation "view" for selection/similarity/PR.

        Recommended projection:
        - pick a fixed bottleneck stage k*
        - sort jobs by their start time at stage k* in the forward semi-active schedule
        - tie-break by job id (deterministic)

        Notes:
        - We avoid building a schedule object; we reuse _eval_complete_encoding.
        - This is a *view* only; objective/caching/uniqueness must remain complete-based.
        """
        # Cache stage choice once per run if you want (recommended):
        #   if not hasattr(self, "_perm_view_stage_idx"):
        #       self._perm_view_stage_idx = self._pick_bottleneck_stage_idx()
        #   k = self._perm_view_stage_idx
        #
        # If you don't want state, just compute each time (slower but simple):
        k = getattr(self, "_perm_view_stage_idx", None)
        if k is None:
            k = self._pick_bottleneck_stage_idx()
            self._perm_view_stage_idx = k  # safe to set once

        _cmax, start_map, _end_map, _loc_map = self._eval_complete_encoding(enc)

        # start times at chosen stage
        def key(job: str) -> tuple[int, str]:
            # start_map should always have (k, job) for legal encodings
            st = start_map.get((k, job), 0)
            return (int(st), job)  # tie-break by job id

        return tuple(sorted(self.job_id_list, key=key))

    def _try_accept(self, sol: CompleteEncoding) -> None:
        st = self._require_state()
        if sol in st.fitness_ce:
            return

        fit = self._evaluate_complete_encoding(sol)

        # replace worst if better (population stores CE)
        worst_ce = st.population[-1]
        worst_fit = st.fitness_ce[worst_ce]
        if fit >= worst_fit:
            logging.info(
                "[PRTS] rejected (candid fit %d not better than worst %d)",
                fit,
                worst_fit,
            )
            return
        logging.info(
            "[PRTS] accepted (candid fit %d better than worst %d)", fit, worst_fit
        )

        st.population[-1] = sol
        st.fitness_ce[sol] = fit
        del st.fitness_ce[worst_ce]
        st.perm_view_cache.pop(worst_ce, None)
        # Cache the permutation view for the accepted CE (used downstream in selection/PR)
        st.perm_view_cache.setdefault(sol, self._complete_to_perm_view(sol))
        self._sort_population()

    def _sort_population(self) -> None:
        st = self._require_state()
        st.population.sort(key=lambda ce: st.fitness_ce[ce])

    # -------------------------
    # Evaluate / best logging
    # -------------------------
    def _evaluate_perm(self, perm: tuple[str, ...]) -> int:
        st = self._require_state()
        if perm in st.perm_eval_cache:
            return st.perm_eval_cache[perm]
        ce = self._perm_to_complete_encoding(perm)
        obj = self._evaluate_complete_encoding(ce)
        st.perm_eval_cache[perm] = obj
        return obj

    def _evaluate_complete_encoding(self, ce: CompleteEncoding) -> int:
        st = self._require_state()
        if ce in st.complete_eval_cache:
            return st.complete_eval_cache[ce]
        sched: HybridFlowshopSchedule = self._dispatch_complete_encoding_to_schedule(ce)
        obj = sched.makespan
        st.complete_eval_cache[ce] = obj
        return obj

    def _update_best_and_log(self, note: str) -> bool:
        st = self._require_state()
        if not st.population:
            return False

        cand_ce = st.population[0]
        cand_fit = st.fitness_ce[cand_ce]
        improved = st.best_fit is None or cand_fit < st.best_fit

        if improved:
            st.best_sol = cand_ce
            st.best_fit = cand_fit
            t = st.timer.elapsed_sec
            st.sub_obj_store.add_obj_value(t, int(cand_fit), is_maximize=False)
            # st.sub_obj_store.add_last_timestamp_note(note, obj_value_is_valid=True)
            logging.info("[PRTS] improved best=%.3f at t=%.3fs (%s)", cand_fit, t, note)

        return improved

    # -------------------------
    # Utilities
    # -------------------------
    def _compute_machine_count(self) -> int:
        return sum(len(mcs) for mcs in self.stage_2_machines_map.values())

    def _dispatch_perm_to_schedule(
        self, perm: tuple[str, ...]
    ) -> HybridFlowshopSchedule:
        ce = self._perm_to_complete_encoding(perm)
        return self._dispatch_complete_encoding_to_schedule(ce)

    def _dispatch_complete_encoding_to_schedule(
        self, enc: CompleteEncoding
    ) -> HybridFlowshopSchedule:
        stage_ids: list[str] = self.instance.stage_id_list

        # Internal states
        job_2_last_comp_time: dict[str, int] = {j: 0 for j in self.job_id_list}
        schedule: HybridFlowshopSchedule = (
            HybridFlowshopSchedule.from_stage_name_2_mc_name_list_map(
                self.stage_2_machines_map
            )
        )

        for s_idx, s in enumerate(stage_ids):
            stage = enc[s_idx]
            for m_idx, mc in enumerate(self.stage_2_machines_map[s]):
                seq = stage[m_idx]
                for j in seq:
                    p = self.stage_2_job_2_p_dict[s][j]
                    stage_in_schedule = schedule.get_stage_by_name(s)
                    machine_in_schedule = stage_in_schedule.get_machine_by_name(mc)
                    start_time = machine_in_schedule.get_earliest_start_time(
                        p, release_t=job_2_last_comp_time[j]
                    )
                    if start_time < job_2_last_comp_time[j]:
                        start_time = job_2_last_comp_time[j]
                    schedule.get_stage_by_name(s).add_operation(
                        HybridFlowshopOperation(
                            job_name=j,
                            stage_name=s,
                            mc_name=mc,
                            start=start_time,
                            end=start_time + p,
                        )
                    )
                    job_2_last_comp_time[j] = start_time + p
        return schedule


def get_kendall_tau_similarity(perm1: tuple[str, ...], perm2: tuple[str, ...]) -> float:
    n = len(perm1)
    if n != len(perm2):
        raise ValueError("Permutations must have the same length.")
    if n <= 1:
        return 1.0
    # job -> position in pos2
    pos2: dict[str, int] = {job: idx for idx, job in enumerate(perm2)}
    # list of positions in perm2 according to order in perm1
    arr = [pos2[job] for job in perm1]

    inv = 0
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            if ai > arr[j]:
                inv += 1
    max_inv = n * (n - 1) // 2
    return 1.0 - (inv / max_inv)


# ============================================================
# Rapid eval integration helpers (stage-only apply)
# ============================================================


def _mv_stage_idx(mv: tuple) -> int:
    """
    mv formats in your code:
      ("N7", stage_idx, machine_idx, pos_a, pos_b)
      ("KSWAP", stage_idx, m_a, pos_a, m_b, pos_b)
    """
    return int(mv[1])


def _apply_move_stage_only(
    stage: tuple[tuple[str, ...], ...],
    mv: tuple,
) -> tuple[tuple[str, ...], ...]:
    """
    Apply mv within a single stage and return the new stage (immutable tuples).
    Does NOT touch other stages.

    stage: stage[machine_idx] = tuple(job,...)
    """
    kind = mv[0]
    s_stage = [list(seq) for seq in stage]  # mutable copy of THIS stage only

    if kind == "N7":
        _, _s, m, a, b = mv
        m = int(m)
        a = int(a)
        b = int(b)
        s_stage[m][a], s_stage[m][b] = s_stage[m][b], s_stage[m][a]
        return tuple(tuple(seq) for seq in s_stage)

    if kind == "KSWAP":
        _, _s, ma, pa, mb, pb = mv
        ma = int(ma)
        mb = int(mb)
        pa = int(pa)
        pb = int(pb)
        s_stage[ma][pa], s_stage[mb][pb] = s_stage[mb][pb], s_stage[ma][pa]
        return tuple(tuple(seq) for seq in s_stage)

    raise ValueError(f"Unknown move kind: {kind}")


def _replace_stage(
    enc: CompleteEncoding,
    s_idx: int,
    new_stage: tuple[tuple[str, ...], ...],
) -> CompleteEncoding:
    """
    Return a new CompleteEncoding with only stage s_idx replaced.
    """
    enc_list = list(enc)
    enc_list[s_idx] = new_stage
    return tuple(enc_list)


# Logging
from collections import Counter


def log_perm_health(logger, tag: str, perm: tuple[str, ...], all_jobs: tuple[str, ...]):
    c = Counter(perm)
    missing = [j for j in all_jobs if c[j] == 0]
    dup = [j for j in all_jobs if c[j] > 1]
    extra = [j for j in c.keys() if j not in set(all_jobs)]
    logger.info(
        f"[{tag}] perm len={len(perm)} unique={len(c)} "
        f"missing={len(missing)} dup={len(dup)} extra={len(extra)} "
        f"head={perm[:20]} tail={perm[-20:]}"
    )
    if missing or dup or extra:
        logger.error(
            f"[{tag}] perm bad: missing={missing[:10]} dup={dup[:10]} extra={extra[:10]}"
        )


def log_ce_health(
    logger, tag: str, ce, all_jobs: tuple[str, ...], stage_ids, stage_2_machines
):
    job_set = set(all_jobs)
    for s_idx, stage_id in enumerate(stage_ids):
        stage = ce[s_idx]
        exp_m = len(stage_2_machines[stage_id])
        seen = []
        for seq in stage:
            seen.extend(list(seq))
        c = Counter(seen)
        missing = [j for j in all_jobs if c[j] == 0]
        dup = [j for j in all_jobs if c[j] > 1]
        logger.info(
            f"[{tag}] stage={stage_id} machines={len(stage)}/{exp_m} "
            f"jobs_seen={len(seen)} uniq={len(c)} missing={len(missing)} dup={len(dup)}"
        )
        if (
            len(stage) != exp_m
            or missing
            or dup
            or (set(seen) != job_set)
            or (len(seen) != len(all_jobs))
        ):
            lens = [len(seq) for seq in stage]
            logger.error(
                f"[{tag}] CE illegal at stage={stage_id}: "
                f"machines={len(stage)}/{exp_m} missing={missing[:10]} dup={dup[:10]} "
                f"seq_lens={lens}"
            )
            # 여기서 바로 raise 해도 됨 (원인 조기 포착)
            # raise ValueError("Illegal CE")


def log_operator_step(logger, ind: int, it: int, op_name: str, details: str = ""):
    logger.info(f"[init] ind={ind} it={it} op={op_name} {details}")
