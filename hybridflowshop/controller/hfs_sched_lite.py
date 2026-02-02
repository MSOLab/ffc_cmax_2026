from dataclasses import dataclass

JobId = str
MachineId = str
CompleteEncoding = tuple[tuple[tuple[str, ...], ...], ...]


@dataclass
class ForwardStageCache:
    """
    Forward schedule cache for ONE complete encoding.

    Stores:
      - stage_end[j][job] = completion time of job at stage j
      - cmax
    """

    stage_end: list[dict[JobId, int]]  # len = num_stages, dict job->end
    cmax: int


class RapidEvaluatorZhou2024:
    """
    Rapid evaluator consistent with Zhou et al. (2024) 4.6.3:
      - neighborhood move occurs within a single stage
      - preceding stages unaffected
      - recompute only stage j and downstream stages to get exact makespan

    NOTE:
      This is an *exact* partial forward re-evaluation, not the paper's
      head/tail closed-form. It matches the same invariance assumption
      (only stage j changes), and is easy to integrate in TS.
    """

    def __init__(
        self,
        job_ids: list[JobId],
        stage_ids: list[str],
        stage_2_machines: dict[str, list[MachineId]],
        p_time: dict[str, dict[JobId, int]],  # p_time[stage][job]
        machine_2_stage: dict[MachineId, int],  # machine -> stage index (0..s-1)
    ) -> None:
        self.job_ids = job_ids
        self.stage_ids = stage_ids
        self.stage_2_machines = stage_2_machines
        self.p_time = p_time
        self.machine_2_stage = machine_2_stage
        self.s = len(stage_ids)

    # -------------------------
    # Core: schedule one stage
    # -------------------------
    def _schedule_stage_forward(
        self,
        stage_idx: int,
        machine_seqs_in_stage: dict[MachineId, list[JobId]],
        prev_stage_end: dict[JobId, int],
    ) -> dict[JobId, int]:
        """
        Given fixed machine sequences for this stage, produce completion times at this stage.

        Forward semi-active schedule:
          start = max(prev_stage_end[job], machine_available[machine])
          end = start + p(stage, job)
        """
        stage_id = self.stage_ids[stage_idx]
        machines = self.stage_2_machines[stage_id]

        # machine availability
        mach_avail: dict[MachineId, int] = {m: 0 for m in machines}
        stage_end: dict[JobId, int] = {}

        for m in machines:
            seq = machine_seqs_in_stage.get(m, [])
            t = mach_avail[m]
            for job in seq:
                st = max(prev_stage_end.get(job, 0), t)
                en = st + int(self.p_time[stage_id][job])
                t = en
                stage_end[job] = en
            mach_avail[m] = t

        # Optional sanity: every job should appear exactly once in a legal encoding at each stage
        # (Zhou 2024 4.2). You may assert this in debug builds.
        # if len(stage_end) != len(self.job_ids):
        #     raise ValueError(f"Illegal encoding at stage {stage_id}: missing jobs")

        return stage_end

    # -------------------------
    # NEW: CE-compatible cache builder
    # -------------------------
    def build_forward_cache_ce(self, enc: CompleteEncoding) -> ForwardStageCache:
        stage_end: list[dict[JobId, int]] = []
        prev_end: dict[JobId, int] = {j: 0 for j in self.job_ids}

        for s_idx, stage_id in enumerate(self.stage_ids):
            stage = enc[s_idx]  # tuple[machine_seq]
            # machine availability inside this stage
            mc_ready = [0] * len(stage)

            end_s: dict[JobId, int] = {}
            for m_idx, seq in enumerate(stage):
                t = mc_ready[m_idx]
                for job in seq:
                    st = max(prev_end[job], t)
                    en = st + int(self.p_time[stage_id][job])
                    t = en
                    end_s[job] = en
                mc_ready[m_idx] = t

            stage_end.append(end_s)
            prev_end = end_s

        cmax = max(prev_end.values()) if prev_end else 0
        return ForwardStageCache(stage_end=stage_end, cmax=cmax)

    # -------------------------
    # NEW: CE-compatible rapid eval after one-stage change
    # -------------------------
    def rapid_eval_after_stage_change_ce(
        self,
        enc: CompleteEncoding,
        cache: ForwardStageCache,
        changed_stage_idx: int,
        changed_stage: tuple[tuple[str, ...], ...],  # stage[machine]=seq
    ) -> ForwardStageCache:
        new_stage_end: list[dict[JobId, int]] = list(cache.stage_end)

        # prev completion times (stage j-1)
        if changed_stage_idx == 0:
            prev_end: dict[JobId, int] = {j: 0 for j in self.job_ids}
        else:
            prev_end = new_stage_end[changed_stage_idx - 1]

        # recompute stage j using changed_stage
        stage_id = self.stage_ids[changed_stage_idx]
        mc_ready = [0] * len(changed_stage)
        end_j: dict[JobId, int] = {}

        for m_idx, seq in enumerate(changed_stage):
            t = mc_ready[m_idx]
            for job in seq:
                st = max(prev_end[job], t)
                en = st + int(self.p_time[stage_id][job])
                t = en
                end_j[job] = en
            mc_ready[m_idx] = t

        new_stage_end[changed_stage_idx] = end_j
        prev_end = end_j

        # downstream stages: use enc's existing sequences, but new release times
        for s_idx in range(changed_stage_idx + 1, len(self.stage_ids)):
            stage_id = self.stage_ids[s_idx]
            stage = enc[s_idx]
            mc_ready = [0] * len(stage)
            end_s: dict[JobId, int] = {}

            for m_idx, seq in enumerate(stage):
                t = mc_ready[m_idx]
                for job in seq:
                    st = max(prev_end[job], t)
                    en = st + int(self.p_time[stage_id][job])
                    t = en
                    end_s[job] = en
                mc_ready[m_idx] = t

            new_stage_end[s_idx] = end_s
            prev_end = end_s

        cmax = max(prev_end.values()) if prev_end else 0
        return ForwardStageCache(stage_end=new_stage_end, cmax=cmax)

    # -------------------------
    # Helpers
    # -------------------------
    def _extract_stage_machine_seqs(
        self,
        enc_machine_seqs: dict[MachineId, list[JobId]],
        stage_idx: int,
    ) -> dict[MachineId, list[JobId]]:
        """
        Return machine->seq for machines belonging to stage_idx.
        """
        stage_id = self.stage_ids[stage_idx]
        out: dict[MachineId, list[JobId]] = {}
        for m in self.stage_2_machines[stage_id]:
            out[m] = list(enc_machine_seqs[m])
        return out


# -------------------------------------------------------
# Convenience: make "changed_stage_machine_seqs" by swap
# -------------------------------------------------------
def stage_swap_move(
    stage_machine_seqs: dict[MachineId, list[JobId]],
    m1: MachineId,
    i1: int,
    m2: MachineId,
    i2: int,
) -> dict[MachineId, list[JobId]]:
    """
    Swap two positions (possibly across machines) within the SAME stage.
    Returns a NEW dict with copied lists (no in-place mutation).
    """
    new_map = {m: list(seq) for m, seq in stage_machine_seqs.items()}
    a = new_map[m1][i1]
    b = new_map[m2][i2]
    new_map[m1][i1] = b
    new_map[m2][i2] = a
    return new_map


def stage_insert_move(
    stage_machine_seqs: dict[MachineId, list[JobId]],
    src_m: MachineId,
    src_idx: int,
    dst_m: MachineId,
    dst_idx: int,
) -> dict[MachineId, list[JobId]]:
    """
    Remove job at (src_m, src_idx) and insert into (dst_m, dst_idx) within the SAME stage.
    Returns a NEW dict with copied lists.
    """
    new_map = {m: list(seq) for m, seq in stage_machine_seqs.items()}
    job = new_map[src_m].pop(src_idx)
    new_map[dst_m].insert(dst_idx, job)
    return new_map


# -------------------------------------------------------
# Adapter (optional): if your CompleteEncoding is a tuple
# aligned with machine_id_list, convert to dict
# -------------------------------------------------------
def encoding_tuple_to_dict(
    machine_id_list: list[MachineId],
    enc_tuple: tuple[tuple[JobId, ...], ...],
) -> dict[MachineId, list[JobId]]:
    return {m: list(enc_tuple[k]) for k, m in enumerate(machine_id_list)}


def encoding_dict_to_tuple(
    machine_id_list: list[MachineId],
    enc_dict: dict[MachineId, list[JobId]],
) -> tuple[tuple[JobId, ...], ...]:
    return tuple(tuple(enc_dict[m]) for m in machine_id_list)
