"""Tests for MachineDispatcher class and related schedule_lite methods."""

import pandas as pd
import pytest
from schore.parameters import JobStageProcessingTimeManager
from schore.parameters_examples import HybridFlowshopParameters

from hybridflowshop.dispatcher import MachineDispatcher
from hybridflowshop.schedule_lite import HybridFlowshopLiteSchedule


@pytest.fixture
def sample_processing_time_df():
    """Fixture to provide a sample processing time DataFrame."""
    return pd.DataFrame(
        [
            [10, 20, 15],  # Job j0: stages i0, i1, i2
            [12, 18, 22],  # Job j1: stages i0, i1, i2
            [8, 25, 16],  # Job j2: stages i0, i1, i2
        ]
    )


@pytest.fixture
def sample_processing_time_manager(sample_processing_time_df):
    """Fixture to provide a JobStageProcessingTimeManager instance."""
    return JobStageProcessingTimeManager(
        name="SampleProcessingTime", df=sample_processing_time_df
    )


@pytest.fixture
def sample_job_ids():
    """Fixture to provide sample job IDs."""
    return ["j0", "j1", "j2"]


@pytest.fixture
def sample_stage_ids():
    """Fixture to provide sample stage IDs."""
    return ["i0", "i1", "i2"]


@pytest.fixture
def sample_stage_2_machines_map():
    """Fixture to provide a sample stage to machines mapping."""
    return {
        "i0": ["m0", "m1"],
        "i1": ["m0", "m1"],
        "i2": ["m0", "m1"],
    }


@pytest.fixture
def sample_hybrid_flowshop_params(
    sample_job_ids,
    sample_stage_ids,
    sample_processing_time_manager,
    sample_stage_2_machines_map,
):
    """Fixture to provide a HybridFlowshopParameters instance."""
    return HybridFlowshopParameters(
        name="TestInstance",
        job_id_list=sample_job_ids,
        stage_id_list=sample_stage_ids,
        stage_2_machines_map=sample_stage_2_machines_map,
        p_manager=sample_processing_time_manager,
    )


def create_hfs_instance(
    name: str,
    job_ids: list[str],
    stage_ids: list[str],
    stage_2_machines_map: dict[str, list[str]],
    job_2_stage_2_p: dict[str, dict[str, int]],
) -> HybridFlowshopParameters:
    """Helper function to create a HybridFlowshopParameters instance.

    Args:
        name: Instance name
        job_ids: List of job IDs
        stage_ids: List of stage IDs
        stage_2_machines_map: Stage to machines mapping
        job_2_stage_2_p: Job to stage to processing time mapping

    Returns:
        HybridFlowshopParameters instance
    """
    # Create processing time DataFrame
    df = pd.DataFrame(
        [[job_2_stage_2_p[job][stage] for stage in stage_ids] for job in job_ids]
    )
    p_manager = JobStageProcessingTimeManager(name=f"{name}_P", df=df)
    return HybridFlowshopParameters(
        name=name,
        job_id_list=job_ids,
        stage_id_list=stage_ids,
        stage_2_machines_map=stage_2_machines_map,
        p_manager=p_manager,
    )
