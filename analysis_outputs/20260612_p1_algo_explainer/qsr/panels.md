# QSR panels (Quantize-Schedule-Reconstruct, tau-coarsening)

Low-resolution-proxy story: solve a cheap 1/tau copy, learn the operation
ORDER, upscale it back to full resolution, then repair.

tau=25 (설명용 과장값; 논문/프로덕션은 tau=5). 슬라이드 전용 figure이며 thesis
실험 결과 및 data/ffc_cmax/references.md 와 무관 (plan §9의 의도적·문서화된 예외).

압축 가시화: 모든 패널은 자체 makespan auto-fit (공유축 폐기). tau 압축은 x축 눈금
숫자로 읽는다 -- real 패널 step=100 (100,200,...) vs surrogate 패널 step=4
(4,8,12,...). 눈금 간격비 100:4 = 25 = tau 가 압축률이며, 막대 너비는 양쪽이 동일.

- tau = 25 (surrogate width = ceil(p / 25)).
- step_01 axis-reference    : 426 (real p_ij schedule; NOT a baseline -- QSR is a pure constructor)
- surrogate dispatch        : 21
- surrogate CP before->after: 21 -> 21 (status=OPTIMAL)
- restore machine-sequence  : 443
- restore stage-sequence    : 443  (tie -> machine-sequence adopted (production list order))
- QSR build -> repair        : 443 -> 420 (reconstruction polished; QSR's own trajectory)

## Panels

- `step_01_quantize_original.svg`: QUANTIZE (left): a full-resolution (real p_ij) schedule, shown ONLY to anchor the real-time 100-axis. QSR quantizes the INSTANCE (p_ij -> ceil(p/tau)), not this schedule -- it is a pure constructor with no baseline, so this makespan is NOT a reference value (real-resolution axis, auto-fit; x ticks step 100). Caption: '실해상도(real-time) 100축 예시 -- 처리시간만 1/tau로 다운샘플(순서 비교 대상 아님)'.
- `step_02_quantize_surrogate.svg`: QUANTIZE (right): surrogate at 1/tau resolution (tau=25, width=ceil(p/25)). makespan=21 (surrogate axis, auto-fit; x ticks step 4 (1/tau resolution)). Bars same width as the real panel; only the x-tick numbers are 25x apart. Caption: '원본을 1/tau 해상도로 다운샘플; x눈금 4단위 (1/tau 해상도, tau=25)'.
- `step_03_schedule_surrogate_before.svg`: SCHEDULE (surrogate, before): first surrogate-CP incumbent. makespan=21 (surrogate axis, auto-fit; x ticks step 4 (1/tau resolution)).
- `step_04_schedule_surrogate_after.svg`: SCHEDULE (surrogate, after): final surrogate-CP optimum (status=OPTIMAL). makespan=21 (surrogate axis, auto-fit; x ticks step 4 (1/tau resolution)). Caption: '작은 탐색공간 -> CP가 좋은 operation 순서를 싸게 학습'.
- `step_05_reconstruct_machine_sequence.svg`: RECONSTRUCT (machine-sequence restore): replay the surrogate per-machine op order at original p_ij. makespan=443 (real-resolution axis, auto-fit; x ticks step 100).
- `step_06_reconstruct_stage_sequence.svg`: RECONSTRUCT (stage-sequence restore): replay the surrogate per-stage op order at original p_ij. makespan=443 (real-resolution axis, auto-fit; x ticks step 100). tie -> machine-sequence adopted (production list order). Caption: '학습한 순서를 원해상도로 업스케일 (2가지 복원, better 채택)'.
- `step_07_repair_before.svg`: REPAIR (before): the reconstruction (machine_sequence) is QSR's own first incumbent (pure constructor -- this IS the schedule QSR built, not an improvement over anything). makespan=443 (real-resolution axis, auto-fit; x ticks step 100).
- `step_08_repair_after.svg`: REPAIR (after): critical-cone local search result. makespan=420 (real-resolution axis, auto-fit; x ticks step 100). Caption: '임계경로 국소수선으로 마무리'.
