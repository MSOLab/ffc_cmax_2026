from datetime import datetime

from main import main

if __name__ == "__main__":
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main()
        # 결과를 cumtime 기준 내림차순으로 정렬하고 파일에 저장
        with open(f"profile_result_main_all_{timestamp}.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f)
            ps.strip_dirs().sort_stats("cumulative").print_stats()
    finally:
        pr.disable()
