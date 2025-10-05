# ==============================================================================
# Advanced Cherry-Picking with Validation
# ==============================================================================
# 설명: 'sub' 폴더에 있는 여러 개의 제출(CSV) 파일들을 분석하여,
#      각 AGV별로 가장 점수가 좋았던 '최고의 경로'를 선택합니다.
#      그 후, 발생할 수 있는 'Task 중복' 및 'Task 누락' 문제를
#      지능적으로 해결하여 완벽한 최종 해를 생성합니다.
# ==============================================================================

import os
import glob
import pandas as pd
from collections import defaultdict
import importlib.util
from datetime import datetime
import csv
import copy

# --- ⚙️ CONFIGURATION (사용자 설정) ---

dPATH = '/home/jjh/Project/competition/15_sea2/data/'
sPATH = '/home/jjh/Project/competition/15_sea2/sub/'

# VrpData, Solution, AlnsSolver, generate_submission_file 클래스/함수가 포함된 파일
SOLVER_FILE_NAME = '/home/jjh/Project/competition/15_sea2/sea2_01_02_local_search_solver_fast.py' 

# --- 코드 시작 ---

def load_solver_module(file_path):
    """지정된 파이썬 파일을 모듈로 동적 로드하는 함수"""
    try:
        spec = importlib.util.spec_from_file_location("solver_module", file_path)
        solver_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solver_module)
        print(f"✅ '{file_path}' 파일에서 기존 솔버 모듈을 성공적으로 불러왔습니다.")
        return solver_module
    except FileNotFoundError:
        print(f"🚨 오류: 솔버 파일 '{file_path}'을(를) 찾을 수 없습니다. SOLVER_FILE_NAME을 확인해주세요.")
        exit()

# 솔버 모듈 로드
solver_module = load_solver_module(SOLVER_FILE_NAME)
VrpData = solver_module.VrpData
Solution = solver_module.Solution
AlnsSolver = solver_module.AlnsSolver
generate_submission_file = solver_module.generate_submission_file

def parse_submission_to_routes(file_path):
    """제출 CSV 파일을 읽어 routes 딕셔너리로 파싱하는 함수"""
    routes = {}
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            agv_id = row['agv_id']
            route_str = row['route'].strip('"')
            tasks = [task for task in route_str.split(',') if task != 'DEPOT' and task]
            routes[agv_id] = tasks
    except Exception as e:
        print(f"- '{os.path.basename(file_path)}' 파일 파싱 중 오류 발생: {e}")
        return None
    return routes

def advanced_cherry_pick_and_repair(data_model, solver_instance):
    """'고급 체리피킹' 메인 함수"""
    
    submission_files = glob.glob(os.path.join(sPATH, '*.csv'))
    if not submission_files:
        print(f"🚨 '{sPATH}' 폴더에서 제출 파일을 찾을 수 없습니다.")
        return None
    
    print(f"\n총 {len(submission_files)}개의 제출 파일을 분석합니다...")

    # --- 1단계: AGV별 최고 경로 '체리피킹' ---
    print("\n--- 1단계: AGV별 최고 경로 선별 시작 ---")
    cherry_picked_routes = {}
    
    for agv_id in data_model.agv_info.keys():
        best_route_for_agv = []
        min_score = float('inf')
        best_file = "None"
        
        for file in submission_files:
            routes = parse_submission_to_routes(file)
            if routes is None:
                continue
            
            route = routes.get(agv_id, [])
            score = solver_instance._calculate_single_route_score(agv_id, route)
            
            if score < min_score:
                min_score = score
                best_route_for_agv = route
                best_file = os.path.basename(file)
        
        cherry_picked_routes[agv_id] = best_route_for_agv
        # print(f"  - {agv_id}: {best_file}에서 경로 선택 (Score: {min_score:.2f})")

    print("✅ AGV별 최고 경로 선별 완료!")
    
    # --- 2단계: Task 유효성 검사 (중복/누락) ---
    print("\n--- 2단계: Task 중복 및 누락 검사 시작 ---")
    
    task_assignments = defaultdict(list)
    all_tasks_in_routes = []
    
    for agv_id, route in cherry_picked_routes.items():
        for task in route:
            task_assignments[task].append(agv_id)
            all_tasks_in_routes.append(task)
            
    duplicate_tasks = {task: agvs for task, agvs in task_assignments.items() if len(agvs) > 1}
    
    original_tasks = set(data_model.task_info.keys()) - {'DEPOT'}
    missing_tasks = list(original_tasks - set(all_tasks_in_routes))
    
    print(f"  - 중복 할당된 Task 수: {len(duplicate_tasks)}")
    print(f"  - 누락된 Task 수: {len(missing_tasks)}")

    final_routes = copy.deepcopy(cherry_picked_routes)

    # --- 3단계: Task 중복 문제 해결 ---
    if duplicate_tasks:
        print("\n--- 3단계: Task 중복 문제 해결 시작 ---")
        for task, agvs in duplicate_tasks.items():
            best_agv_to_keep = None
            min_score_without_task = float('inf')
            
            # 이 Task를 어떤 AGV가 '유지'하는 것이 가장 효율적인지 계산
            for agv_id in agvs:
                original_route = final_routes[agv_id]
                route_without_task = [t for t in original_route if t != task]
                
                # Task를 뺐을 때의 점수
                score_without_task = solver_instance._calculate_single_route_score(agv_id, route_without_task)
                
                if score_without_task < min_score_without_task:
                    min_score_without_task = score_without_task
                    best_agv_to_keep = agv_id

            # 최고의 AGV를 제외한 나머지 AGV 경로에서 이 Task를 제거
            for agv_id in agvs:
                if agv_id != best_agv_to_keep:
                    final_routes[agv_id] = [t for t in final_routes[agv_id] if t != task]
                    print(f"  - {task}: {agv_id}에서 제거. {best_agv_to_keep}가 유지.")
        print("✅ Task 중복 문제 해결 완료.")

    # --- 4단계: Task 누락 문제 해결 ---
    if missing_tasks:
        print("\n--- 4단계: Task 누락 문제 해결 시작 ---")
        print(f"Regret Insertion을 사용하여 {len(missing_tasks)}개의 누락된 Task를 삽입합니다...")
        
        # 기존 솔버의 강력한 삽입 연산자 재활용
        solver_instance.regret_insertion(missing_tasks, routes_to_modify=final_routes)
        print("✅ Task 누락 문제 해결 완료.")

    print("\n✅ 모든 재구성 및 보정 작업 완료!")
    final_solution = Solution(final_routes, data_model, solver_instance)
    
    return final_solution


if __name__ == '__main__':
    # ==============================================================================
    # 🚀 하이브리드 솔버 파이프라인 실행부
    # ==============================================================================
    
    # --- ⚙️ CONFIGURATION (하이브리드 파이프라인 설정) ---
    
    # 1단계(재구성)에서 사용할 엘리트 링크 임계값
    LINK_FREQUENCY_THRESHOLD = 0.5 
    
    # 2단계(심층 탐색)에서 실행할 ALNS 반복 횟수
    FINAL_ALNS_ITERATIONS = 100000

    print("="*60)
    print("🚀 하이브리드 솔버 파이프라인 시작")
    print("="*60)

    # --- 공통 준비 단계: 데이터 및 헬퍼(helper) 솔버 인스턴스 생성 ---
    # 1. 데이터 모델 로드
    data = VrpData(agv_csv='agv.csv', task_csv='task.csv')
    
    # 2. 재구성 단계에서 내부 함수(regret_insertion 등)를 사용하기 위한
    #    '헬퍼' ALNS Solver 인스턴스 생성
    dummy_initial_routes = {agv_id: [] for agv_id in data.agv_info.keys()}
    helper_solver = AlnsSolver(data_model=data, initial_solution_routes=dummy_initial_routes)
    
    
    # ==============================================================================
    # --- PHASE 1: 엘리트 초기 해 생성 ---
    # ==============================================================================
    print("\n--- PHASE 1: 여러 솔루션으로부터 '엘리트 초기 해' 생성 시작 ---")
    
    # advanced_cherry_pick_and_repair 함수를 사용하여 고품질 초기 해 생성
    # (내부적으로는 유전 정보 재구성과 유사한 효과를 냄)
    elite_initial_solution = advanced_cherry_pick_and_repair(data, helper_solver)
    
    if not elite_initial_solution:
        print("🚨 엘리트 초기 해 생성에 실패하여 파이프라인을 중단합니다.")
        exit()

    elite_score = elite_initial_solution.score
    print(f"\n✨ 엘리트 초기 해 생성 완료! (점수: {elite_score:.2f})")
    print("--- PHASE 1 완료 ---\n")


    # ==============================================================================
    # --- PHASE 2: ALNS 심층 최적화 ---
    # ==============================================================================
    print("--- PHASE 2: 생성된 엘리트 해를 기반으로 ALNS 심층 최적화 시작 ---")
    print(f"시작점수 {elite_score:.2f}에서 {FINAL_ALNS_ITERATIONS}번의 추가 탐색을 진행합니다.")

    # 1단계에서 얻은 엘리트 해의 경로(routes)를 추출
    elite_initial_routes = elite_initial_solution.routes

    # "진짜" ALNS 솔버를 '엘리트 초기 해'와 함께 생성하여 'Warm Start'
    final_solver = AlnsSolver(data_model=data, initial_solution_routes=elite_initial_routes)

    # ALNS 솔버 실행
    final_solver.run(iterations=FINAL_ALNS_ITERATIONS)

    best_solution = final_solver.best_solution
    final_score = best_solution.score

    print(f"\n✨ 심층 최적화 완료! (최종 점수: {final_score:.2f})")
    print(f"개선된 점수: {elite_score - final_score:.2f}")
    print("--- PHASE 2 완료 ---\n")


    # ==============================================================================
    # --- 최종 결과 저장 ---
    # ==============================================================================
    print("--- 최종 제출 파일 생성 ---")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    submission_filename = f"submission_final_{timestamp}.csv"
    generate_submission_file(best_solution, data, submission_filename)

    print("\n="*60)
    print("🎉 하이브리드 솔버 파이프라인 성공적으로 종료")
    print("="*60)