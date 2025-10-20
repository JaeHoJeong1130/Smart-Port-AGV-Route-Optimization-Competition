import pandas as pd

dPATH = '/home/jjh/Project/competition/15_sea2/data/'

# --- 1. 데이터 파일 불러오기 ---
# 사용자가 업로드한 agv.csv와 task.csv 파일을 DataFrame으로 읽어옵니다.
try:
    agv_df = pd.read_csv(dPATH + 'agv.csv')
    task_df = pd.read_csv(dPATH + 'task.csv')
    print("✅ agv.csv와 task.csv 파일을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("🚨 오류: 'agv.csv' 또는 'task.csv' 파일이 코드와 같은 폴더에 있는지 확인해주세요.")
    # 파일이 없는 경우를 대비해 빈 데이터프레임 생성
    agv_df = pd.DataFrame()
    task_df = pd.DataFrame()

# --- 2. DEPOT(창고) 정보 추가하기 ---
# 모든 AGV의 출발점이자 도착점인 DEPOT 정보를 task 데이터에 추가합니다.
# DEPOT는 좌표 (0,0)에 위치하며, 다른 요구사항은 없습니다.
if not task_df.empty:
    depot_info = {
        'task_id': 'DEPOT',
        'x': 0,
        'y': 0,
        'service_time': 0,
        'demand': 0,
        'deadline': float('inf') # 마감 기한이 무한대임을 의미
    }
    # 기존 task_df 맨 앞에 DEPOT 정보를 추가합니다.
    task_df = pd.concat([pd.DataFrame([depot_info]), task_df], ignore_index=True)
    print("✅ DEPOT(창고) 정보를 Task 데이터에 추가했습니다.")

# --- 3. 핵심 계산 함수 정의 ---
# 문제의 규칙에 따라 두 지점 사이의 거리를 계산하는 함수를 만듭니다.
def manhattan_distance(p1_x, p1_y, p2_x, p2_y):
    """두 점 (p1_x, p1_y)와 (p2_x, p2_y) 사이의 맨해튼 거리를 계산합니다."""
    return abs(p1_x - p2_x) + abs(p1_y - p2_y)

print("✅ 맨해튼 거리 계산 함수를 정의했습니다.")

# --- 4. 불러온 데이터 확인 ---
# 데이터가 어떻게 생겼는지 상위 5개 행을 출력하여 확인합니다.
print("\n--- AGV 데이터 (상위 5개) ---")
print(agv_df.head())

print("\n--- Task 데이터 (DEPOT 포함, 상위 5개) ---")
print(task_df.head())

import numpy as np
from sklearn.cluster import KMeans
import warnings

# KMeans 실행 시 발생하는 경고를 무시합니다.
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- 2단계: K-Means 클러스터링으로 초기 해 생성 시작 ---")

# --- 1. 클러스터링할 데이터 준비 ---
# DEPOT를 제외한 실제 작업들의 좌표(x, y)만 추출합니다.
tasks_for_clustering = task_df[task_df['task_id'] != 'DEPOT']
task_coordinates = tasks_for_clustering[['x', 'y']].values

# --- 2. K-Means 모델 설정 및 실행 ---
# K값(클러스터의 개수)은 전체 AGV의 수로 설정합니다.
num_agvs = len(agv_df)
print(f"AGV의 수: {num_agvs}개. 이 값을 K로 사용하여 클러스터링을 실행합니다.")

# K-Means 모델을 생성하고 학습시킵니다.
# random_state를 고정하면 실행할 때마다 항상 같은 결과가 나옵니다.
kmeans = KMeans(n_clusters=num_agvs, random_state=42, n_init=10)
tasks_for_clustering['cluster'] = kmeans.fit_predict(task_coordinates)

# --- 3. 클러스터링 결과 확인 및 AGV에 할당 ---
# 각 클러스터(작업 묶음)에 어떤 작업들이 포함되었는지 확인합니다.
initial_solution = {}
for i in range(num_agvs):
    # i번째 클러스터에 속한 task_id들을 리스트로 가져옵니다.
    assigned_tasks = tasks_for_clustering[tasks_for_clustering['cluster'] == i]['task_id'].tolist()
    
    # i번째 AGV의 ID를 가져와서 할당합니다. (예: A001, A002, ...)
    agv_id = agv_df.iloc[i]['agv_id']
    initial_solution[agv_id] = assigned_tasks

print("\n✅ K-Means 클러스터링을 통해 생성된 초기 해(AGV별 작업 할당):")

# 결과가 너무 길어질 수 있으므로, 처음 5개 AGV의 할당 결과만 출력
for i, (agv_id, tasks) in enumerate(initial_solution.items()):
    if i >= 5:
        print("...")
        break
    print(f"- {agv_id}: {tasks}")