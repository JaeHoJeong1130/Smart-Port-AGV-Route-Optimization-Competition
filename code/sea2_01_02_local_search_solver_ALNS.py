import pandas as pd
import numpy as np
import random
import copy
import json
import csv
from datetime import datetime

dPATH = '/home/jjh/Project/competition/15_sea2/data/'
sPATH = '/home/jjh/Project/competition/15_sea2/sub/'

# --- 1. 데이터 클래스 ---
class VrpData:
    def __init__(self, agv_csv, task_csv):
        self.agv_df = pd.read_csv(dPATH + agv_csv)
        task_df_orig = pd.read_csv(dPATH + task_csv)
        depot_info = {'task_id': 'DEPOT', 'x': 0, 'y': 0, 'service_time': 0, 'demand': 0, 'deadline': float('inf')}
        self.task_df = pd.concat([pd.DataFrame([depot_info]), task_df_orig], ignore_index=True)
        self.agv_info = self.agv_df.set_index('agv_id').to_dict('index')
        self.task_info = self.task_df.set_index('task_id').to_dict('index')

    def get_manhattan_distance(self, task1_id, task2_id):
        p1 = self.task_info[task1_id]
        p2 = self.task_info[task2_id]
        return abs(p1['x'] - p2['x']) + abs(p1['y'] - p2['y'])

# --- 2. 솔루션 클래스 ---
class Solution:
    def __init__(self, routes, data_model):
        self.routes = routes
        self.data = data_model
        self.score = self.calculate_total_score()

    def calculate_total_score(self):
        total_travel_time = 0
        total_service_time = 0
        total_lateness_penalty = 0

        for agv_id, task_sequence in self.routes.items():
            if not task_sequence: continue
                
            agv = self.data.agv_info[agv_id]
            time_cursor = 0.0
            
            # Multi-tour 시뮬레이션
            current_tour_distance = 0
            current_tour_capacity = 0
            last_stop = 'DEPOT'
            
            for task_id in task_sequence:
                task = self.data.task_info[task_id]
                dist_to_task = self.data.get_manhattan_distance(last_stop, task_id)
                dist_from_task_to_depot = self.data.get_manhattan_distance(task_id, 'DEPOT')

                if (current_tour_distance + dist_to_task + dist_from_task_to_depot > agv['max_distance'] or
                    current_tour_capacity + task['demand'] > agv['capacity']):
                    
                    dist_to_depot = self.data.get_manhattan_distance(last_stop, 'DEPOT')
                    travel_time_to_depot = dist_to_depot / agv['speed_cells_per_sec']
                    time_cursor += travel_time_to_depot
                    total_travel_time += travel_time_to_depot

                    last_stop = 'DEPOT'
                    current_tour_distance = 0
                    current_tour_capacity = 0
                    dist_to_task = self.data.get_manhattan_distance(last_stop, task_id)
                
                travel_time = dist_to_task / agv['speed_cells_per_sec']
                time_cursor += travel_time
                total_travel_time += travel_time
                
                service_time = task['service_time']
                completion_time = time_cursor + service_time
                lateness = max(0, completion_time - task['deadline'])
                
                total_service_time += service_time
                total_lateness_penalty += lateness
                
                time_cursor = completion_time
                current_tour_distance += dist_to_task
                current_tour_capacity += task['demand']
                last_stop = task_id

            dist_to_depot = self.data.get_manhattan_distance(last_stop, 'DEPOT')
            total_travel_time += dist_to_depot / agv['speed_cells_per_sec']

        return total_travel_time + total_service_time + total_lateness_penalty

# --- 3. ALNS 솔버 클래스 ---
class AlnsSolver:
    def __init__(self, data_model, initial_solution_routes):
        self.data = data_model
        all_tasks = set(self.data.task_info.keys()) - {'DEPOT'}
        assigned_tasks = set(t for r in initial_solution_routes.values() for t in r)
        self.unassigned_tasks = list(all_tasks - assigned_tasks)
        random.shuffle(self.unassigned_tasks)
        self.current_solution = Solution(initial_solution_routes, self.data)
        print("미할당된 Task들을 경로에 삽입합니다...")
        self.greedy_insertion(self.unassigned_tasks)
        self.current_solution.score = self.current_solution.calculate_total_score()
        self.best_solution = copy.deepcopy(self.current_solution)

    def run(self, iterations):
        print(f"ALNS 탐색을 시작합니다. 초기 점수: {self.best_solution.score:.2f}")
        for i in range(iterations):
            temp_solution = copy.deepcopy(self.current_solution)
            removed_tasks = self.random_removal(temp_solution, num_to_remove=3)
            self.greedy_insertion(removed_tasks, solution_to_modify=temp_solution)
            temp_solution.score = temp_solution.calculate_total_score()
            if temp_solution.score < self.best_solution.score:
                self.best_solution = copy.deepcopy(temp_solution)
                self.current_solution = temp_solution
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i+1}/{iterations}: 새 최고 점수 발견! {self.best_solution.score:.2f}")
        print("\n탐색 완료!")
        print(f"최종 점수: {self.best_solution.score:.2f}")

    def random_removal(self, solution, num_to_remove):
        removed_tasks = []
        all_assigned_tasks = [(agv_id, task_id) for agv_id, tasks in solution.routes.items() for task_id in tasks]
        if len(all_assigned_tasks) < num_to_remove: num_to_remove = len(all_assigned_tasks)
        if not all_assigned_tasks: return []
        tasks_to_remove_info = random.sample(all_assigned_tasks, num_to_remove)
        for agv_id, task_id in tasks_to_remove_info:
            solution.routes[agv_id].remove(task_id)
            removed_tasks.append(task_id)
        return removed_tasks

    def greedy_insertion(self, tasks_to_insert, solution_to_modify=None):
        sol = self.current_solution if solution_to_modify is None else solution_to_modify
        for task_id in tasks_to_insert:
            best_agv, best_pos, min_score = None, -1, float('inf')
            for agv_id in sol.routes.keys():
                original_route = sol.routes[agv_id]
                for i in range(len(original_route) + 1):
                    temp_route = original_route[:i] + [task_id] + original_route[i:]
                    temp_sol_routes = copy.deepcopy(sol.routes)
                    temp_sol_routes[agv_id] = temp_route
                    current_score = Solution(temp_sol_routes, self.data).score
                    if current_score < min_score:
                        min_score, best_agv, best_pos = current_score, agv_id, i
            if best_agv is not None:
                sol.routes[best_agv].insert(best_pos, task_id)

# --- 4. 파일 입출력 및 제출 파일 생성 함수 ---
def load_solution_from_file(filename="initial_solution.json"):
    with open(dPATH + filename, 'r') as f:
        routes = json.load(f)
    print(f"'{filename}' 파일에서 초기 해를 불러왔습니다.")
    return routes

def generate_submission_file(solution, data_model, filename):
    print(f"\n최종 제출 파일 '{filename}' 생성을 시작합니다...")
    submission_data = []
    for agv_id, task_sequence in solution.routes.items():
        agv = data_model.agv_info[agv_id]
        final_route_str = "DEPOT"
        if not task_sequence:
            submission_data.append({'agv_id': agv_id, 'route': final_route_str})
            continue
        current_tour, current_distance, current_capacity = [], 0, 0
        last_stop = 'DEPOT'
        for task_id in task_sequence:
            task = data_model.task_info[task_id]
            dist_to_next = data_model.get_manhattan_distance(last_stop, task_id)
            dist_from_next_to_depot = data_model.get_manhattan_distance(task_id, 'DEPOT')
            if (current_distance + dist_to_next + dist_from_next_to_depot > agv['max_distance'] or
                current_capacity + task['demand'] > agv['capacity']):
                if current_tour: final_route_str += "," + ",".join(current_tour)
                final_route_str += ",DEPOT"
                current_tour = [task_id]
                current_distance = data_model.get_manhattan_distance('DEPOT', task_id)
                current_capacity = task['demand']
                last_stop = task_id
            else:
                current_tour.append(task_id)
                current_distance += dist_to_next
                current_capacity += task['demand']
                last_stop = task_id
        if current_tour: final_route_str += "," + ",".join(current_tour)
        final_route_str += ",DEPOT"
        submission_data.append({'agv_id': agv_id, 'route': final_route_str})
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(sPATH + filename, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ '{filename}' 파일 생성이 완료되었습니다!")

if __name__ == '__main__':
    try:
        initial_routes = load_solution_from_file("initial_solution.json")
    except FileNotFoundError:
        print("🚨 'initial_solution.json' 파일을 찾을 수 없습니다.")
    else:
        data = VrpData('agv.csv', 'task.csv')
        for agv_id in data.agv_info:
            if agv_id not in initial_routes: initial_routes[agv_id] = []
        solver = AlnsSolver(data, initial_routes)
        solver.run(iterations=1000000)
        # 50000 13900
        # 100000 13874
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        submission_filename = f"submission_{timestamp}.csv"
        generate_submission_file(solver.best_solution, data, submission_filename)