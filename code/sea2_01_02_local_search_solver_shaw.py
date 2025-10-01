import pandas as pd
import numpy as np
import random
import copy
import json
import csv
import math 
from datetime import datetime

# ⭐️ Set a global random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

dPATH = '/home/jjh/Project/competition/15_sea2/data/'
sPATH = '/home/jjh/Project/competition/15_sea2/sub/'

# --- 1. DATA CLASS (No changes) ---
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

# --- 2. SOLUTION CLASS (No changes) ---
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

# --- 3. ALNS SOLVER CLASS (Upgraded with Shaw Removal) ---
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

        # ⭐️ 1. Add shaw_removal to the operator list ⭐️
        self.destroy_operators = [self.random_removal, self.worst_removal, self.shaw_removal]
        self.operator_weights = {op.__name__: 1 for op in self.destroy_operators}
        self.operator_scores = {op.__name__: 0 for op in self.destroy_operators}
        self.operator_uses = {op.__name__: 0 for op in self.destroy_operators}
        self.reaction_factor = 0.1 
        self.initial_temperature = self.best_solution.score * 0.05 
        self.cooling_rate = 0.9995

    def run(self, iterations):
        # This entire method remains the same, as the new operator fits into the adaptive logic
        print(f"ALNS 탐색을 시작합니다. 초기 점수: {self.best_solution.score:.2f}")
        temperature = self.initial_temperature
        for i in range(iterations):
            temp_solution = copy.deepcopy(self.current_solution)
            
            op_names = list(self.operator_weights.keys())
            weights = list(self.operator_weights.values())
            chosen_op_name = random.choices(op_names, weights=weights, k=1)[0]
            destroy_operator = getattr(self, chosen_op_name)
            
            removed_tasks = destroy_operator(temp_solution, num_to_remove=3)
            self.greedy_insertion(removed_tasks, solution_to_modify=temp_solution)
            
            temp_solution.score = temp_solution.calculate_total_score()

            if temp_solution.score < self.current_solution.score:
                self.current_solution = temp_solution
                if temp_solution.score < self.best_solution.score:
                    self.best_solution = copy.deepcopy(temp_solution)
                    self.operator_scores[chosen_op_name] += 3 
                    print(f"Iteration {i+1}/{iterations}: 🚀 새 최고 점수! {self.best_solution.score:.2f} (by {chosen_op_name})")
                else:
                    self.operator_scores[chosen_op_name] += 1
            elif math.exp((self.current_solution.score - temp_solution.score) / temperature) > random.random():
                self.current_solution = temp_solution 
                self.operator_scores[chosen_op_name] += 0.5 
            
            self.operator_uses[chosen_op_name] += 1
            temperature *= self.cooling_rate

            if (i + 1) % 100 == 0:
                self._update_operator_weights()

        print("\n탐색 완료!")
        print(f"최종 점수: {self.best_solution.score:.2f}")

    def _update_operator_weights(self):
        for op_name in self.operator_weights:
            uses = self.operator_uses[op_name]
            if uses > 0:
                performance = self.operator_scores[op_name] / uses
                current_weight = self.operator_weights[op_name]
                self.operator_weights[op_name] = (1 - self.reaction_factor) * current_weight + self.reaction_factor * performance
        self.operator_scores = {op.__name__: 0 for op in self.destroy_operators}
        self.operator_uses = {op.__name__: 0 for op in self.destroy_operators}

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
        
    def worst_removal(self, solution, num_to_remove):
        removed_tasks = []
        costs = []
        current_score = solution.calculate_total_score()
        all_assigned_tasks = [(agv_id, task_id) for agv_id, tasks in solution.routes.items() for task_id in tasks]
        for agv_id, task_id in all_assigned_tasks:
            temp_routes = copy.deepcopy(solution.routes)
            temp_routes[agv_id].remove(task_id)
            score_without_task = Solution(temp_routes, self.data).score
            cost_saving = current_score - score_without_task
            costs.append((cost_saving, agv_id, task_id))
        costs.sort(key=lambda x: x[0], reverse=True)
        for i in range(min(num_to_remove, len(costs))):
            _, agv_id, task_id = costs[i]
            if task_id in solution.routes[agv_id]:
                solution.routes[agv_id].remove(task_id)
                removed_tasks.append(task_id)
        return removed_tasks
    
    # ⭐️ 2. Implement the Shaw (Relatedness) Removal operator ⭐️
    def shaw_removal(self, solution, num_to_remove):
        """Removes tasks that are related to each other."""
        removed_tasks = []
        all_assigned_tasks = [task_id for tasks in solution.routes.values() for task_id in tasks]
        if not all_assigned_tasks: return []

        # Start with a random task
        first_task = random.choice(all_assigned_tasks)
        
        # Find the AGV for the first task
        first_agv_id = None
        for agv_id, tasks in solution.routes.items():
            if first_task in tasks:
                first_agv_id = agv_id
                break
        
        if first_agv_id:
            solution.routes[first_agv_id].remove(first_task)
            removed_tasks.append(first_task)
            all_assigned_tasks.remove(first_task)

        while len(removed_tasks) < num_to_remove and all_assigned_tasks:
            # Find the task in the remaining pool that is most related to the last removed task
            last_removed = removed_tasks[-1]
            relatedness_scores = []
            
            for task_id in all_assigned_tasks:
                # Relatedness = weight1 * (normalized distance) + weight2 * (normalized deadline difference)
                # Normalizing by dividing by a large number (max possible value)
                dist = self.data.get_manhattan_distance(last_removed, task_id) / 120 
                time_diff = abs(self.data.task_info[last_removed]['deadline'] - self.data.task_info[task_id]['deadline']) / 1000
                relatedness = 0.6 * dist + 0.4 * time_diff # Adjust weights as needed
                relatedness_scores.append((relatedness, task_id))
            
            if not relatedness_scores: break

            # Select the most related task (lowest score)
            relatedness_scores.sort(key=lambda x: x[0])
            most_related_task = relatedness_scores[0][1]

            # Find and remove the most related task
            for agv_id, tasks in solution.routes.items():
                if most_related_task in tasks:
                    solution.routes[agv_id].remove(most_related_task)
                    removed_tasks.append(most_related_task)
                    all_assigned_tasks.remove(most_related_task)
                    break
                    
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

# --- 4. FILE I/O AND SUBMISSION (No changes) ---
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
        solver.run(iterations=600000)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        submission_filename = f"submission_shaw_{timestamp}.csv"
        generate_submission_file(solver.best_solution, data, submission_filename)