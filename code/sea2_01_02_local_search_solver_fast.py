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

# --- 2. SOLUTION CLASS (Simplified) ---
class Solution:
    def __init__(self, routes, data_model, alns_solver_instance):
        self.routes = routes
        self.data = data_model
        # Use the solver's reusable calculation method
        self.route_scores = {agv_id: alns_solver_instance._calculate_single_route_score(agv_id, route) for agv_id, route in routes.items()}
        self.score = sum(self.route_scores.values())

# --- 3. ALNS SOLVER CLASS (Corrected and Optimized) ---
class AlnsSolver:
    def _calculate_single_route_score(self, agv_id, task_sequence):
        """A standalone, reusable method to score a single route."""
        if not task_sequence: return 0
        
        agv = self.data.agv_info[agv_id]
        time_cursor = 0.0
        current_tour_distance, current_tour_capacity = 0, 0
        last_stop = 'DEPOT'
        
        route_travel_time, route_service_time, route_lateness_penalty = 0, 0, 0

        for task_id in task_sequence:
            task = self.data.task_info[task_id]
            dist_to_task = self.data.get_manhattan_distance(last_stop, task_id)
            dist_from_task_to_depot = self.data.get_manhattan_distance(task_id, 'DEPOT')

            if (current_tour_distance + dist_to_task + dist_from_task_to_depot > agv['max_distance'] or
                current_tour_capacity + task['demand'] > agv['capacity']):
                dist_to_depot = self.data.get_manhattan_distance(last_stop, 'DEPOT')
                travel_time_to_depot = dist_to_depot / agv['speed_cells_per_sec']
                time_cursor += travel_time_to_depot
                route_travel_time += travel_time_to_depot
                last_stop = 'DEPOT'
                current_tour_distance, current_tour_capacity = 0, 0
                dist_to_task = self.data.get_manhattan_distance(last_stop, task_id)
            
            travel_time = dist_to_task / agv['speed_cells_per_sec']
            time_cursor += travel_time
            route_travel_time += travel_time
            
            service_time = task['service_time']
            completion_time = time_cursor + service_time
            lateness = max(0, completion_time - task['deadline'])
            
            route_service_time += service_time
            route_lateness_penalty += lateness
            
            time_cursor = completion_time
            current_tour_distance += dist_to_task
            current_tour_capacity += task['demand']
            last_stop = task_id

        dist_to_depot = self.data.get_manhattan_distance(last_stop, 'DEPOT')
        route_travel_time += dist_to_depot / agv['speed_cells_per_sec']

        return route_travel_time + route_service_time + route_lateness_penalty

    def __init__(self, data_model, initial_solution_routes):
        self.data = data_model
        all_tasks = set(self.data.task_info.keys()) - {'DEPOT'}
        assigned_tasks = set(t for r in initial_solution_routes.values() for t in r)
        self.unassigned_tasks = list(all_tasks - assigned_tasks)
        random.shuffle(self.unassigned_tasks)
        
        print("미할당된 Task들을 경로에 삽입합니다...")
        
        # ⭐️⭐️⭐️ (핵심 버그 수정) Operate on a dictionary first, then create the Solution
        initial_routes_copy = copy.deepcopy(initial_solution_routes)
        self.regret_insertion(self.unassigned_tasks, routes_to_modify=initial_routes_copy)
        
        self.current_solution = Solution(initial_routes_copy, self.data, self)
        self.best_solution = copy.deepcopy(self.current_solution)
        
        self.destroy_operators = [self.random_removal, self.worst_removal, self.shaw_removal]
        self.repair_operators = [self.greedy_insertion, self.regret_insertion]
        
        self.operator_weights = {op.__name__: 1 for op in self.destroy_operators + self.repair_operators}
        self.operator_scores = {op.__name__: 0 for op in self.destroy_operators + self.repair_operators}
        self.operator_uses = {op.__name__: 0 for op in self.destroy_operators + self.repair_operators}
        
        self.reaction_factor = 0.1
        self.initial_temperature = self.best_solution.score * 0.03
        self.cooling_rate = 0.999995

    def run(self, iterations):
        print(f"ALNS 탐색을 시작합니다. 초기 점수: {self.best_solution.score:.2f}")
        temperature = self.initial_temperature
        for i in range(iterations):
            temp_routes = copy.deepcopy(self.current_solution.routes)
            
            destroy_op_names = [op.__name__ for op in self.destroy_operators]
            destroy_weights = [self.operator_weights[name] for name in destroy_op_names]
            chosen_destroy_op_name = random.choices(destroy_op_names, weights=destroy_weights, k=1)[0]
            destroy_operator = getattr(self, chosen_destroy_op_name)
            
            removed_tasks = destroy_operator(temp_routes, num_to_remove=8)
            
            repair_op_names = [op.__name__ for op in self.repair_operators]
            repair_weights = [self.operator_weights[name] for name in repair_op_names]
            chosen_repair_op_name = random.choices(repair_op_names, weights=repair_weights, k=1)[0]
            repair_operator = getattr(self, chosen_repair_op_name)
            
            modified_routes_agv_ids = repair_operator(removed_tasks, routes_to_modify=temp_routes)
            
            for agv_id in modified_routes_agv_ids:
                self.two_opt(temp_routes, agv_id)
            
            temp_solution = Solution(temp_routes, self.data, self)
            
            op_combo_name = f"{chosen_destroy_op_name}+{chosen_repair_op_name}"
            
            if temp_solution.score < self.current_solution.score:
                self.current_solution = temp_solution
                if temp_solution.score < self.best_solution.score:
                    self.best_solution = copy.deepcopy(temp_solution)
                    self.operator_scores[chosen_destroy_op_name] += 3
                    self.operator_scores[chosen_repair_op_name] += 3
                    if (i + 1) % 100 == 0:
                      print(f"Iteration {i+1}/{iterations}: 🚀 새 최고 점수! {self.best_solution.score:.2f} (by {op_combo_name})")
                else:
                    self.operator_scores[chosen_destroy_op_name] += 1
                    self.operator_scores[chosen_repair_op_name] += 1
            elif math.exp((self.current_solution.score - temp_solution.score) / temperature) > random.random():
                self.current_solution = temp_solution 
                self.operator_scores[chosen_destroy_op_name] += 0.5
                self.operator_scores[chosen_repair_op_name] += 0.5
            
            self.operator_uses[chosen_destroy_op_name] += 1
            self.operator_uses[chosen_repair_op_name] += 1
            temperature *= self.cooling_rate

            if (i + 1) % 100 == 0: self._update_operator_weights()

        print("\n탐색 완료!")
        print(f"최종 점수: {self.best_solution.score:.2f}")

    def _update_operator_weights(self):
        for op_name in self.operator_weights:
            uses = self.operator_uses[op_name]
            if uses > 0:
                performance = self.operator_scores[op_name] / uses
                current_weight = self.operator_weights[op_name]
                self.operator_weights[op_name] = (1 - self.reaction_factor) * current_weight + self.reaction_factor * performance
        all_operators = self.destroy_operators + self.repair_operators
        self.operator_scores = {op.__name__: 0 for op in all_operators}
        self.operator_uses = {op.__name__: 0 for op in all_operators}

    def random_removal(self, routes, num_to_remove):
        removed_tasks = []
        all_assigned_tasks = [(agv_id, task_id) for agv_id, tasks in routes.items() for task_id in tasks]
        if len(all_assigned_tasks) < num_to_remove: num_to_remove = len(all_assigned_tasks)
        if not all_assigned_tasks: return []
        tasks_to_remove_info = random.sample(all_assigned_tasks, num_to_remove)
        for agv_id, task_id in tasks_to_remove_info:
            routes[agv_id].remove(task_id)
            removed_tasks.append(task_id)
        return removed_tasks
        
    def worst_removal(self, routes, num_to_remove):
        removed_tasks = []
        costs = []
        
        current_route_scores = {agv_id: self._calculate_single_route_score(agv_id, r) for agv_id, r in routes.items()}

        for agv_id, task_list in routes.items():
            for task_id in task_list:
                original_score = current_route_scores[agv_id]
                temp_route = [t for t in task_list if t != task_id]
                new_score = self._calculate_single_route_score(agv_id, temp_route)
                cost_saving = original_score - new_score
                costs.append((cost_saving, agv_id, task_id))
            
        costs.sort(key=lambda x: x[0], reverse=True)
        for i in range(min(num_to_remove, len(costs))):
            _, agv_id, task_id = costs[i]
            if task_id in routes[agv_id]:
                routes[agv_id].remove(task_id)
                removed_tasks.append(task_id)
        return removed_tasks
    
    def shaw_removal(self, routes, num_to_remove):
        removed_tasks = []
        all_assigned_tasks = [task_id for tasks in routes.values() for task_id in tasks]
        if not all_assigned_tasks: return []
        first_task = random.choice(all_assigned_tasks)
        first_agv_id = None
        for agv_id, tasks in routes.items():
            if first_task in tasks:
                first_agv_id = agv_id
                break
        if first_agv_id:
            routes[first_agv_id].remove(first_task)
            removed_tasks.append(first_task)
            all_assigned_tasks.remove(first_task)
        while len(removed_tasks) < num_to_remove and all_assigned_tasks:
            last_removed = removed_tasks[-1]
            relatedness_scores = []
            for task_id in all_assigned_tasks:
                dist = self.data.get_manhattan_distance(last_removed, task_id) / 120 
                time_diff = abs(self.data.task_info[last_removed]['deadline'] - self.data.task_info[task_id]['deadline']) / 1000
                relatedness = 0.6 * dist + 0.4 * time_diff
                relatedness_scores.append((relatedness, task_id))
            if not relatedness_scores: break
            relatedness_scores.sort(key=lambda x: x[0])
            most_related_task = relatedness_scores[0][1]
            for agv_id, tasks in routes.items():
                if most_related_task in tasks:
                    routes[agv_id].remove(most_related_task)
                    removed_tasks.append(most_related_task)
                    all_assigned_tasks.remove(most_related_task)
                    break
        return removed_tasks

    def greedy_insertion(self, tasks_to_insert, routes_to_modify):
        modified_routes = set()
        for task_id in tasks_to_insert:
            best_agv, best_pos, min_cost_increase = None, -1, float('inf')
            
            for agv_id, original_route in routes_to_modify.items():
                original_route_score = self._calculate_single_route_score(agv_id, original_route)
                for i in range(len(original_route) + 1):
                    temp_route = original_route[:i] + [task_id] + original_route[i:]
                    new_route_score = self._calculate_single_route_score(agv_id, temp_route)
                    cost_increase = new_route_score - original_route_score

                    if cost_increase < min_cost_increase:
                        min_cost_increase, best_agv, best_pos = cost_increase, agv_id, i
            
            if best_agv is not None:
                routes_to_modify[best_agv].insert(best_pos, task_id)
                modified_routes.add(best_agv)
        return modified_routes

    def regret_insertion(self, tasks_to_insert, routes_to_modify):
        modified_routes = set()
        
        while tasks_to_insert:
            max_regret = -float('inf')
            best_insertion = None
            task_to_remove_from_pool = None

            for task_id in tasks_to_insert:
                insertion_costs = []
                for agv_id, original_route in routes_to_modify.items():
                    original_route_score = self._calculate_single_route_score(agv_id, original_route)
                    for i in range(len(original_route) + 1):
                        temp_route = original_route[:i] + [task_id] + original_route[i:]
                        new_score = self._calculate_single_route_score(agv_id, temp_route)
                        cost_increase = new_score - original_route_score
                        insertion_costs.append({'cost': cost_increase, 'agv': agv_id, 'pos': i, 'abs_score': new_score})
                
                insertion_costs.sort(key=lambda x: x['cost'])
                
                regret = (insertion_costs[1]['cost'] - insertion_costs[0]['cost']) if len(insertion_costs) > 1 else insertion_costs[0]['cost']

                if regret > max_regret:
                    max_regret = regret
                    best_insertion = insertion_costs[0]
                    task_to_remove_from_pool = task_id

            if best_insertion is not None:
                best_agv = best_insertion['agv']
                best_pos = best_insertion['pos']
                routes_to_modify[best_agv].insert(best_pos, task_to_remove_from_pool)
                tasks_to_insert.remove(task_to_remove_from_pool)
                modified_routes.add(best_agv)
            else: break
        return modified_routes

    def two_opt(self, routes, agv_id):
        route = routes[agv_id]
        if len(route) < 2: return
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    original_score = self._calculate_single_route_score(agv_id, route)
                    new_score = self._calculate_single_route_score(agv_id, new_route)
                    
                    if new_score < original_score:
                        routes[agv_id] = new_route
                        route = new_route
                        improved = True
                        break
                if improved: break

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
        solver.run(iterations=1000000) 
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        submission_filename = f"submission_fast_{timestamp}.csv"
        generate_submission_file(solver.best_solution, data, submission_filename)