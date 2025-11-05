# Smart-Port-AGV-Route-Optimization-Competition

## 프로젝트 개요

스마트 해운물류 x AI 미션 챌린지: 스마트 항만 AGV 경로 최적화 경진대회

항만에서 AGV(자동화 가이드 차량)의 경로를 최적화하는 문제입니다.

## 주요 기능

- OR-Tools를 활용한 초기 솔루션 생성
- Local Search 알고리즘 (2-Opt, Shaw Removal, Regret Insertion 등)
- ALNS (Adaptive Large Neighborhood Search)
- 체리피킹 및 재구성 알고리즘
- 여러 솔버 모듈 통합

## 파일 구조

```
Smart-Port-AGV-Route-Optimization-Competition/
├── code/
│   ├── sea2_00.py
│   ├── sea2_01_01_generate_solution_ortools.py
│   ├── sea2_01_02_local_search_solver_*.py
│   ├── sea2_01_03_reconstruct_from_elites.py
│   └── ...
└── README.md
```

## 주요 알고리즘

- OR-Tools (초기 해 생성)
- 2-Opt (경로 개선)
- Shaw Removal (제거 휴리스틱)
- Regret Insertion (삽입 휴리스틱)
- ALNS (적응형 대규모 이웃 탐색)

## 기술 스택

- Python
- OR-Tools
- numpy, pandas