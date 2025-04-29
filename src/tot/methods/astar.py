import itertools
import numpy as np
from functools import partial
from tot.models import gpt

class PrioritizedItem:
    def __init__(self, priority: float, f_score: float, item: str):
        self.priority = priority
        self.f_score = f_score
        self.item = item
    
    def __lt__(self, other):
        return self.f_score < other.f_score

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    print(value_prompt)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    print(value_outputs)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_heuristic(task, x, y, n_evaluate_sample, cache_value=True):
    """
    计算启发式值，估计从当前状态到目标的成本
    """
    # 使用与value相似的方式，但提示词关注于"距离目标的估计"
    heuristic_prompt = task.astar_prompt_wrap(x, y) 
    if cache_value and hasattr(task, 'heuristic_cache') and heuristic_prompt in task.heuristic_cache:
        return task.heuristic_cache[heuristic_prompt]
    heuristic_outputs = gpt(heuristic_prompt, n=n_evaluate_sample, stop=None)
    print(heuristic_outputs)
    heuristic = task.astar_outputs_unwrap(x, y, heuristic_outputs)
    if cache_value:
        if not hasattr(task, 'heuristic_cache'):
            task.heuristic_cache = {}
        task.heuristic_cache[heuristic_prompt] = heuristic
    print(heuristic)
    return heuristic

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    print(propose_prompt)
    #这里可以查看模型输出结果
    # proposals = gpt(propose_prompt, n=1, stop=None)[0]

    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    print("****************A****************")
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
   
    print(gpt)
    
    # 缓存属性
    if not hasattr(task, 'value_cache'):
        task.value_cache = {}
    if not hasattr(task, 'heuristic_cache'):
        task.heuristic_cache = {}
    
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            print("inupt:",x)
            print("current output candidates:",ys)
            new_ys = [get_proposals(task, x, y) for y in ys]
            # print(new_ys)

        new_ys = list(itertools.chain(*new_ys))
       
        ids = list(range(len(new_ys)))

        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        
        # A* selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'astar':
            # 使用A*的f(n) = g(n) + h(n)进行选择
            f_scores = []
            for i, y in enumerate(new_ys):
                g_score = values[i]  # 使用value作为g(n)
                print(x)
                print(y)
                h_score = get_heuristic(task, x, y, args.n_evaluate_sample)
                # 打印 h(n) 的值
                #print(f"步骤 {step}, 状态索引 {i}: h(n) = {h_score}")
                f_scores.append(g_score + h_score)
            select_ids = sorted(ids, key=lambda x: f_scores[x], reverse=True)[:args.n_select_sample]
        
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}