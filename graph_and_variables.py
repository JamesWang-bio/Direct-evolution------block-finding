"""
====================================================
本程序用于一次性处理多个序列（列表集），计算整个集合的梯子路径指数，
并支持绘制包含所有序列的梯子图。核心参数集中配置，支持 DNA 和蛋白质序列。
"""

# ==================== 1. 模块导入 ====================
import ladderpath as lp
import os
import matplotlib.pyplot as plt  # 新增：用于绘图

# ==================== 2. 核心参数配置区 ====================
# η估计参数（用于 lp.get_ladderpath）
ESTIMATE_ETA_PARA = {
    'n': 20,                           # 随机化次数
    'max_method': 'AllIdentical',       # ω_max 计算方法
    'min_method': 'EvenDist',           # ω_min 计算方法
    'min_method_nBase': 20              # 基本符号种类数（蛋白质20，DNA 4）
}

# 绘图默认参数（用于 lp.draw_laddergraph）
DRAW_GRAPH_DEFAULTS = {
    'show_longer_than': 8,              # 只显示长度大于特定数的 ladderon
    'style': 'ellipse',                  # 节点样式：ellipse / box / ...
    'figformat': 'png',                  # 输出图片格式
    'rankdir': 'BT',                      # 布局方向：BT（下至上）
    'color': 'grey',                      # 节点和边颜色
    'figsize': None,                     # 图片尺寸
    'cleanGVfile': True                   # 渲染后删除中间 .gv 文件
}

# 支持的序列类型及其允许字符集
DNA_LETTERS = set('ACTG')                     # DNA/RNA 碱基
PROTEIN_LETTERS = set('ARNDCQEGHILKMFPSTWYV')  # 20种标准氨基酸（单字母代码，大写）

ALLOWED_CHARS = {
    'dna': DNA_LETTERS,
    'protein': PROTEIN_LETTERS
}


# ==================== 3. 序列加载函数 ====================
def load_fna_sequence(filepath, seq_type='dna'):
    """
    读取 .fna 格式文件，根据指定的序列类型提取并返回纯字符序列字符串。

    Args:
        filepath (str): FASTA 文件路径。
        seq_type (str): 序列类型，可选 'dna' 或 'protein'。默认为 'dna'。

    Returns:
        str: 纯序列字符串（大写），仅包含允许的字符。

    Raises:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果指定的 seq_type 不支持，或提取到的序列为空。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在：{filepath}")

    if seq_type not in ALLOWED_CHARS:
        raise ValueError(f"不支持的序列类型：{seq_type}。可选类型：{list(ALLOWED_CHARS.keys())}")

    allowed = ALLOWED_CHARS[seq_type]

    with open(filepath, 'r') as f:
        seq_chars = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                seq_chars.extend(c for c in line.upper() if c in allowed)
        seq = ''.join(seq_chars)
        if not seq:
            raise ValueError(f"文件 {filepath} 中未提取到任何有效字符（类型：{seq_type}），请检查文件内容或过滤规则。")
        return seq


def build_sequence_dict(file_dict, seq_type='dna'):
    """
    根据文件路径字典构建序列数据字典（每条序列以列表形式存储）。

    Args:
        file_dict (dict): 键为序列名称，值为 .fna 文件路径。
        seq_type (str): 序列类型。

    Returns:
        dict: 形如 {name: {'sequence': [seq_str]}, ...} 的字典。
              此字典可用于需要分别处理序列的场景（本程序未使用，但保留以保持接口一致性）。
    """
    data = {}
    for name, path in file_dict.items():
        seq_str = load_fna_sequence(path, seq_type=seq_type)
        data[name] = {'sequence': [seq_str]}
    return data


def build_sequence_list(file_dict, seq_type='dna'):
    """
    根据文件路径字典构建一个包含所有序列的字符串列表，用于一次性统一处理。

    Args:
        file_dict (dict): 键为序列名称，值为 .fna 文件路径。
        seq_type (str): 序列类型。

    Returns:
        list: 每个元素是一条序列的字符串，顺序与 file_dict 的迭代顺序一致。
    """
    seq_list = []
    for name, path in file_dict.items():
        seq_str = load_fna_sequence(path, seq_type=seq_type)
        seq_list.append(seq_str)
    return seq_list


# ==================== 4. 核心计算函数（无绘图） ====================
def compute_ladderpath(sequence):
    """
    计算单个或多个序列的梯子路径指标。输入可以是字符串（单条序列）或字符串列表（多条序列）。

    Args:
        sequence (str 或 list): 输入序列。

    Returns:
        tuple: (size_index, ladderpath_index, order_index, eta, full_lpjson)
               各返回值的含义如下：
               - size_index: 规模指数（序列长度或总操作次数）
               - ladderpath_index: 梯子路径指数（生成序列所需的最小操作数）
               - order_index: 有序指数（基于层次结构的度量）
               - eta: 有序率（Order rate，介于0~1之间）
               - full_lpjson: 完整的梯子路径结果字典（包含ladderons、基本模块、targets映射等详细信息）

    Raises:
        TypeError: 输入类型错误。
        ValueError: 如果 lp.get_ladderpath 返回 None。
    """
    if isinstance(sequence, str):
        sequence = [sequence]   # 单条序列转为列表
    elif not isinstance(sequence, list):
        raise TypeError("sequence 必须是字符串或字符串列表")

    lpjson = lp.get_ladderpath(
        sequence,
        estimate_eta=True,
        estimate_eta_para=ESTIMATE_ETA_PARA
    )

    if lpjson is None:
        raise ValueError(
            "ladderpath.get_ladderpath 返回了 None。可能原因：\n"
            "1. 输入序列为空或全为非法字符；\n"
            "2. 输入格式错误（应传入字符串列表）；\n"
            "3. ladderpath 库内部错误。\n"
            f"当前输入类型: {type(sequence)}, 内容预览: {str(sequence)[:100]}"
        )

    size_idx = lpjson['size-index']
    ladder_idx = lpjson['ladderpath-index']
    order_idx = lpjson['order-index']
    eta = lpjson['eta']

    return size_idx, ladder_idx, order_idx, eta, lpjson


# ==================== 5. 独立绘图函数 ====================
def draw_ladderpath(lpjson, fig_path=None, **kwargs):
    """
    根据已有的 lpjson 对象绘制梯子图。
    绘图参数使用全局默认值 DRAW_GRAPH_DEFAULTS，可通过 kwargs 覆盖。

    Args:
        lpjson (dict): compute_ladderpath 返回的完整 JSON 对象。
        fig_path (str, optional): 图形保存路径（含文件名，不含扩展名）。
                                   若为 None，则直接显示图形（不保存）。
        **kwargs: 可覆盖的绘图参数，支持：
                  show_longer_than, style, figformat, rankdir, color, figsize, cleanGVfile
                  具体含义见 DRAW_GRAPH_DEFAULTS 注释。
    """
    draw_params = DRAW_GRAPH_DEFAULTS.copy()
    draw_params.update(kwargs)

    if fig_path:
        save_name = f"{fig_path}.{draw_params['figformat']}"
    else:
        save_name = None

    lp.draw_laddergraph(
        lpjson,
        save_fig_name=save_name,
        show_longer_than=draw_params['show_longer_than'],
        style=draw_params['style'],
        figformat=draw_params['figformat'],
        rankdir=draw_params['rankdir'],
        color=draw_params['color'],
        figsize=draw_params['figsize'],
        cleanGVfile=draw_params['cleanGVfile']
    )

# ==================== 新增函数：绘制公共 ladderon 数量 vs 最小长度阈值 与 绘制 ladderon 数量 vs 最小长度阈值 ====================
def get_target_used_ladderons(lpjson):
    """
    从 lpjson 中提取每个目标序列所使用的所有 ladderon ID（包括间接使用的）。

    Args:
        lpjson (dict): 完整的梯子路径结果。

    Returns:
        dict: {target_id: set_of_ladderon_ids}
    """
    ladderons = lpjson.get('ladderons', {})
    targets = lpjson.get('targets', {})

    # 构建 ladderon 的组成关系（用于递归展开）
    comp_map = {}
    for lid, info in ladderons.items():
        # info 格式可能为 [components, len, ...] 或 {'comp': ..., 'len': ...}
        if isinstance(info, list) and len(info) >= 1:
            comp = info[0]
        elif isinstance(info, dict) and 'comp' in info:
            comp = info['comp']
        else:
            comp = []
        # 将组件中的数字ID提取出来（忽略字符串）
        ids = [x for x in comp if isinstance(x, int)]
        comp_map[lid] = ids

    # 递归获取一个组件（ladderon或target）使用的所有 ladderon ID
    def get_all_used(item_id, memo):
        if item_id in memo:
            return memo[item_id]
        if item_id >= 0:  # ladderon
            used = set(comp_map.get(item_id, []))
            for sub_id in list(used):
                used |= get_all_used(sub_id, memo)
        else:  # target
            target_info = targets.get(item_id, [])
            comp = target_info[0] if isinstance(target_info, list) and len(target_info) >= 1 else []
            used = set(x for x in comp if isinstance(x, int))
            for sub_id in list(used):
                used |= get_all_used(sub_id, memo)
        memo[item_id] = used
        return used

    target_used = {}
    memo = {}
    for tid in targets.keys():
        target_used[tid] = get_all_used(tid, memo)

    return target_used

def plot_ladderon_analysis(lpjson, mode='all', use_rate=True, save_path=None, show=True):
    """
    绘制 ladderon 数量（或公共 ladderon 数量）随最小长度阈值变化的折线图（高级整合函数）。

    Args:
        lpjson (dict): compute_ladderpath 返回的完整 JSON 对象。
        mode (str): 绘图模式，可选 'all'（所有 ladderon）或 'shared'（公共 ladderon）。默认为 'all'。
        use_rate (bool): 如果 True，纵轴为变化率（相邻阈值间数量的减少量）；
                         如果 False，纵轴为数量。默认为 True。
        save_path (str, optional): 保存图片的路径（不含扩展名），自动添加 .png。
        show (bool): 是否显示图像。
    """
    import matplotlib.pyplot as plt

    # 根据模式获取 ladderon 长度列表
    if mode == 'all':
        # 从所有 ladderons 中提取长度
        ladderons = lpjson.get('ladderons', {})
        lengths = []
        for lid, info in ladderons.items():
            if isinstance(info, list) and len(info) >= 2 and isinstance(info[1], int):
                lengths.append(info[1])
            elif isinstance(info, dict) and 'len' in info:
                lengths.append(info['len'])
        title_prefix = "Ladderon"
    elif mode == 'shared':
        # 获取公共 ladderon 的 ID 和长度
        target_used = get_target_used_ladderons(lpjson)
        if not target_used:
            print("警告：未找到目标序列信息。")
            return
        # 取交集
        common_ladderons = None
        for used in target_used.values():
            if common_ladderons is None:
                common_ladderons = set(used)
            else:
                common_ladderons &= set(used)
        if not common_ladderons:
            print("警告：没有公共 ladderon。")
            return
        # 获取长度
        ladderons = lpjson.get('ladderons', {})
        lengths = []
        for lid in common_ladderons:
            info = ladderons.get(lid)
            if info:
                if isinstance(info, list) and len(info) >= 2 and isinstance(info[1], int):
                    lengths.append(info[1])
                elif isinstance(info, dict) and 'len' in info:
                    lengths.append(info['len'])
        title_prefix = "Shared ladderon"
    else:
        raise ValueError("mode 必须是 'all' 或 'shared'")

    if not lengths:
        print(f"警告：未找到 {title_prefix.lower()} 的长度信息。")
        return

    max_len = max(lengths)
    min_len = min(lengths)

    # 统计每个长度出现次数
    len_counts = {}
    for l in lengths:
        len_counts[l] = len_counts.get(l, 0) + 1

    # 累计数量（长度 ≥ 阈值）
    thresholds = list(range(min_len, max_len + 1))
    counts = [sum(cnt for l, cnt in len_counts.items() if l >= t) for t in thresholds]

    if use_rate:
        y_values = [counts[i-1] - counts[i] for i in range(1, len(thresholds))]
        x_values = thresholds[1:]
        ylabel = "Number reduction (Δcount)"
        title = f"{title_prefix} count reduction vs. minimum length threshold"
    else:
        y_values = counts
        x_values = thresholds
        ylabel = "Number"
        title = f"{title_prefix} count vs. minimum length threshold"

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Minimum length threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        print(f"图像已保存至：{save_path}.png")
    if show:
        plt.show()
    else:
        plt.close()

# ==================== 6. 高层封装函数（支持绘图） ====================
def find_ladderpath(sequence, draw_graph=False, fig_path=None, **kwargs):
    """
    计算梯子路径指标，并可选择绘制图形。内部调用 compute_ladderpath 和 draw_ladderpath。

    Args:
        sequence (str 或 list): 输入序列（字符串）或序列列表。
        draw_graph (bool): 是否绘制梯子图。
        fig_path (str, optional): 图形保存路径（仅当 draw_graph=True 时有效）。
        **kwargs: 传递给 draw_ladderpath 的其他绘图参数。

    Returns:
        tuple: 同 compute_ladderpath 的返回值。
    """
    result = compute_ladderpath(sequence)
    if draw_graph:
        draw_ladderpath(result[4], fig_path=fig_path, **kwargs)
    return result


# ==================== 7. 多序列统一处理专用函数（推荐使用） ====================
def process_multiple_sequences(seq_list, draw_graph=False, fig_path=None, **kwargs):
    """
    一次性处理多个序列（列表集），即所有序列一起分析，找出共同 ladderons，
    计算整个集合的梯子路径指数，并可选择绘制包含所有序列的梯子图。

    Args:
        seq_list (list of str): 包含多个序列字符串的列表。
        draw_graph (bool): 是否绘制梯子图。
        fig_path (str, optional): 图形保存路径。
        **kwargs: 传递给 draw_ladderpath 的绘图参数。

    Returns:
        tuple: 同 compute_ladderpath 的返回值（size, ladder, order, eta, full_lpjson）。
    """
    return find_ladderpath(seq_list, draw_graph=draw_graph, fig_path=fig_path, **kwargs)


# ==================== 8. 主程序示例 ====================
if __name__ == '__main__':
    # 8.1 定义多个基因文件路径（请根据实际情况修改）
    gene_files = {
        'MYB42': r"C:\Users\wangx\Desktop\Ladder path 定向进化\MYB基因两兄弟\MYB42_datasets\ncbi_dataset\data\gene.fna",
        'MYB85': r"C:\Users\wangx\Desktop\Ladder path 定向进化\MYB基因两兄弟\MYB85_datasets\ncbi_dataset\data\gene.fna",  # 可添加更多
    }

    seq_type = 'dna'  # 可选 'dna' 或 'protein'

    # 8.2 构建序列列表集（用于一次性统一处理）
    seq_list = build_sequence_list(gene_files, seq_type=seq_type)
    print(f"已构建序列列表，共 {len(seq_list)} 条序列。")

    # 8.3 一次性处理所有序列（统一分析）
    print("\n--- 开始一次性处理所有序列 ---")
    size_all, ladder_all, order_all, eta_all, json_all = process_multiple_sequences(
        seq_list,
        draw_graph=False,          # 如需绘图，设为 True 并指定 fig_path（需安装 Graphviz）
        fig_path="MYB_family"
    )

    print(f"统一处理结果：")
    print(f"  size-index      = {size_all}")
    print(f"  ladderpath-index= {ladder_all}")
    print(f"  order-index     = {order_all}")
    print(f"  eta (order rate) = {eta_all:.4f}")

    # ========== 新增：绘制 ladderon 数量或共享的 ladderon数量 随最小长度阈值变化的图像 ==========
    plot_ladderon_analysis(json_all, mode='shared', use_rate=True, save_path="shared_ladderon_rate", show=True)