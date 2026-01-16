#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

try:
    from pymatgen.io.vasp import Poscar
except Exception as e:
    raise RuntimeError("需要 pymatgen：pip install pymatgen") from e

# shapely 仅用于多边形相交与面积；如未安装则自动安装
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.errors import TopologicalError
except Exception:
    print("警告: 未安装 shapely 库，正在尝试自动安装 ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.errors import TopologicalError


# ====== 全局参数 ======
# 扩展判断区域（分数坐标范围）；用于“重心在扩展主晶胞”过滤
EXT_X_RANGE = (-0.2, 1.2)
EXT_Y_RANGE = (-0.2, 1.2)
EXT_Z_RANGE = (0.0, 1.0)

# 可选：仅对某个环尺寸打印去重原因（例如 7）；不需要调试时置为 None
DEBUG_RING_FILTER_REASON = None  # 例如设为 7 可只看 7 环被过滤原因


# ====== 几何与图构建 ======
def find_minimum_distance_with_pbc(coord1, coord2, lattice_matrix):
    """计算两个原子在周期性边界条件下的最短距离和对应的坐标（返回 best_coord2）"""
    min_dist = float('inf')
    best_coord2 = coord2.copy()
    # 27 个镜像（含原胞）
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                offset = dx * lattice_matrix[0] + dy * lattice_matrix[1] + dz * lattice_matrix[2]
                shifted = coord2 + offset
                dist = np.linalg.norm(coord1 - shifted)
                if dist < min_dist:
                    min_dist = dist
                    best_coord2 = shifted
    return min_dist, best_coord2


def build_fast_periodic_graph(poscar_file="POSCAR", cutoff=1.6):
    """读取 VASP 结构、构建键图（基于 PBC 最短距离，阈值 cutoff）"""
    print("读取POSCAR文件...")
    poscar = Poscar.from_file(poscar_file)
    structure = poscar.structure
    n_atoms = len(structure)

    lattice_matrix = structure.lattice.matrix  # 3x3
    print(f"原子数量: {n_atoms}")
    print("晶格参数:")
    for i, vec in enumerate(lattice_matrix):
        print(f"  {['a','b','c'][i]}: [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")

    original_coords = np.array([site.coords for site in structure])

    G = nx.Graph()
    for i in range(n_atoms):
        G.add_node(i)

    edge_coords = {}  # (i,j) -> (coord_i, best_coord_j)
    print("计算原子间距离并建立连接...")
    edges_added = 0
    total_pairs = n_atoms * (n_atoms - 1) // 2

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist, best_j = find_minimum_distance_with_pbc(original_coords[i], original_coords[j], lattice_matrix)
            if dist < cutoff:
                G.add_edge(i, j)
                edge_coords[(i, j)] = (original_coords[i], best_j)
                edges_added += 1
        if (i + 1) % 50 == 0 or i == n_atoms - 1:
            progress = ((i + 1) * i // 2) / total_pairs * 100 if total_pairs else 100.0
            print(f"  进度: {progress:.1f}% ({i+1}/{n_atoms} 原子)")

    print(f"建立了 {edges_added} 个连接")
    if n_atoms > 0:
        print(f"平均每个原子的配位数: {2 * edges_added / n_atoms:.2f}")

    return G, original_coords, edge_coords, structure, lattice_matrix


def adjust_ring_coordinates(cycle, original_coords, lattice_matrix):
    """将跨边界环的点坐标通过最近镜像展开到连续空间，便于绘制与几何计算"""
    if len(cycle) < 3:
        return original_coords[cycle]
    adjusted = [original_coords[cycle[0]]]
    for k in range(1, len(cycle)):
        prev = adjusted[k - 1]
        cur_orig = original_coords[cycle[k]]
        _, best = find_minimum_distance_with_pbc(prev, cur_orig, lattice_matrix)
        adjusted.append(best)
    return np.array(adjusted)


def find_best_shift(center, lattice_matrix, cell_center):
    """在 27 个镜像中选使重心靠近主晶胞中心的偏移量"""
    min_dist = float('inf')
    best_offset = np.zeros(3)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                offset = dx * lattice_matrix[0] + dy * lattice_matrix[1] + dz * lattice_matrix[2]
                shifted_center = center + offset
                dist = np.linalg.norm(shifted_center - cell_center)
                if dist < min_dist:
                    min_dist = dist
                    best_offset = offset
    return best_offset


# ====== 环面均值 (toroidal mean) ======
def _torus_mean_1d(fracs):
    """对一组分数坐标 (0..1) 做环面均值，返回 [0,1) 上的中心值"""
    if len(fracs) == 0:
        return 0.5
    angles = 2*np.pi * (np.asarray(fracs) % 1.0)
    c = np.cos(angles).mean()
    s = np.sin(angles).mean()
    if c == 0 and s == 0:
        return (np.asarray(fracs).mean() % 1.0)  # 极端退化：回退
    ang = np.arctan2(s, c)  # (-pi, pi]
    if ang < 0:
        ang += 2*np.pi
    return (ang / (2*np.pi)) % 1.0


def _ring_center_frac_toroidal(cycle, original_coords, lattice_matrix):
    """返回环在分数坐标系的‘环面中心’(x,y,z)，对 0/1 接缝稳定。"""
    lattice_inv = np.linalg.inv(lattice_matrix)
    fracs = []
    for idx in cycle:
        fc = np.dot(original_coords[idx], lattice_inv)  # 不强行裁到 [0,1)
        fracs.append(fc)
    fracs = np.asarray(fracs)
    fx = _torus_mean_1d(fracs[:, 0])
    fy = _torus_mean_1d(fracs[:, 1])
    fz = _torus_mean_1d(fracs[:, 2])
    return np.array([fx, fy, fz])


def is_ring_center_in_primary_cell(cycle, original_coords, lattice_matrix,
                                   margin=0.02, use_extended_box=True, DEBUG=False):
    """
    用环面均值在分数坐标下判断环重心是否在主胞（或扩展主胞）内。
    - margin: 判定边界松弛 (默认 0.02)
    - use_extended_box: True 使用全局 EXT_*_RANGE；False 使用 [0,1] 外扩一点
    """
    try:
        fcenter = _ring_center_frac_toroidal(cycle, original_coords, lattice_matrix)

        if use_extended_box:
            xr, yr, zr = EXT_X_RANGE, EXT_Y_RANGE, EXT_Z_RANGE
        else:
            xr = (-margin, 1.0 + margin)
            yr = (-margin, 1.0 + margin)
            zr = (-margin, 1.0 + margin)

        inside = (xr[0] <= fcenter[0] <= xr[1] and
                  yr[0] <= fcenter[1] <= yr[1] and
                  zr[0] <= fcenter[2] <= zr[1])

        if DEBUG and not inside:
            print(f"[DEBUG] 排除环 (size={len(cycle)}): torus center frac={fcenter}, "
                  f"box=({xr},{yr},{zr})")
        return inside
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] is_ring_center_in_primary_cell 异常，保留该环: {e}")
        return True


def get_normalized_polygon(cycle, original_coords, lattice_matrix, cell_center):
    """将环几何标准化到主晶胞附近（仅取 xy 形成多边形）"""
    try:
        ring_coords = adjust_ring_coordinates(cycle, original_coords, lattice_matrix)
        center = np.mean(ring_coords, axis=0)
        offset = find_best_shift(center, lattice_matrix, cell_center)
        ring_coords = ring_coords + offset
        poly = Polygon(ring_coords[:, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)  # 修复自交等
        return poly
    except (ValueError, TopologicalError):
        return None


# ====== 环枚举（DFS + 最小环基补充）======
def get_ring_canonical_form(cycle):
    """将环转换为规范形式（旋转/翻转等价去重）"""
    if not cycle:
        return tuple()
    mn = min(cycle)
    idxs = [i for i, v in enumerate(cycle) if v == mn]
    forms = []
    for idx in idxs:
        fwd = cycle[idx:] + cycle[:idx]
        rev = (cycle[:idx + 1][::-1] + cycle[idx + 1:][::-1])
        forms.append(tuple(fwd))
        forms.append(tuple(rev))
    return min(forms)


def find_all_cycles_dfs(G, start_node, max_length):
    """从指定起点做 DFS，收集到起点的所有简单环（长度<=max_length）"""
    all_cycles = []

    def dfs(path, used_edges):
        if len(path) > max_length:
            return
        cur = path[-1]
        # 回到起点形成环（长度>=3）
        if len(path) >= 3 and start_node in G.neighbors(cur):
            ekey = tuple(sorted((cur, start_node)))
            if ekey not in used_edges:
                all_cycles.append(path[:])
        if len(path) < max_length:
            for nb in G.neighbors(cur):
                ekey = tuple(sorted((cur, nb)))
                # 不允许立即回边（在 path 长度很小时）
                if len(path) >= 2 and nb == path[-2] and len(path) < 3:
                    continue
                if ekey in used_edges:
                    continue
                if nb in path[1:] and nb != start_node:
                    continue
                new_used = used_edges.copy()
                new_used.add(ekey)
                dfs(path + [nb], new_used)

    dfs([start_node], set())
    return all_cycles


def find_rings_comprehensive(G, max_ring_size=10):
    """DFS 全搜 + networkx 最小环基补充，得到不重复的候选环集合"""
    print("开始全面环检测（包括非凸环）...")
    t0 = time.time()

    unique_cycles = []
    seen = set()

    nodes_by_deg = sorted(G.nodes(), key=lambda x: G.degree(x))
    total = len(nodes_by_deg)

    for idx, s in enumerate(nodes_by_deg):
        if idx % 20 == 0:
            print(f"  环检测进度: {idx/total*100:.1f}% ({idx}/{total} 节点)")
        cycles = find_all_cycles_dfs(G, s, max_ring_size)
        for cyc in cycles:
            if len(cyc) <= max_ring_size:
                sig = get_ring_canonical_form(cyc)
                if sig not in seen:
                    seen.add(sig)
                    unique_cycles.append(cyc)

    print(f"  初步检测到 {len(unique_cycles)} 个唯一环")

    try:
        print("  补充检测基础环（minimum cycle basis）...")
        mcb = nx.minimum_cycle_basis(G)
        added = 0
        for cyc in mcb:
            if len(cyc) <= max_ring_size:
                sig = get_ring_canonical_form(cyc)
                if sig not in seen:
                    seen.add(sig)
                    unique_cycles.append(cyc)
                    added += 1
        print(f"  基础环补充 {added} 个；合计 {len(unique_cycles)}")
    except Exception as e:
        print(f"  补充检测失败: {e}")

    print(f"环检测阶段完成，耗时: {time.time()-t0:.2f}秒")
    return unique_cycles


# ====== 覆盖/复合判定：新增工具 ======
def _vertex_overlap_ratio(cycA, cycB):
    """两环共享顶点比例 = 交集顶点数 / min(lenA, lenB)"""
    setA, setB = set(cycA), set(cycB)
    if not setA or not setB:
        return 0.0
    return len(setA & setB) / float(min(len(setA), len(setB)))


def _polygon_iou(polyA, polyB, eps=1e-9):
    """二维多边形 IoU = 面积交 / 面积并（用于同尺寸环几何相似度）"""
    if polyA is None or polyB is None or polyA.is_empty or polyB.is_empty:
        return 0.0
    try:
        inter = polyA.intersection(polyB)
        if not hasattr(inter, "area"):
            return 0.0
        inter_area = max(0.0, inter.area)
        if inter_area < eps:
            return 0.0
        union_area = max(eps, polyA.area + polyB.area - inter_area)
        return inter_area / union_area
    except Exception:
        return 0.0


def is_ring_crossing_boundary(cycle, original_coords, lattice_matrix, boundary_threshold=0.7):
    """
    判断环是否跨越周期性边界：
    将环点转分数坐标，若 x 或 y 范围超过阈值则认为跨边界
    """
    try:
        lattice_inv = np.linalg.inv(lattice_matrix)
        frac = []
        for idx in cycle:
            cart = original_coords[idx]
            fc = np.dot(cart, lattice_inv)
            frac.append(fc)
        frac = np.array(frac)
        xr = np.max(frac[:, 0]) - np.min(frac[:, 0])
        yr = np.max(frac[:, 1]) - np.min(frac[:, 1])
        return xr > boundary_threshold or yr > boundary_threshold
    except (np.linalg.LinAlgError, ValueError):
        return False


def check_shifted_ring_contains_smaller_rings(
    large_cycle,
    selected_small_rings,
    original_coords,
    lattice_matrix,
    cell_center,
    contain_threshold=0.8
):
    """
    对“大环”做 9 种 (dx,dy ∈ {-1,0,1}) 平移，检查是否几乎包含某个已选的小环；
    仅在大环尺寸 >= 8 时进行。
    """
    L = len(large_cycle)
    if L < 8:
        return False

    smallers = [r for r in selected_small_rings if len(r) < L]
    if not smallers:
        return False

    small_polys = []
    for sr in smallers:
        sp = get_normalized_polygon(sr, original_coords, lattice_matrix, cell_center)
        if sp is not None and (not sp.is_empty):
            small_polys.append(sp)
    if not small_polys:
        return False

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shift_vec = dx * lattice_matrix[0] + dy * lattice_matrix[1]
            try:
                coords = adjust_ring_coordinates(large_cycle, original_coords, lattice_matrix)
                coords_shift = coords + shift_vec
                polyL = Polygon(coords_shift[:, :2])
                if not polyL.is_valid:
                    polyL = polyL.buffer(0)
                if polyL.is_empty or polyL.area <= 1e-10:
                    continue
                for sp in small_polys:
                    try:
                        if polyL.intersects(sp):
                            inter = polyL.intersection(sp)
                            if hasattr(inter, "area") and sp.area > 0:
                                if inter.area / sp.area >= contain_threshold:
                                    return True
                    except (TopologicalError, ValueError):
                        continue
            except (TopologicalError, ValueError):
                continue
    return False


def is_composite_boundary_ring(cycle, selected_rings, original_coords, lattice_matrix, cell_center):
    """先判跨边界，再判平移后是否包含小环；是则视为“跨边界复合环”（需要过滤）"""
    if not is_ring_crossing_boundary(cycle, original_coords, lattice_matrix):
        return False
    return check_shifted_ring_contains_smaller_rings(
        cycle, selected_rings, original_coords, lattice_matrix, cell_center
    )


# ====== 覆盖去重（几何 + 拓扑上下文 + 重心距离保护）======
def is_ring_already_covered_with_context(
    cycle,
    original_coords,
    lattice_matrix,
    selected_pairs,            # [(ring_cycle, poly), ...]
    cell_center,
    general_area_cover=0.5,    # 对不同尺寸/不高度相似的环，用较稳健的 50% 面积覆盖
    same_size_iou=0.85,        # 同尺寸环 IoU 阈值（更严格）
    same_size_vertex=0.50,     # 同尺寸环 顶点重合率阈值（与 IoU 需“且”满足）
    contain_threshold=0.99,    # 小环被几乎完全包含时仍视为重复
    eps=1e-9
):
    """
    综合“几何 + 拓扑 + 重心距离”判断当前环是否应视为已覆盖：
    - 同尺寸：仅当重心接近（≤一条平均键长）且 (IoU >= same_size_iou 且 顶点重合率 >= same_size_vertex) 才视为重复
    - 大环包含小环：intersection >= contain_threshold * area(small) 视为重复（沿用旧规则）
    - 其他情况：只有当覆盖比例 >= general_area_cover 才视为覆盖
    """
    polyC = get_normalized_polygon(cycle, original_coords, lattice_matrix, cell_center)
    if polyC is None or polyC.is_empty or polyC.area <= 1e-10:
        return True  # 无法可靠计算时保守跳过

    sizeC = len(cycle)

    for (ringS, polyS) in selected_pairs:
        if polyS is None or polyS.is_empty or polyS.area <= 1e-10:
            continue

        # 面积交
        try:
            inter = polyC.intersection(polyS)
            if not hasattr(inter, "area"):
                continue
            inter_area = max(0.0, inter.area)
        except (TopologicalError, ValueError):
            continue

        if inter_area < eps:
            continue  # 仅线性接触或无交

        sizeS = len(ringS)

        if sizeC == sizeS:
            # —— 同尺寸分支：先做“重心距离保护”，再做 IoU & 顶点重合的 AND 判定 ——
            iou = _polygon_iou(polyC, polyS, eps=eps)
            vratio = _vertex_overlap_ratio(cycle, ringS)

            # 计算两环在展开+平移到主胞附近后的重心距离（3D），以及一条平均“边长”（在 XY）
            coordsC = adjust_ring_coordinates(cycle, original_coords, lattice_matrix)
            coordsS = adjust_ring_coordinates(ringS, original_coords, lattice_matrix)
            offC = find_best_shift(coordsC.mean(axis=0), lattice_matrix, cell_center)
            offS = find_best_shift(coordsS.mean(axis=0), lattice_matrix, cell_center)
            coordsC, coordsS = coordsC + offC, coordsS + offS

            d_cent = np.linalg.norm(coordsC.mean(axis=0) - coordsS.mean(axis=0))

            def _mean_edge_len_2d(coords):
                n = len(coords)
                if n < 2:
                    return 0.0
                # 用 XY 邻边距离的平均值作为“键长”估计
                return np.mean([np.linalg.norm(coords[(k+1) % n, :2] - coords[k, :2]) for k in range(n)])

            bl = max(_mean_edge_len_2d(coordsC), _mean_edge_len_2d(coordsS))

            if DEBUG_RING_FILTER_REASON == sizeC:
                print(f"[DEBUG] same-size check: IoU={iou:.2f}, vratio={vratio:.2f}, d_cent={d_cent:.3f}, bl={bl:.3f}")

            # 重心距离保护：如果两环的重心距明显大于一条平均键长，则视为不同环（即使 IoU 很高也不去重）
            if bl > 0 and d_cent > bl:
                continue  # 与此已选环不视为重复，继续检查其他已选环

            # 满足几何+拓扑“双高相似”才视为同一环
            if iou >= same_size_iou and vratio >= same_size_vertex:
                return True

        elif sizeC > sizeS:
            # 大环 vs 小环：若几乎把小环包含（原 contain 规则），仍视为重复
            if polyS.area > eps and inter_area / polyS.area >= contain_threshold:
                if DEBUG_RING_FILTER_REASON == sizeC:
                    print(f"[DEBUG] large-vs-small contain: inter/small={inter_area/polyS.area:.2f}")
                return True
        else:
            # 小环 vs 已选大环：需要明显几何覆盖才判重复（避免误杀小环）
            cover_ratio = inter_area / min(polyC.area, polyS.area)
            if DEBUG_RING_FILTER_REASON == sizeC:
                print(f"[DEBUG] small-vs-large cover_ratio={cover_ratio:.2f}")
            if cover_ratio >= general_area_cover:
                return True

    return False


# ====== 去重主流程（增强版）======
def remove_overlapping_rings_enhanced(cycles, original_coords, lattice_matrix):
    """
    增强版环去重：
    1) 重心在扩展主晶胞的筛选（环面均值）
    2) 几何+拓扑+重心距离的覆盖去重（IoU/顶点重合/大环包含小环）
    3) 跨边界复合环检测（排除周期性组合形成的虚大环）
    """
    print("增强版环去重检测（跨边界复合环检测）...")

    if not cycles:
        return []

    cell_center = np.dot([0.5, 0.5, 0.5], lattice_matrix)

    # Step-1: 只保留重心在扩展主晶胞内的环（使用环面均值）
    print("  第一步：过滤跨边界环（使用扩展范围 + 环面重心）...")
    primary = [cyc for cyc in cycles if is_ring_center_in_primary_cell(
        cyc, original_coords, lattice_matrix, margin=0.02, use_extended_box=True, DEBUG=False)]
    print(f"    原始环数: {len(cycles)}")
    print(f"    扩展范围内环数: {len(primary)}")
    print(f"    过滤掉环数: {len(cycles) - len(primary)}")

    # 小环优先
    cyc_with_size = [(c, len(c)) for c in primary]
    cyc_with_size.sort(key=lambda x: x[1])  # by size asc

    selected = []          # 仅存环索引列表（用于复合判断的小环集合）
    selected_pairs = []    # [(ring_cycle, poly)] 用于覆盖判定
    total = len(cyc_with_size)
    overlap_filtered = 0
    boundary_composite_filtered = 0

    print("  第二步：重叠区域去重 + 跨边界复合环过滤...")
    for idx, (cur, sz) in enumerate(cyc_with_size, 1):
        if total > 0 and idx % 50 == 0:
            print(f"    去重进度: {idx/total*100:.1f}% ({idx}/{total})")

        # 覆盖去重 —— 使用“带上下文”的版本
        if is_ring_already_covered_with_context(
            cur, original_coords, lattice_matrix, selected_pairs, cell_center,
            general_area_cover=0.5, same_size_iou=0.85, same_size_vertex=0.50,
            contain_threshold=0.99
        ):
            overlap_filtered += 1
            if DEBUG_RING_FILTER_REASON == sz:
                print(f"[DEBUG] 过滤 {sz} 环：覆盖去重触发（IoU/顶点/重心距离）")
            continue

        # 跨边界复合环过滤（依赖已选小环；仅 >=8）
        if is_composite_boundary_ring(cur, selected, original_coords, lattice_matrix, cell_center):
            boundary_composite_filtered += 1
            print(f"    检测到跨边界复合环: {sz}元环（已过滤）")
            if DEBUG_RING_FILTER_REASON == sz:
                print(f"[DEBUG] 过滤 {sz} 环：跨边界复合环")
            continue

        # 通过所有检查：加入已选集合
        selected.append(cur)
        poly = get_normalized_polygon(cur, original_coords, lattice_matrix, cell_center)
        if poly is not None and (not poly.is_empty):
            selected_pairs.append((cur, poly))

    print(f"    重叠过滤掉的环数: {overlap_filtered}")
    print(f"    跨边界复合环过滤掉的环数: {boundary_composite_filtered}")
    print(f"    最终选中环数: {len(selected)}")

    # 统计详情
    origC = Counter([len(c) for c in cycles])
    primC = Counter([len(c) for c in primary])
    finalC = Counter([len(c) for c in selected])

    print("  各尺寸环去重详情:")
    print("    尺寸  原始 → 边界过滤 → 最终选中")
    print("    " + "-" * 35)
    for size in sorted(origC.keys()):
        print(f"    {size:2d}元环: {origC[size]:4d} → {primC.get(size,0):8d} → {finalC.get(size,0):6d}")

    return selected


def analyze_unique_rings_enhanced(G, original_coords, lattice_matrix, max_ring_size=10):
    """枚举 + 去重（增强版） → 返回计数与环列表"""
    all_cycles = find_rings_comprehensive(G, max_ring_size)
    unique_cycles = remove_overlapping_rings_enhanced(all_cycles, original_coords, lattice_matrix)
    ring_sizes = [len(c) for c in unique_cycles]
    counter = Counter(ring_sizes)
    print("增强去重后的环尺寸分布:")
    for s in sorted(counter.keys()):
        print(f"  {s}元环: {counter[s]} 个")
    return counter, unique_cycles


# ====== 统计与绘图 ======
def print_statistics_enhanced(ring_counter, total_atoms, total_edges):
    """终端打印汇总统计"""
    print("=" * 70)
    print("非晶石墨烯环分布统计（增强版：跨边界复合环检测）")
    print("=" * 70)
    total_rings = sum(ring_counter.values())
    print(f"总原子数: {total_atoms}")
    print(f"总化学键数: {total_edges}")
    if total_atoms > 0:
        print(f"平均每原子配位数: {2*total_edges/total_atoms:.2f}")
        print(f"平均每原子唯一环数: {total_rings/total_atoms:.2f}")
    print()
    print("唯一环尺寸分布:")
    print("------------------------------")
    for sz in sorted(ring_counter.keys()):
        cnt = ring_counter[sz]
        pct = (cnt / total_rings * 100) if total_rings > 0 else 0.0
        print(f"{sz:2d}-环: {cnt:4d} 个 ({pct:6.1f}%)")
    print("=" * 70)
    print("注意：已应用跨边界复合环检测，排除通过周期性平移包含小环的大环")
    print("特别针对8~10元环的过度统计问题进行了优化")


def plot_rings_enhanced(original_coords, edge_coords, cycles, ring_counter, lattice_matrix, outfile="rings_enhanced.png"):
    """结构图 + 统计图（增强版）"""
    print("绘制环分布图（增强版）...")
    cell_center = np.dot([0.5, 0.5, 0.5], lattice_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    try:
        # --- 左图：结构 + 环 ---
        ax1.scatter(original_coords[:, 0], original_coords[:, 1], c="black", s=18, alpha=0.85, zorder=5)
        # 画键
        for (i, j), (ci, cj) in edge_coords.items():
            ax1.plot([ci[0], cj[0]], [ci[1], cj[1]], linewidth=0.6, alpha=0.6, zorder=1)

        # 环颜色表
        colors = {3: "red", 4: "blue", 5: "green", 6: "orange", 7: "purple", 8: "brown", 9: "pink", 10: "cyan"}
        print("绘制环结构...")
        ring_counts_drawn = defaultdict(int)
        max_draw = 200  # 每类最多展示数量（避免遮挡）

        for cyc in cycles:
            k = len(cyc)
            if k not in colors:
                continue
            if ring_counts_drawn[k] >= max_draw:
                continue
            coords = adjust_ring_coordinates(cyc, original_coords, lattice_matrix)
            center = np.mean(coords, axis=0)
            offset = find_best_shift(center, lattice_matrix, cell_center)
            coords = coords + offset

            # 边
            for t in range(len(cyc)):
                u = (t + 1) % len(cyc)
                ax1.plot([coords[t, 0], coords[u, 0]], [coords[t, 1], coords[u, 1]],
                         c=colors[k], lw=1.8, alpha=0.85, zorder=3)
            # 标注
            cxy = np.mean(coords, axis=0)
            ax1.text(cxy[0], cxy[1], str(k), color=colors[k], ha="center", va="center",
                     fontsize=8, fontweight="bold", zorder=4,
                     bbox=dict(boxstyle="circle,pad=0.1", facecolor="white",
                               alpha=0.7, edgecolor=colors[k]))
            ring_counts_drawn[k] += 1

        # 晶胞边界（2D 投影）
        a2d = lattice_matrix[:2, 0]
        b2d = lattice_matrix[:2, 1]
        origin = np.array([0.0, 0.0])
        corners = np.array([
            origin,
            origin + a2d,
            origin + a2d + b2d,
            origin + b2d,
            origin
        ])
        ax1.plot(corners[:, 0], corners[:, 1], 'k--', linewidth=2, alpha=0.7, zorder=2)

        ax1.set_xlabel('X (Å)', fontsize=12)
        ax1.set_ylabel('Y (Å)', fontsize=12)
        ax1.set_title('Ring Distribution (Enhanced: Boundary Composite Filter)\nin Amorphous Graphene', fontsize=12)
        ax1.axis('equal')
        ax1.grid(True, alpha=0.25)

        # 图例
        legend_handles = []
        for sz in sorted(colors.keys()):
            if sz in ring_counter:
                legend_handles.append(plt.Line2D([0], [0], color=colors[sz], lw=2,
                                                 label=f'{sz}-ring ({ring_counter[sz]})'))
        if legend_handles:
            ax1.legend(handles=legend_handles, loc='upper right')

        # --- 右图：柱状统计 ---
        if ring_counter:
            sizes = sorted(ring_counter.keys())
            counts = [ring_counter[s] for s in sizes]
            bars = ax2.bar(sizes, counts, edgecolor='black', linewidth=1,
                           color=[colors.get(s, 'gray') for s in sizes], alpha=0.9)
            for b, v in zip(bars, counts):
                ax2.text(b.get_x() + b.get_width()/2., v + max(counts)*0.01,
                         f"{v}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax2.set_xlabel('Ring Size', fontsize=12)
            ax2.set_ylabel('Count (Enhanced Filter)', fontsize=12)
            ax2.set_title('Enhanced Ring Size Distribution\n(Boundary Composite Filtered)', fontsize=12)
            ax2.grid(True, alpha=0.25)
            ax2.set_xticks(sizes)

        plt.tight_layout()
        plt.savefig(outfile, dpi=220, bbox_inches='tight')
        print(f"[OK] 增强版环分布图已保存到: {outfile}")
    finally:
        plt.close(fig)


def _plot_total_hist_enhanced(total_counter, outfile="total_rings_enhanced.png"):
    """总体柱状图（100 个样本汇总等）"""
    if not total_counter:
        print("[WARN] 无总计数据，跳过绘图")
        return
    sizes = sorted(total_counter.keys())
    counts = [total_counter[s] for s in sizes]
    fig = plt.figure(figsize=(6.6, 4.8))
    try:
        bars = plt.bar(sizes, counts, edgecolor='black', linewidth=1)
        for b, v in zip(bars, counts):
            plt.text(b.get_x() + b.get_width()/2., v + max(counts) * 0.02,
                     f"{v}", ha="center", va="bottom", fontsize=10)
        plt.xlabel("Ring Size")
        plt.ylabel("Total Count (Enhanced Filter)")
        plt.title("Total Unique Ring Size Distribution (Enhanced)\nBatch Summary with Boundary Composite Filter")
        plt.xticks(sizes)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outfile, dpi=220, bbox_inches='tight')
        print(f"[OK] 增强版总计图保存为: {outfile}")
    finally:
        plt.close(fig)


def _format_counts_block_simple(title, counter, sizes=(3, 4, 5, 6, 7, 8, 9, 10)):
    """将计数器格式化为简单文本块"""
    lines = [str(title), "-" * 30]
    total = 0
    for s in sizes:
        v = counter.get(s, 0)
        total += v
        lines.append(f"{s:2d}-环: {v:4d}")
    lines.append(f"合计: {total}")
    lines.append("")
    return "\n".join(lines)


# ====== 主流程（批处理 001~100）======
def main_processing_enhanced():
    """
    主处理函数（增强版）
    扫描 001.vasp ~ 100.vasp，逐个统计并绘图，汇总结果
    """
    t0 = time.time()
    plt.rcParams['figure.max_open_warning'] = 0

    summary_path = "ring_counts_summary_enhanced.txt"
    total_counter = Counter()
    blocks = []

    for idx in range(1, 101):
        poscar_file = f"{idx:03d}.vasp"
        png_path = f"{idx:03d}_enhanced.png"
        if not os.path.exists(poscar_file):
            print(f"[SKIP] 未找到 {poscar_file}")
            continue

        print(f"\n处理文件 {idx}/100: {poscar_file} (增强版)")
        try:
            G, coords, edge_coords, structure, lattice_matrix = build_fast_periodic_graph(
                poscar_file, cutoff=1.6
            )

            ring_counter, unique_cycles = analyze_unique_rings_enhanced(
                G, coords, lattice_matrix, max_ring_size=10
            )

            print_statistics_enhanced(ring_counter, len(structure), G.number_of_edges())

            plot_rings_enhanced(coords, edge_coords, unique_cycles, ring_counter,
                                lattice_matrix, outfile=png_path)

            blocks.append(_format_counts_block_simple(f"{poscar_file} (Enhanced)", ring_counter))
            total_counter.update(ring_counter)

        except Exception as e:
            print(f"[ERROR] 处理 {poscar_file} 时出错：{e}")
            import traceback
            traceback.print_exc()

    # 写入总结
    blocks.append("001.vasp ... 100.vasp 总的唯一环尺寸分布 (Enhanced)")
    blocks.append("-" * 40)
    for s in (3, 4, 5, 6, 7, 8, 9, 10):
        blocks.append(f"{s:2d}-环: {total_counter.get(s, 0):4d}")
    blocks.append("")
    blocks.append("注意：已应用跨边界复合环检测，特别优化了大环的过度统计问题")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print(f"[OK] 已写入 {summary_path}")

    _plot_total_hist_enhanced(total_counter, outfile="total_rings_enhanced.png")
    print(f"\n总耗时: {time.time() - t0:.2f} 秒")


# ====== 辅助：测试/对比/用法 ======
def test_boundary_detection():
    """最小自测：构造正交晶胞 + 四点环，期望跨边界=True"""
    print("\n测试跨边界检测功能...")
    test_lattice = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    test_coords = np.array([
        [0.5, 0.5, 1.5],
        [9.5, 0.5, 1.5],
        [0.5, 9.5, 1.5],
        [9.5, 9.5, 1.5],
    ])
    cyc = [0, 1, 2, 3]
    flag = is_ring_crossing_boundary(cyc, test_coords, test_lattice)
    print(f"  测试环是否跨边界: {flag}")
    print("  预期结果: True （因为原子分布在晶胞的对角）")
    return flag


def compare_versions():
    """
    将本增强版输出与旧版（文件名 ring_counts_summary.txt）对比的占位函数
    """
    try:
        with open("ring_counts_summary.txt", "r", encoding="utf-8") as f:
            _ = f.read()
        with open("ring_counts_summary_enhanced.txt", "r", encoding="utf-8") as f:
            _ = f.read()
        print("\n版本对比分析：")
        print("=" * 50)
        print("原版本文件存在，增强版本文件存在")
        print("可以对比两个文件，观察 8~10 元环数量是否下降，小环基本不变，总环数略减更准确")
        print("=" * 50)
        return True
    except FileNotFoundError as e:
        print(f"\n版本对比失败：{e}")
        print("请先运行原版本和增强版本程序（或仅保留增强版结果进行检查）")
        return False


def usage_instructions():
    print(r"""
使用说明
========
1. 准备数据：
   - 确保工作目录中有 001.vasp 到 100.vasp（或其中一部分）
2. 运行程序：
   - python enhanced_ring_analysis.py
   - 可选参数：
       --help / -h     显示帮助
       --test          运行跨边界检测自测
       --compare       与旧版结果文件对比（若 ring_counts_summary.txt 存在）
3. 输出文件：
   - ring_counts_summary_enhanced.txt: 详细统计
   - total_rings_enhanced.png: 总计柱状图
   - XXX_enhanced.png: 各结构的环分布图
4. 关键参数（如需调整）：
   - build_fast_periodic_graph(..., cutoff=1.6) 中的 cutoff
   - is_ring_crossing_boundary 的 boundary_threshold（默认 0.7）
   - check_shifted_ring_contains_smaller_rings 的 contain_threshold（默认 0.8）
   - 覆盖去重：same_size_iou=0.85 / same_size_vertex=0.50 / general_area_cover=0.5
   - DEBUG：将 DEBUG_RING_FILTER_REASON 设为某个数字（如 7）查看该类环被过滤原因
""")


# ====== 入口 ======
if __name__ == "__main__":
    # 命令行分支
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--help", "-h"):
            usage_instructions()
            sys.exit(0)
        elif sys.argv[1] == "--test":
            ok = test_boundary_detection()
            sys.exit(0 if ok else 1)
        elif sys.argv[1] == "--compare":
            res = compare_versions()
            sys.exit(0 if res else 1)
        else:
            print("未知参数。使用 --help 查看使用说明")
            sys.exit(1)

    print("=" * 80)
    print("非晶石墨烯环分布分析 - 增强版（环面均值 + 几何/拓扑/重心距离覆盖 + 跨边界复合）")
    print("=" * 80)
    print("新增功能：")
    print("1. 跨边界环识别")
    print("2. 周期性平移检测")
    print("3. 复合环过滤（≥8 元环）")
    print("4. 环面均值（toroidal mean）重心判定，修复 0/1 接缝误杀")
    print("5. 几何+拓扑覆盖去重（IoU/顶点重合）+ 重心距离保护")
    print("=" * 80)

    main_processing_enhanced()

    print("\n" + "=" * 80)
    print("增强版处理完成！")
    print("输出文件：")
    print("  - ring_counts_summary_enhanced.txt (详细统计)")
    print("  - total_rings_enhanced.png (总计图)")
    print("  - XXX_enhanced.png (各文件的环分布图)")
    print("=" * 80)

