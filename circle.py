#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四轴机械臂 + 横杆：参数化正逆运动学最小实现。

坐标约定
--------
基座系：X 向前，Z 向上，Y 按右手系。
长度单位：m。角度在内部一律用 rad，打印时转 deg。

运动链
------
肩(z0) -> 大臂俯仰 a -> 伸缩 d -> 腕俯仰 p -> 横杆自转 r
横杆中心 C 在工具轴上，两端吸盘在横杆局部 Y 方向 ±L 处。

正解
----
theta = a + p
C = [ (L0+d)*cos(a) + L2*cos(theta), 0,
      z0 + (L0+d)*sin(a) + L2*sin(theta) ]
R = Ry(-theta) @ Rx(r)
u = R @ [0,1,0]          # 横杆方向
n = R @ [0,0,1]          # 吸盘法向（工具 Z）
P1 = C + L*u + h*n
P2 = C - L*u + h*n

逆解
----
给定 P1、P2，先检查 |P1-P2| = 2L，再求 u、r、theta，
由中点扣掉杯高 h*n 得到横杆中心 C，最后解 a、d、p。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# ---------------------------------------------------------------------------
# 1. 参数：所有长度、限位集中在这里，改尺寸只改这一处
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    z0: float = 0.07          # 肩关节高度，m
    L0: float = 0.543         # 肩到腕的零位距离，m
    L2: float = 0.065         # 腕到横杆中心的距离，m
    d_min: float = 0.0        # 伸缩下限，m
    d_max: float = 0.266      # 伸缩上限，m
    bar_half: float = 0.10    # 横杆半长 L，m（两杯中心距 = 2L）
    cup_h: float = 0.0        # 吸盘接触点相对横杆中心的法向偏移，m

    a_min: float = 0.0
    a_max: float = math.radians(170.0)
    p_min: float = math.radians(-360.0)
    p_max: float = math.radians(360.0)
    r_min: float = math.radians(-360.0)
    r_max: float = math.radians(360.0)


# ---------------------------------------------------------------------------
# 2. 状态：四个关节，单位 rad / m
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmState:
    a: float   # 大臂俯仰，rad
    d: float   # 伸缩，m
    p: float   # 末端相对俯仰，rad
    r: float   # 横杆自转，rad


# ---------------------------------------------------------------------------
# 3. 正解结果：把中间量都带出来，方便画图和调试
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CupPair:
    center: np.ndarray       # 横杆中心 C
    rotation: np.ndarray     # 工具旋转 R
    bar_dir: np.ndarray      # 横杆方向 u
    normal: np.ndarray       # 吸盘法向 n
    cup_plus: np.ndarray     # P+
    cup_minus: np.ndarray    # P-
    theta: float             # 工具绝对俯仰 a+p


# ---------------------------------------------------------------------------
# 4. 基础旋转矩阵
# ---------------------------------------------------------------------------
def ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


# ---------------------------------------------------------------------------
# 5. 正解：给定关节，算横杆中心、方向、两个吸盘
# ---------------------------------------------------------------------------
def forward(state: ArmState, spec: ArmSpec = ArmSpec()) -> CupPair:
    theta = state.a + state.p
    L1 = spec.L0 + state.d

    center = np.array([
        L1 * math.cos(state.a) + spec.L2 * math.cos(theta),
        0.0,
        spec.z0 + L1 * math.sin(state.a) + spec.L2 * math.sin(theta),
    ])

    rotation = ry(-theta) @ rx(state.r)
    bar_dir = rotation @ np.array([0.0, 1.0, 0.0])
    normal = rotation @ np.array([0.0, 0.0, 1.0])

    # 两个杯子在横杆局部坐标里是 (0, ±L, h)
    cup_plus = center + spec.bar_half * bar_dir + spec.cup_h * normal
    cup_minus = center - spec.bar_half * bar_dir + spec.cup_h * normal

    return CupPair(center, rotation, bar_dir, normal, cup_plus, cup_minus, theta)


# ---------------------------------------------------------------------------
# 6. 限位检查：逆解最后必须过这一关
# ---------------------------------------------------------------------------
def validate(state: ArmState, spec: ArmSpec = ArmSpec()) -> None:
    checks = (
        ("a", state.a, spec.a_min, spec.a_max, True),
        ("d", state.d, spec.d_min, spec.d_max, False),
        ("p", state.p, spec.p_min, spec.p_max, True),
        ("r", state.r, spec.r_min, spec.r_max, True),
    )
    for name, value, low, high, is_angle in checks:
        if not math.isfinite(value):
            raise ValueError(f"{name} is not finite")
        if not (low - 1e-12 <= value <= high + 1e-12):
            unit_value = math.degrees(value) if is_angle else value
            unit_low = math.degrees(low) if is_angle else low
            unit_high = math.degrees(high) if is_angle else high
            unit = "deg" if is_angle else "m"
            raise ValueError(
                f"{name}={unit_value:.3f}{unit} outside [{unit_low:.3f}, {unit_high:.3f}]{unit}"
            )


# ---------------------------------------------------------------------------
# 7. 逆解：给定两个吸盘位置，回算 a、d、p、r
# ---------------------------------------------------------------------------
def inverse_from_cups(
    cup_plus,
    cup_minus,
    spec: ArmSpec = ArmSpec(),
    *,
    theta_hint: float = 0.0,
    tol: float = 1e-6,
) -> ArmState:
    """由两个吸盘位置反算四个关节。

    参数
    ----
    cup_plus, cup_minus : 形状 (3,) 的基座系坐标，单位 m。
    theta_hint : 横杆正好沿 ±Y 时，theta 不被横杆方向约束，用它选一个解。
    tol : 长度/中点检查容差，单位 m。

    返回
    ----
    ArmState

    异常
    ----
    ValueError：长度不匹配、中点不在 Y=0 平面、关节超限、回代误差过大。
    """
    P1 = np.asarray(cup_plus, dtype=float)
    P2 = np.asarray(cup_minus, dtype=float)
    if P1.shape != (3,) or P2.shape != (3,):
        raise ValueError("cup positions must be 3D vectors")

    v = P1 - P2
    length = float(np.linalg.norm(v))
    if abs(length - 2.0 * spec.bar_half) > tol:
        raise ValueError(
            f"bar length mismatch: |P1-P2|={length:.6f} m, expected {2*spec.bar_half:.6f} m"
        )

    u = v / length
    midpoint = 0.5 * (P1 + P2)
    sin_abs = float(math.hypot(u[0], u[2]))

    # u = [-sin(theta)*sin(r), cos(r), cos(theta)*sin(r)]
    # sin(r) 的符号不能由 |u_xz| 单独确定，因此正负两个分支都试。
    # h=0 时两支可能都合法（只差 theta 的 180°），选 |r| 小的；
    # h≠0 时杯中点 C=m-h*n 的 Cy=0 会选出唯一一支。
    candidates: list[tuple[float, float, ArmState]] = []

    def try_candidate(r_value: float, theta_value: float) -> None:
        normal = np.array([
            -math.sin(theta_value) * math.cos(r_value),
            -math.sin(r_value),
            math.cos(theta_value) * math.cos(r_value),
        ])
        center = midpoint - spec.cup_h * normal

        # 从横杆中心扣掉工具段 L2，得到腕关节位置
        wrist = center - spec.L2 * np.array([
            math.cos(theta_value), 0.0, math.sin(theta_value)
        ])
        L1 = math.hypot(float(wrist[0]), float(wrist[2] - spec.z0))
        a = math.atan2(float(wrist[2] - spec.z0), float(wrist[0]))
        d = L1 - spec.L0
        p = theta_value - a

        state = ArmState(a=a, d=d, p=p, r=r_value)
        try:
            validate(state, spec)
        except ValueError:
            return

        # 回代检查：防止公式和几何约定不一致
        check = forward(state, spec)
        error = max(
            float(np.linalg.norm(check.cup_plus - P1)),
            float(np.linalg.norm(check.cup_minus - P2)),
        )
        if error <= tol:
            candidates.append((abs(float(center[1])), abs(r_value), state))

    if sin_abs > 1e-9:
        for sign in (1.0, -1.0):
            sin_r = sign * sin_abs
            r = math.atan2(sin_r, float(u[1]))
            # sin(theta) = -u_x/sin(r), cos(theta) = u_z/sin(r)
            theta = math.atan2(-float(u[0]) / sin_r, float(u[2]) / sin_r)
            try_candidate(r, theta)
    else:
        # 横杆沿 ±Y：theta 是零空间，用 hint 的两个相差 180° 的分支试。
        r = 0.0 if float(u[1]) >= 0.0 else math.pi
        try_candidate(r, float(theta_hint))
        try_candidate(r, float(theta_hint) + math.pi)

    if not candidates:
        raise ValueError(
            "no feasible inverse solution; check bar length, Y=0 plane, joint limits"
        )

    # 先保证中点确实在 Y=0 平面，再取 |r| 最小的解（连续性更好）
    best_cy, _, best = min(candidates, key=lambda item: (item[0], item[1]))
    if best_cy > tol:
        raise ValueError(
            f"cup midpoint is not in the Y=0 plane: Cy={best_cy:.6f} m"
        )
    return best


# ---------------------------------------------------------------------------
# 8. 演示：正解 -> 逆解 -> 回代
# ---------------------------------------------------------------------------
def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    spec = ArmSpec(bar_half=0.10, cup_h=0.0)

    state = ArmState(
        a=math.radians(20.0),
        d=0.05,
        p=math.radians(-10.0),
        r=math.radians(15.0),
    )

    cups = forward(state, spec)
    print("=== 正解 ===")
    print("C  =", cups.center)
    print("P+ =", cups.cup_plus)
    print("P- =", cups.cup_minus)
    print("u  =", cups.bar_dir)
    print("n  =", cups.normal)
    print(f"theta = {math.degrees(cups.theta):.3f} deg")

    solved = inverse_from_cups(cups.cup_plus, cups.cup_minus, spec)
    print("\n=== 逆解 ===")
    print(f"a = {math.degrees(solved.a):.3f} deg")
    print(f"d = {solved.d:.6f} m")
    print(f"p = {math.degrees(solved.p):.3f} deg")
    print(f"r = {math.degrees(solved.r):.3f} deg")

    again = forward(solved, spec)
    print("\n=== 回代误差 ===")
    print("P+ error =", np.linalg.norm(again.cup_plus - cups.cup_plus))
    print("P- error =", np.linalg.norm(again.cup_minus - cups.cup_minus))


if __name__ == "__main__":
    main()