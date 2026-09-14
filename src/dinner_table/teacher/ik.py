"""Weighted damped least-squares inverse kinematics solver for SO-101 robotic arms."""

from __future__ import annotations

import mujoco
import numpy as np

from dinner_table.config import DinnerTableError
from dinner_table.contracts.geometry import ARM_MOUNTS, HOME_JOINTS
from dinner_table.teacher.kinematics import (
    joint_limits,
    set_arm_q,
    site_jacobian,
    site_pose,
)

MAX_ARM_REACH_M = 0.60
SHOULDER_Z_OFFSET_M = 0.195


class IKUnreachable(DinnerTableError):
    """Exception raised when an inverse kinematics target cannot be reached within tolerances."""

    def __init__(self, site: str, target_pos: np.ndarray) -> None:
        self.site = site
        self.target_pos = np.array(target_pos, dtype=np.float64, copy=True)
        super().__init__(
            f"Target position {self.target_pos.tolist()} is unreachable for site '{self.site}'."
        )


def _resolve_arm(identifier: str) -> str:
    """Resolve arm prefix 'A' or 'B' from site name or arm identifier."""
    if identifier.startswith("A"):
        return "A"
    if identifier.startswith("B"):
        return "B"
    raise IKUnreachable(identifier, np.zeros(3, dtype=np.float64))


def _shoulder_pivot_pos(arm: str) -> np.ndarray:
    """Compute world position of the shoulder lift pivot for an arm."""
    mount_xyz = np.array(ARM_MOUNTS[arm], dtype=np.float64)
    mount_xyz[2] = mount_xyz[2] + SHOULDER_Z_OFFSET_M
    return mount_xyz


DOWN_SEEDS = (
    (-1.0847, -1.4114, -1.5410, -1.6040, 3.1293),
    (-0.3154, 1.4599, 1.5755, -1.4394, -2.9934),
    (-1.5513, 1.4383, 1.6434, 1.5668, 0.0980),
    (0.5294, -1.7021, -0.6105, 0.7208, 0.0137),
    (-0.0056, 1.5500, 1.1744, -1.2116, -3.0774),
    (0.2954, -1.6863, -0.6410, 0.7343, -0.1512),
    (0.2140, -1.4868, -1.3738, 1.2985, -0.0954),
    (-1.0561, -1.4573, -1.7147, 1.6115, -0.1367),
    (0.8420, 1.5982, 1.0315, -1.1779, -3.0903),
    (-1.4648, 1.6332, 0.7766, -0.8308, 3.0120),
    (-0.9034, 1.4352, 1.5768, 1.6827, -0.1212),
    (-1.9157, 1.4707, 1.3641, -1.2993, 3.1288),
    (0.0898, 1.3762, 1.6231, 1.6059, -0.0760),
    (0.8363, -1.4313, -1.4966, 1.3548, -0.0696),
    (-0.2216, 1.7325, 0.4155, -0.5116, -3.0190),
    (0.6725, 1.7244, 0.5466, -0.7996, -3.0440),
    (-0.7855, -1.5320, -0.9113, 0.7528, -0.0409),
    (1.5160, 1.5953, 0.8668, -0.9411, -3.0970),
    (1.6999, 1.6854, 0.5621, -0.7048, -3.1344),
    (-0.3664, -1.5931, -0.6844, 0.5644, 0.0037),
    (0.1931, 1.4743, 1.0617, -0.8689, -3.0074),
    (-1.3766, -1.4423, -1.2971, 1.1883, 0.0436),
    (-0.9607, -1.4501, -1.4144, 1.4028, -0.1190),
    (-0.0037, -1.4145, -1.3294, 1.1401, 0.1507),
    (-1.6632, 1.5332, 1.0262, -1.0796, 3.1266),
    (-0.5919, 1.4642, 1.3282, -1.3429, 3.0609),
    (1.1419, -1.7275, -0.3611, 0.4877, 0.1368),
    (-0.3859, -1.3996, -1.6566, 1.5916, -0.0799),
    (-0.3801, 1.5511, 0.8705, -0.8769, -3.1283),
    (-0.2218, -1.6341, -0.5938, 0.6768, 0.0435),
    (1.3401, 1.4673, 1.0886, -1.0487, -3.0403),
    (0.4002, 1.5147, 0.7518, -0.5767, -3.0664),
    (-0.9389, 1.4658, 1.0431, -1.0203, 3.1321),
    (1.7393, -1.3097, -1.5407, 1.2913, -0.0568),
    (1.8456, -1.6415, -0.4323, 0.4417, -0.0787),
    (-1.7628, 1.6047, 0.5390, -0.5398, -3.0832),
    (0.0797, 1.6122, 0.5108, -0.5264, -3.1007),
    (-1.1470, -1.6095, -0.4669, 0.4210, -0.0200),
    (1.5390, 1.2757, 1.4702, -1.1215, -3.1183),
    (1.1381, 1.2869, 1.4927, -1.2163, -3.0012),
    (0.9945, -1.4085, -1.0219, 0.8268, 0.1642),
    (0.2086, -1.3833, -1.1889, 1.0759, -0.0975),
    (0.3687, 1.2241, 1.5215, -1.0709, 3.0826),
    (-0.6503, -1.6914, -0.2176, 0.2499, -0.0582),
    (0.8989, 1.7009, 0.2808, -0.4740, 3.1218),
    (-1.5824, -1.7126, -0.2317, 0.4198, 0.0061),
    (1.4864, -1.1828, -1.7006, 1.3050, -0.1568),
    (-0.2677, 1.3376, 1.1764, -1.0025, -3.0590),
    (-1.0618, 1.4573, 0.8218, -0.7597, 3.1003),
    (0.2386, 1.4786, 0.7605, -0.7769, -3.1385),
    (-0.0150, -1.4858, -0.6400, 0.5257, 0.1106),
    (-0.8082, 1.1785, 1.4438, -0.9760, 3.0451),
    (-0.4148, -1.2175, -1.4560, 1.1558, -0.0799),
    (1.7414, -1.2497, -1.1514, 0.6916, 0.0558),
    (1.4675, 1.3274, 1.1084, -0.9606, 3.0966),
    (-1.4383, -1.1880, -1.3920, 0.9744, -0.1525),
    (0.7161, -1.1355, -1.6195, 1.1921, 0.0648),
    (-0.5613, 1.3735, 0.8720, -0.6659, -3.0129),
    (-1.8324, -1.4732, -0.6136, 0.5184, -0.1554),
    (-0.1594, -1.1639, -1.4724, 1.0818, 0.1156),
    (-1.1371, -1.1484, -1.4549, 1.0059, 0.0142),
    (-0.8919, -1.1602, -1.5939, 1.3242, 0.0020),
    (0.7762, 1.0789, 1.5069, -0.8664, -3.1244),
    (0.3840, -1.3602, -0.9061, 0.7913, 0.0553),
    (1.8526, 1.2904, 1.0619, -0.8470, 3.1415),
    (0.2446, -1.4100, -0.8083, 0.8115, 0.0165),
    (-1.2609, 1.2947, 0.9056, -0.5858, -3.0719),
    (0.4519, 1.4367, 0.5079, -0.2866, -3.0176),
    (1.4213, -1.2933, -0.9139, 0.6350, -0.0168),
    (-1.9089, 1.4414, 0.5662, -0.4632, 2.9958),
    (-0.0015, 1.1244, 1.2581, -0.6928, 3.0531),
    (-0.3756, -1.4335, -0.5561, 0.4187, 0.0214),
    (-0.2995, 1.4620, 0.5052, -0.4357, 2.9902),
    (-0.7545, -1.2583, -0.9316, 0.5798, 0.1029),
    (1.6665, -1.3408, -0.7263, 0.5275, -0.0864),
    (1.8775, -1.0240, -1.4982, 0.9722, 0.0108),
    (1.3300, 1.0058, 1.5330, -0.9793, 3.1021),
    (0.3223, 0.9947, 1.5956, -1.0678, 3.1188),
)

HORIZ_SEEDS = (
    (1.6120, 1.4001, 0.7943, 0.8193, -1.9429),
    (1.2284, 1.6684, 0.1600, 1.1386, 1.4191),
    (0.4646, -1.2210, -1.4562, 0.1285, -1.4155),
    (0.3191, 1.3070, 1.0659, 0.5742, -1.4833),
    (-0.4949, -1.1447, -1.5935, -0.7062, -1.7220),
    (1.1430, 1.1504, 1.5457, 0.2707, 1.5927),
    (-0.8836, -1.1919, -1.4022, -0.8457, -1.8374),
    (0.5068, -1.4220, -1.3249, -1.6161, 1.6402),
    (1.7381, -1.3194, -1.4283, -1.3722, -1.6691),
    (1.4499, 1.5092, 0.5007, 1.2062, 0.3640),
    (1.6881, 1.6378, 0.3603, 0.3771, -1.6106),
    (0.3432, 1.7074, 0.2710, 0.1874, -1.6747),
    (-0.7154, -1.5772, -0.5257, -0.2079, 1.6564),
    (0.2347, 1.5014, 1.3875, -1.3133, 1.5707),
    (0.8362, -1.2117, -1.2059, -0.6539, 1.8728),
    (-0.8678, 1.5451, 0.5829, 0.0477, -1.4517),
    (1.6403, 1.4994, 0.5641, 0.1788, 1.6977),
    (0.1993, -1.4278, -0.9112, 0.3798, -1.5713),
    (1.2889, -1.6321, -0.2397, -0.2932, -1.5141),
    (0.7567, -1.6966, -0.4792, 0.6845, 1.6389),
    (-1.6824, -0.9389, -1.6812, -0.1193, 1.3741),
    (-0.8619, 1.1385, 1.1058, 0.5020, -1.6658),
    (-0.1083, 1.1019, 1.0863, 0.8672, 3.0793),
    (0.6467, -1.1122, -1.0507, -0.6988, -1.2071),
    (-1.1768, 0.9567, 1.6172, 1.2088, 1.7359),
    (0.6615, -1.1942, -1.7193, -1.7172, -1.5000),
    (1.9050, -1.1091, -1.1653, -0.1389, -1.5113),
    (-0.7158, -0.8876, -1.5131, -0.6572, -1.3089),
    (-1.7049, -1.0529, -1.1191, -0.8997, -2.3584),
    (0.1021, 0.7935, 1.7442, 0.7443, 2.1935),
    (0.5293, -1.7326, 0.3980, -1.2505, 1.7229),
    (-1.6707, -1.0672, -1.7309, -1.7427, 1.6604),
    (-1.0960, 1.4850, 0.2568, 0.5086, 1.4822),
    (0.1456, 1.3407, 0.3333, 1.5295, 1.9695),
    (-1.7287, -1.0335, -1.1273, -1.5262, -1.4058),
    (0.2570, -0.7730, -1.5391, -0.6310, 1.2551),
    (-1.0261, -1.6043, 0.2817, -1.2014, 1.4548),
    (1.1343, 1.4550, 0.2233, 0.4794, 1.4783),
    (-1.0271, -0.8755, -1.2188, -1.0503, -0.5212),
    (-0.0408, -1.7305, 0.5955, -1.3752, 1.7254),
    (0.3851, 0.7453, 1.4778, 0.6573, -1.4172),
    (-1.4347, -0.7301, -1.4305, -1.0493, 2.6973),
    (-0.6932, 0.6796, 1.7072, 1.4873, 1.6280),
    (0.9797, -0.5615, -1.7211, -0.8281, 0.6387),
    (0.8789, 0.8044, 1.5214, 1.7205, -1.6411),
    (0.5441, 1.4190, -0.1168, 1.5673, -1.9310),
    (-0.2959, -1.3148, -1.1385, 1.5564, 1.4894),
    (-1.8737, -0.8389, -1.1064, -1.1648, -2.3932),
    (-1.0511, 0.6735, 1.4391, 1.2975, 1.8818),
    (-1.1015, 1.3854, 0.7060, -0.8946, -1.4633),
    (-0.8665, -1.4377, -0.4766, 0.6380, 1.4857),
    (-1.5112, 1.5548, -0.2254, 0.7240, 1.5279),
    (-1.6750, -0.9272, -1.0213, -0.3544, -1.6800),
    (1.9167, 0.9659, 1.4615, -0.9884, 1.5894),
    (0.2551, -0.6664, -1.2913, -1.2560, -0.6709),
    (1.8419, 1.6185, 0.0921, -0.7350, 1.5764),
    (-0.6040, 0.7005, 1.2102, 0.7424, 1.6765),
    (-0.9765, 0.9979, 0.5143, 1.5254, 2.7095),
    (-1.0152, -1.5202, -0.4758, 1.7263, 1.2167),
    (-0.2270, -0.9749, -1.6399, 1.7268, -1.5682),
    (0.4791, 0.5106, 1.4070, 0.8709, 1.7491),
    (0.8253, 1.2880, 0.4923, -0.3663, 1.4983),
    (-1.8693, 0.5106, 1.4131, 0.7569, 1.3942),
    (-1.0797, -1.0478, -1.1028, 0.9068, -1.6629),
    (1.4758, -1.3941, -0.3570, 0.6856, -1.4902),
    (-0.6762, -1.3990, 0.3013, -1.1253, 1.5944),
    (-0.8241, 1.3477, 0.6606, -1.4060, 1.5271),
    (-0.0019, -0.6793, -0.9584, -1.6009, -1.7418),
    (-0.0404, -1.4322, -0.5005, 1.6940, -1.9558),
    (-0.1526, 0.1332, 1.7155, 1.3189, 1.4341),
    (-1.2646, -0.3542, -1.4470, -1.6887, 1.3774),
)


def _mirror_seed(vals: tuple[float, ...]) -> np.ndarray:
    """Mirror a seed across the table centerline for the opposite arm."""
    return np.array((-vals[0], vals[1], vals[2], vals[3], -vals[4]), dtype=np.float64)


def _seed_bank(arm: str) -> list[np.ndarray]:
    """Assemble the full seed bank for one arm from both posture families."""
    bank: list[np.ndarray] = []
    for family in (DOWN_SEEDS, HORIZ_SEEDS):
        for row in family:
            bank.append(np.array(row, dtype=np.float64))
            bank.append(_mirror_seed(row))
    return bank


def _get_basin_seeds(arm: str, q0: np.ndarray) -> list[np.ndarray]:
    """Generate ordered list of diverse kinematic seed configurations for basin hopping."""
    seeds: list[np.ndarray] = [np.array(q0, dtype=np.float64, copy=True)]
    home_q = np.array(HOME_JOINTS[arm][:5], dtype=np.float64)
    seeds.append(home_q)
    seeds.extend(_seed_bank(arm))
    return seeds


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site: str,
    target_pos: np.ndarray,
    target_approach: np.ndarray,
    q0: np.ndarray,
    pos_tol: float = 2e-3,
    ang_tol: float = 0.05235987755982988,
    iters: int = 100,
    damping: float = 0.05,
) -> np.ndarray:
    """Solve inverse kinematics using weighted damped least squares with multi-seed basin hopping."""
    arm = _resolve_arm(site)
    t_pos = np.array(target_pos, dtype=np.float64)
    t_app = np.array(target_approach, dtype=np.float64)
    app_norm = float(np.linalg.norm(t_app))
    if app_norm > 1e-9:
        t_app = t_app / app_norm
    else:
        t_app = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    pivot = _shoulder_pivot_pos(arm)
    dist_from_pivot = float(np.linalg.norm(t_pos - pivot))
    if dist_from_pivot > MAX_ARM_REACH_M:
        raise IKUnreachable(site, t_pos)

    q_min, q_max = joint_limits(model, arm)
    weight_diag = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3], dtype=np.float64)
    w_mat = np.diag(weight_diag)
    lambda_sq = damping**2
    eye6 = np.eye(6, dtype=np.float64)

    seeds = _get_basin_seeds(arm, q0)
    saved_qpos = np.array(data.qpos, dtype=np.float64, copy=True)
    hop_rng = np.random.default_rng(0)

    # Order the bank seeds by FK proximity to the target so the limited
    # per-seed iteration budget is spent where it matters.
    q0_seed = seeds[0]
    home_seed = seeds[1]
    bank = seeds[2:]
    ranked: list[tuple[float, np.ndarray]] = []
    for seed in bank:
        q_try = np.clip(seed, q_min, q_max)
        set_arm_q(data, site, q_try)
        mujoco.mj_forward(model, data)
        seed_pos, _ = site_pose(data, site)
        ranked.append((float(np.linalg.norm(seed_pos - t_pos)), q_try))
    ranked.sort(key=lambda item: item[0])
    ordered = [q0_seed, home_seed] + [q for _, q in ranked[:32]]

    try:
        for seed_idx, seed in enumerate(ordered):
            q = np.clip(seed, q_min, q_max)
            if seed_idx == 0:
                steps_for_seed = iters
            else:
                steps_for_seed = 35
            stall_count = 0

            for _ in range(steps_for_seed):
                set_arm_q(data, site, q)
                mujoco.mj_forward(model, data)
                cur_pos, cur_rot = site_pose(data, site)

                pos_err = t_pos - cur_pos
                cur_approach = cur_rot[:, 2]
                ang_err = np.cross(cur_approach, t_app)

                pos_err_norm = float(np.linalg.norm(pos_err))
                # The cross-product norm is zero for anti-aligned tools too,
                # so the convergence angle must come from the dot product.
                approach_dot = float(np.clip(np.dot(cur_approach, t_app), -1.0, 1.0))
                ang_err_norm = float(np.arccos(approach_dot))

                if pos_err_norm < pos_tol and ang_err_norm < ang_tol:
                    return q

                err = np.concatenate([pos_err, ang_err])
                j_mat = site_jacobian(model, data, site)

                wj = w_mat @ j_mat
                a_mat = wj @ wj.T + lambda_sq * eye6
                dq = wj.T @ np.linalg.solve(a_mat, w_mat @ err)

                max_step = float(np.max(np.abs(dq)))
                if max_step < 1e-4:
                    stall_count += 1
                    if stall_count >= 3:
                        perturb = hop_rng.uniform(-0.08, 0.08, size=5)
                        q = np.clip(q + perturb, q_min, q_max)
                        stall_count = 0
                    continue
                stall_count = 0
                if max_step > 0.2:
                    dq = dq * (0.2 / max_step)

                q = np.clip(q + dq, q_min, q_max)
    finally:
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

    raise IKUnreachable(site, t_pos)


def ik_above(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: str,
    grasp_pose: tuple[np.ndarray, np.ndarray] | np.ndarray | object,
    height: float,
) -> np.ndarray:
    """Solve inverse kinematics for an approach waypoint offset along world positive Z."""
    clean_arm = _resolve_arm(arm)
    if isinstance(grasp_pose, (tuple, list)):
        base_pos = np.array(grasp_pose[0], dtype=np.float64)
    elif hasattr(grasp_pose, "position"):
        base_pos = np.array(grasp_pose.position, dtype=np.float64)
    else:
        base_pos = np.array(grasp_pose, dtype=np.float64)

    target_pos = base_pos + np.array([0.0, 0.0, height], dtype=np.float64)
    target_approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    q0 = np.array(HOME_JOINTS[clean_arm][:5], dtype=np.float64)
    site_name = f"{clean_arm}.ee"

    return solve_ik(
        model=model,
        data=data,
        site=site_name,
        target_pos=target_pos,
        target_approach=target_approach,
        q0=q0,
    )
