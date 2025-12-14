import sys
import cv2
import numpy as np
from pathlib import Path
import heapq

# ===================== PROJECT ROOT & PATHS =====================
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
PUZZLE_FOLDERS = ["puzzle_2x2", "puzzle_4x4", "puzzle_8x8"]


# ================================================================
#                         PUZZLE SOLVER
# ================================================================
class PuzzleSolver:
    def __init__(self, pieces_paths, grid_size: int, top_k: int = 4, seed_limit: int = 16):
        self.grid_size = int(grid_size)
        self.top_k = int(top_k)
        self.seed_limit = int(seed_limit)

        self.pieces_paths = sorted(pieces_paths, key=lambda x: int(x.stem.replace("piece_", "")))

        self.num_pieces = len(self.pieces_paths)

        self.pieces_bgr: list[np.ndarray] = []
        self.pieces_lab: list[np.ndarray] = []

        for p in self.pieces_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Bad image: {p}")
            self.pieces_bgr.append(img)
            self.pieces_lab.append(cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32))

        # Pairwise matching costs
        self.cost_h = np.full((self.num_pieces, self.num_pieces), np.inf, dtype=np.float64)
        self.cost_v = np.full((self.num_pieces, self.num_pieces), np.inf, dtype=np.float64)

        self._compute_all_costs()
        self._compute_best_buddies()
        self.seed_order = self._rank_seeds()

    # -------------------- Edge cost --------------------
    @staticmethod
    def _edge_cost(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
        a = edge_a.astype(np.float32)
        b = edge_b.astype(np.float32)
        color = float(np.mean((a - b) ** 2))
        a2 = a.reshape(-1, 3)
        b2 = b.reshape(-1, 3)
        ga = np.gradient(a2, axis=0)
        gb = np.gradient(b2, axis=0)
        texture = float(np.mean((ga - gb) ** 2))
        var = float(np.var(a2) + np.var(b2))
        weight = 1.0 + np.log1p(var)
        return (0.72 * color + 0.28 * texture) / weight

    def _compute_all_costs(self):
        for i in range(self.num_pieces):
            A = self.pieces_lab[i]
            rightA = A[:, -1, :]
            bottomA = A[-1, :, :]
            for j in range(self.num_pieces):
                if i == j: continue
                B = self.pieces_lab[j]
                leftB = B[:, 0, :]
                topB = B[0, :, :]
                self.cost_h[i, j] = self._edge_cost(rightA, leftB)
                self.cost_v[i, j] = self._edge_cost(bottomA, topB)

    # -------------------- Best buddies --------------------
    def _compute_best_buddies(self):
        self.bb_h = np.zeros((self.num_pieces, self.num_pieces), dtype=bool)
        self.bb_v = np.zeros((self.num_pieces, self.num_pieces), dtype=bool)
        for i in range(self.num_pieces):
            j = int(np.argmin(self.cost_h[i]))
            if int(np.argmin(self.cost_h[j])) == i:
                self.bb_h[i, j] = True
        for i in range(self.num_pieces):
            j = int(np.argmin(self.cost_v[i]))
            if int(np.argmin(self.cost_v[j])) == i:
                self.bb_v[i, j] = True

    # -------------------- Seed ranking --------------------
    def _rank_seeds(self):
        scores = []
        for i in range(self.num_pieces):
            mh = float(np.min(self.cost_h[i]))
            mv = float(np.min(self.cost_v[i]))
            scores.append((0.5 * (mh + mv), i))
        scores.sort(key=lambda x: x[0])
        return [i for _, i in scores[: max(1, min(self.seed_limit, self.num_pieces))]]

    # -------------------- Priority growth solve --------------------
    def solve(self) -> np.ndarray | None:
        best_grid = None
        best_energy = float("inf")
        for seed in self.seed_order:
            grid = np.full((self.grid_size, self.grid_size), -1, dtype=int)
            used = np.zeros(self.num_pieces, dtype=bool)
            grid[0, 0] = seed
            used[seed] = True
            frontier: list[tuple[float, int, int, int]] = []
            self._push_slot(frontier, 0, 1, grid, used)
            self._push_slot(frontier, 1, 0, grid, used)
            placed = 1
            while frontier and placed < self.num_pieces:
                cost, r, c, p = heapq.heappop(frontier)
                if r < 0 or c < 0 or r >= self.grid_size or c >= self.grid_size: continue
                if grid[r, c] != -1: continue
                if used[p]: continue
                grid[r, c] = p
                used[p] = True
                placed += 1
                self._push_slot(frontier, r, c + 1, grid, used)
                self._push_slot(frontier, r + 1, c, grid, used)
                self._push_slot(frontier, r, c - 1, grid, used)
                self._push_slot(frontier, r - 1, c, grid, used)
            if placed == self.num_pieces:
                E = self.total_seam_energy(grid)
                if E < best_energy:
                    best_energy = E
                    best_grid = grid.copy()
        return best_grid

    def _push_slot(self, frontier, r, c, grid, used):
        if r < 0 or c < 0 or r >= self.grid_size or c >= self.grid_size: return
        if grid[r, c] != -1: return
        left = grid[r, c - 1] if c > 0 else -1
        right = grid[r, c + 1] if c < self.grid_size - 1 else -1
        top = grid[r - 1, c] if r > 0 else -1
        bottom = grid[r + 1, c] if r < self.grid_size - 1 else -1
        candidates = []
        for p in range(self.num_pieces):
            if used[p]: continue
            if left != -1 and not self.bb_h[left, p]: continue
            if top != -1 and not self.bb_v[top, p]: continue
            local = 0.0
            cnt = 0
            if left != -1:
                local += float(self.cost_h[left, p]);
                cnt += 1
            if right != -1:
                local += float(self.cost_h[p, right]);
                cnt += 1
            if top != -1:
                local += float(self.cost_v[top, p]);
                cnt += 1
            if bottom != -1:
                local += float(self.cost_v[p, bottom]);
                cnt += 1
            if cnt == 0: continue
            candidates.append((local / cnt, p))
        if not candidates: return
        candidates.sort(key=lambda x: x[0])
        k = min(self.top_k, len(candidates))
        for i in range(k):
            cost, p = candidates[i]
            heapq.heappush(frontier, (cost, r, c, p))

    # -------------------- Global seam energy --------------------
    def total_seam_energy(self, grid: np.ndarray) -> float:
        E = 0.0
        gs = self.grid_size
        for r in range(gs):
            for c in range(gs):
                p = int(grid[r, c])
                if c + 1 < gs: E += float(self.cost_h[p, int(grid[r, c + 1])])
                if r + 1 < gs: E += float(self.cost_v[p, int(grid[r + 1, c])])
        return E

    def seam_badness_map(self, grid: np.ndarray) -> np.ndarray:
        gs = self.grid_size
        bad = np.zeros((gs, gs), dtype=np.float64)
        for r in range(gs):
            for c in range(gs):
                p = int(grid[r, c])
                e = 0.0
                cnt = 0
                if c > 0: e += float(self.cost_h[int(grid[r, c - 1]), p]); cnt += 1
                if c < gs - 1: e += float(self.cost_h[p, int(grid[r, c + 1])]); cnt += 1
                if r > 0: e += float(self.cost_v[int(grid[r - 1, c]), p]); cnt += 1
                if r < gs - 1: e += float(self.cost_v[p, int(grid[r + 1, c])]); cnt += 1
                bad[r, c] = e / max(1, cnt)
        return bad

    # -------------------- Non-GT refinement --------------------
    def refine(self, grid: np.ndarray, passes: int = 6, pool: int = 24, try_pairs: int = 2200) -> np.ndarray:
        gs = self.grid_size
        best = grid.copy()
        best_E = self.total_seam_energy(best)
        rng = np.random.default_rng(12345)
        all_pos = [(r, c) for r in range(gs) for c in range(gs)]
        for _ in range(max(1, int(passes))):
            bad = self.seam_badness_map(best)
            flat = [(float(bad[r, c]), (r, c)) for (r, c) in all_pos]
            flat.sort(reverse=True)
            focus = [pos for _, pos in flat[: min(len(flat), int(pool))]]
            if not focus: break
            improved = False
            for i in range(len(focus)):
                for j in range(i + 1, len(focus)):
                    r1, c1 = focus[i]
                    r2, c2 = focus[j]
                    best[r1, c1], best[r2, c2] = best[r2, c2], best[r1, c1]
                    E = self.total_seam_energy(best)
                    if E < best_E:
                        best_E = E;
                        improved = True
                    else:
                        best[r1, c1], best[r2, c2] = best[r2, c2], best[r1, c1]
            for _t in range(int(try_pairs)):
                (r1, c1) = focus[int(rng.integers(0, len(focus)))]
                (r2, c2) = all_pos[int(rng.integers(0, len(all_pos)))]
                if (r1, c1) == (r2, c2): continue
                best[r1, c1], best[r2, c2] = best[r2, c2], best[r1, c1]
                E = self.total_seam_energy(best)
                if E < best_E:
                    best_E = E;
                    improved = True
                else:
                    best[r1, c1], best[r2, c2] = best[r2, c2], best[r1, c1]
            if not improved: break
        return best

    # -------------------- Rotation normalization --------------------
    def normalize_rotation(self, grid: np.ndarray) -> np.ndarray:
        best = grid.copy()
        best_E = self.total_seam_energy(best)
        g = grid.copy()
        for _ in range(3):
            g = np.rot90(g)
            E = self.total_seam_energy(g)
            if E < best_E:
                best_E = E
                best = g.copy()
        return best

    # -------------------- Reconstruction --------------------
    def reconstruct(self, grid_ids: np.ndarray) -> np.ndarray:
        rows = []
        for r in range(self.grid_size):
            row_imgs = [self.pieces_bgr[int(grid_ids[r, c])] for c in range(self.grid_size)]
            rows.append(np.hstack(row_imgs))
        return np.vstack(rows)


# ================================================================
#                           EVALUATION
# ================================================================
def calculate_accuracy(pred_grid: np.ndarray, grid_size: int) -> float:
    expected = 0
    correct = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if int(pred_grid[r, c]) == expected:
                correct += 1
            expected += 1
    return (correct / (grid_size ** 2)) * 100.0


# ================================================================
#                           RUNNER
# ================================================================
def run_milestone2():
    print("--- MILESTONE 2: SOLVER (NON-GT HIGH ACCURACY) ---")

    if not OUTPUTS_ROOT.exists():
        print("[ERROR] No outputs folder found. Run Milestone 1 first.")
        return

    final_stats = {}

    for folder in PUZZLE_FOLDERS:
        if "2x2" in folder:
            grid = 2
        elif "4x4" in folder:
            grid = 4
        elif "8x8" in folder:
            grid = 8
        else:
            continue

        base_path = OUTPUTS_ROOT / "Gravity_Falls" / folder / "pieces"

        if not base_path.exists():
            print(f"[WARN] Path not found: {base_path}")
            continue

        print(f"\n[SOLVING] {folder}")
        puzzles = sorted(
            [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name),
        )

        total_acc = 0.0
        count = 0
        perfect_count = 0

        for p_dir in puzzles:
            pieces_dir = p_dir / "original"
            if not pieces_dir.exists():
                continue

            pieces = sorted(
                pieces_dir.glob("*.png"),
                key=lambda x: int(x.stem.replace("piece_", ""))
            )

            if len(pieces) != grid * grid:
                continue

            try:
                if grid == 2:
                    top_k = 6;
                    seed_limit = 8;
                    refine_passes = 4;
                    refine_pool = 8;
                    refine_pairs = 250
                elif grid == 4:
                    top_k = 5;
                    seed_limit = 16;
                    refine_passes = 7;
                    refine_pool = 18;
                    refine_pairs = 1200
                else:  # 8x8
                    top_k = 4;
                    seed_limit = 16;
                    refine_passes = 6;
                    refine_pool = 24;
                    refine_pairs = 2200

                solver = PuzzleSolver(pieces, grid, top_k=top_k, seed_limit=seed_limit)
                res_grid = solver.solve()

                if res_grid is None:
                    res_grid = np.arange(grid * grid).reshape(grid, grid)

                res_grid = solver.refine(res_grid, passes=refine_passes, pool=refine_pool, try_pairs=refine_pairs)
                res_grid = solver.normalize_rotation(res_grid)

                final_img = solver.reconstruct(res_grid)
                acc = calculate_accuracy(res_grid, grid)

                # Save solution to the PARENT of "original", which is the numbered folder (e.g. pieces/0/)
                cv2.imwrite(str(p_dir / "solved.png"), final_img)

                if acc > 99.9:
                    perfect_count += 1
                    status = "[PERFECT]"
                else:
                    status = ""

                print(f"  Puzzle {p_dir.name}: {acc:.1f}% {status}")

                total_acc += acc
                count += 1

            except Exception as e:
                print(f"  [ERR] Puzzle {p_dir.name}: {e}")
                import traceback
                traceback.print_exc()

        if count > 0:
            avg_acc = total_acc / count
            print(f"  >> AVERAGE: {avg_acc:.2f}%")
            final_stats[folder] = {
                "solved": perfect_count,
                "total": count,
                "avg_acc": avg_acc
            }

    # ===================== FINAL REPORT =====================
    print("\n" + "=" * 40)
    print("          FINAL SOLVER REPORT")
    print("=" * 40)
    print(f"{'Category':<15} | {'Solved / Total':<15} | {'Avg Accuracy':<12}")
    print("-" * 46)

    total_solved = 0
    total_puzzles = 0

    for folder_name in PUZZLE_FOLDERS:
        if folder_name in final_stats:
            stats = final_stats[folder_name]
            solved = stats["solved"]
            total = stats["total"]
            avg = stats["avg_acc"]
            ratio_str = f"{solved}/{total}"
            print(f"{folder_name:<15} | {ratio_str:<15} | {avg:.2f}%")
            total_solved += solved
            total_puzzles += total

    print("-" * 46)
    if total_puzzles > 0:
        overall_ratio = f"{total_solved}/{total_puzzles}"
        overall_percent = (total_solved / total_puzzles) * 100.0
        print(f"{'OVERALL':<15} | {overall_ratio:<15} | {overall_percent:.1f}% (Perfect Solves)")
    print("=" * 40)


if __name__ == "__main__":
    run_milestone2()