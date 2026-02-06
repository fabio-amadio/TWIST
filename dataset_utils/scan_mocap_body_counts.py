import os
import pickle
from collections import defaultdict


def _find_pkl_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".pkl"):
                yield os.path.join(dirpath, name)


def main():
    root = "../track_dataset/twist_motion_dataset_29dof"
    print(f"Scanning: {root}")

    counts = defaultdict(int)
    total = 0
    errors = 0

    for path in sorted(_find_pkl_files(root)):
        total += 1
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            body_pos = data.get("local_body_pos")
            if body_pos is None:
                print(f"[WARN] missing local_body_pos: {path}")
                errors += 1
                continue
            num_bodies = body_pos.shape[1]
            counts[num_bodies] += 1
            print(f"{path}: {num_bodies}")
        except Exception as e:
            errors += 1
            print(f"[ERR] {path}: {e}")

    print("\nSummary:")
    for num_bodies in sorted(counts.keys()):
        print(f"  bodies={num_bodies}: {counts[num_bodies]}")
    print(f"  total files: {total}")
    print(f"  errors: {errors}")


if __name__ == "__main__":
    main()
