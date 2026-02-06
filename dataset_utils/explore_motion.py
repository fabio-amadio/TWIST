import pickle


def main():
    path = "../track_dataset/twist_motion_dataset_29dof/mocap/0.pkl"
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    k = "fps"
    print(f"  {k}: {data[k]}")
    k = "root_pos"
    print(f"  {k}: {data[k].shape}")
    k = "root_rot"
    print(f"  {k}: {data[k].shape}")
    k = "dof_pos"
    print(f"  {k}: {data[k].shape}")
    k = "local_body_pos"
    print(f"  {k}: {data[k].shape}")
    k = "link_body_list"
    print(f"  {k}: {len(data[k])}")
    for body in data[k]:
        print(f"    {body}")


if __name__ == "__main__":
    main()
