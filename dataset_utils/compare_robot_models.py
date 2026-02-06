import xml.etree.ElementTree as ET


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _collect_names_from_urdf(path: str):
    tree = ET.parse(path)
    root = tree.getroot()

    links = []
    joints = []

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "link":
            name = elem.attrib.get("name")
            if name:
                links.append(name)
        elif tag == "joint":
            name = elem.attrib.get("name")
            joint_type = elem.attrib.get("type")
            if name and joint_type != "fixed":
                joints.append(name)

    return links, joints


def main():
    path_a = "../assets/g1/g1_custom_collision_with_fixed_hand.urdf"
    path_b = "../assets/g1/g1_29dof_custom.urdf"

    print(f"Loading A: {path_a}")
    bodies_a, joints_a = _collect_names_from_urdf(path_a)
    print(f"Loading B: {path_b}")
    bodies_b, joints_b = _collect_names_from_urdf(path_b)

    def print_compare(label, a_list, b_list):
        a_set = set(a_list)
        b_set = set(b_list)

        common = sorted(a_set & b_set)
        only_a = sorted(a_set - b_set)
        only_b = sorted(b_set - a_set)

        print(f"\n{label} comparison:")
        print(f"  common: {len(common)}")
        for name in common:
            print(f"    {name}")

        print(f"  only in A: {len(only_a)}")
        for name in only_a:
            print(f"    {name}")

        print(f"  only in B: {len(only_b)}")
        for name in only_b:
            print(f"    {name}")

    print_compare("Bodies", bodies_a, bodies_b)
    print_compare("Joints", joints_a, joints_b)



if __name__ == "__main__":
    main()
