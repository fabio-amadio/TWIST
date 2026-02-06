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
    path = "../assets/g1/g1_custom_collision_with_fixed_hand.urdf"
    # path = "../assets/g1/g1_29dof_custom.urdf"

    print(f"Loading: {path}")
    bodies, joints = _collect_names_from_urdf(path)

    print(f"\nBodies (links): {len(bodies)}")
    for name in bodies:
        print(f"  {name}")

    print(f"\nJoints (non-fixed): {len(joints)}")
    for idx, name in enumerate(joints):
        print(f"  [{idx}] {name}")



if __name__ == "__main__":
    main()
