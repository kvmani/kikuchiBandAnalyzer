#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from packaging.version import parse as _v
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import Phase
from orix.vector import Miller

def parse_indices(obj):
    if isinstance(obj, str):
        nums = re.findall(r"-?\d+", obj)
        return tuple(int(n) for n in nums)
    elif isinstance(obj, (tuple, list)):
        return tuple(int(n) for n in obj)
    else:
        raise ValueError(f"Bad input: {obj!r}")

def make_phase(cfg):
    L = Lattice(*cfg["lattice"])
    struct = Structure(lattice=L, atoms=[Atom(cfg["name"], [0,0,0])])
    return Phase(name=cfg["name"], space_group=cfg["space_group"], structure=struct)

def belongs_to_group(hkl, desired, phase, tolerance_deg=0.5):
    h = parse_indices(hkl)
    d = parse_indices(desired)
    m1 = Miller(hkl=[h] if len(h)==3 else None, hkil=[h] if len(h)==4 else None, phase=phase)
    m2 = Miller(hkl=[d] if len(d)==3 else None, hkil=[d] if len(d)==4 else None, phase=phase)
    ang = m1.angle_with(m2, use_symmetry=True, degrees=True)[0]
    return abs(ang) < tolerance_deg, ang

def run_tests():
    cubic_cfg = {"name":"Ni", "space_group":225, "lattice":(3.5236,)*3 + (90,90,90)}
    hcp_cfg   = {"name":"Ti", "space_group":194, "lattice":(2.950,2.950,4.686,90,90,120)}
    tests = [
        # cubic
        ("(1,0,0)", "(0,1,0)", "cubic", True),
        ("(1,0,0)", "(1,1,0)", "cubic", False),
        # HCP basal {110}
        ("(1,1,-2,0)", "(2,-1,-1,0)", "hcp", True),
        ("(-1,-1,2,0)", "(1,1,-2,0)", "hcp", True),
        ("(1,1,0", "(1,1,-2,0)", "hcp", True),
        ("-1,-1,2,3", "(1,1,-2,0)", "hcp", False),
    ]
    print("hkl1 vs hkl2 | lattice | equivalent? | angle (°)")
    for h, d, latt, exp in tests:
        cfg = cubic_cfg if latt=="cubic" else hcp_cfg
        phase = make_phase(cfg)
        ok, ang = belongs_to_group(h, d, phase, tolerance_deg=0.5)
        print(f"{h:<12} {d:<12} {latt:<6} → {ok:5} (exp {exp})  angle={ang:.3f}")

if __name__=="__main__":
    run_tests()
