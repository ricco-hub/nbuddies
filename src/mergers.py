import numpy as np
import time
from .BlackHoles_Struct import BlackHole
from .merge_gravitree import Node

# https://www.nist.gov/system/files/documents/pml/div683/museum-timeline.pdf
C = 2.99792458E5 # speed of light [km/s]
GG = 4.30104733e-06


def calc_schwarzschild_radius(M: float) -> float:
    """
    Input:
        M, mass [solar mass] of black hole
    Output:
        Schwarzschild radius [kpc]
    """

    return 2*GG*M/(C**2)

def find_binding_x_section(R_s: float, v: float) -> float:
    """
    Based on Mouri and Taniguchi 2002 
    https://arxiv.org/pdf/astro-ph/0201102 

    Inputs:
        R_s, average Schwarzchild radius [kpc] of black holes
        v, relative velocities [km/s] of black holes
    Output:
        sigma_sqrd^0.5, the binding cross-section [kpc]
    """

    v = np.linalg.norm(v) # make scalar

    sigma_sqrd = np.pi*(85*np.pi/3)**(2/7)*(R_s)**2*(v / C)**(-18/7)
    return np.sqrt(sigma_sqrd) # Derived from Equation (4)

def r_cross(x_section: float) -> float:
    # ask Simeon if it's cross-section squared or not
    return np.sqrt(np.sqrt(x_section)/np.pi)

def find_spacing(BHs: list[BlackHole]) -> float:
    """
    Find the average initial spacing [kpc] of a list of black holes in a given node.

    Input:
        BHs, list of black holes in a node
    Output:
        Average spacing between black holes in a node
    """

    N = len(BHs)
    
    spacing = 0.0
    # for possibility that a node has 1 black hole
    if N == 1:
        return spacing
    else:
        positions = np.array([bh.position for bh in BHs]) # get all black hole positions
        for i in range(0, len(positions)-1):
            for j in range(i+1, len(positions)):
                spacing += np.linalg.norm(positions[i] - positions[j])

        return spacing / N

def check_x_section_threshold(R_s: float, v: float, spacing: float) -> bool:
    """
    Inputs:
        R_s, average Schrwarzschild radius [kpc] of black holes
        v, magnitude of relative velocities [km/s] of black holes
        spacing, initial average spacing [kpc] of black holes in a node
    Output:
        True if spacing is greater than binding cross-section, False otherwise
    """

    # Ensures that our cross-section isn't too big, otherwise everything merges
    return find_binding_x_section(R_s, v) < ((spacing) / 20.0)**2

def merge(BH_1: BlackHole, BH_2: BlackHole) -> BlackHole:
    """
    Inputs:
        BH_1, first black hole to merge
        BH_2, second black hole to merge
    Output:
        merged_BH, new black hole after the merging of BH_1 and BH_2
    """

    new_mass = BH_1.mass + BH_2.mass # sum of masses
    new_pos = (BH_1.mass*BH_1.position + BH_2.mass*BH_2.position) / new_mass # com
    new_vel = (BH_1.mass*BH_1.velocity + BH_2.mass*BH_2.velocity) / new_mass # conservation of momentum
    merged_BH = BlackHole(new_mass, new_pos, new_vel, [1e-10, 1e-10, 1e-10])
    
    # return new merged black hole 
    return merged_BH

def find_10_mergers_time(mergers: list[BlackHole]) -> float:
    """
    Input:
        mergers, list of black holes that have been merged
    Output:
        Time [min] until 10 black holes have merged
    """
    # think this is wrong if the function gets called multiple times
    start = time.time() # WIP
    if len(mergers) == 10:
        end = time.time()
        return (end - start)/60
    
def comp_distance(bh: BlackHole, node: Node, R_s: float, v: float) -> bool:
    """
    Compare distance to a node and cross-section of a black hole. We only consider a node where a
    cross-section overlaps a node

    Inputs:
        bh, black hole to be merged
        node, node
        R_s, Schwarszchild radius [kpc] of black hole bh
        v, velocity [km/s] of black hole bh
    """

    length = 2*node.radius*np.sqrt(3) # node with maximum uncertainty 
    threshold_dist = np.linalg.norm(bh.position - node.center)**2 # this is a difference between two vectors, which is why we use norm
    # position of black hole - position of node center

    sigma = find_binding_x_section(R_s, v) 
    length_to_bh = length + sigma
    
    return  threshold_dist < length_to_bh

# def old_find_BH_candidates(bh: BlackHole, tree: Node, R_s: float, bh_vel: float) -> list[BlackHole]:
#     merged_bhs = []
#     for child in tree.children: 
#         BHs = child.enclosed_blackholes
#         if len(child.enclosed_blackholes) > 1: # more than one black hole within a child
#             spacing = find_spacing(BHs) # spacing of all BHs in a child
#             if check_x_section_threshold(R_s, bh_vel, spacing):                 
#                 # this is never hit b/c of check in line 104
#                 if len(child.enclosed_blackholes) == 1: # merge current black hole with single black hole in node
#                     merged_bhs.append(BHs[0])
#                 else:
#                     merger_list = find_merger(BHs, child)   
#                     for bh in merger_list:
#                         merged_bhs.append(bh) 
#             else: # node n fails threshold criteria
#                 continue
#         elif len(BHs) == 1: # single black hole in node
#             merged_bhs.append(BHs[0])
#             continue
#         else: # no black holes to merge!
#             continue
    
#     return merged_bhs

# def find_merger_new(bh: BlackHole, tree: Node, radius: float) -> BlackHole:
#     """
#     Find a black hole to merge depending on certain criteria. We do this by first checking
#     the cross-section threshold of the top nodes in tree. If this threshold is not met, return
#     bh, the unmerged black hole. If the threshold is met, we know to descend that specific node.
#     If the child of the node is empty (is this possible?), return the unmerged black hole bh. 
#     If the child has a single black hole, we return the merged black holes. If the child contains
#     multiple black holes, we use recursion on the child node and repeat the above process.

#     Inputs:
#         bh, black hole that will (possibly) be merged
#         tree, tree of black holes
#         radius, radius of node
#     Output:
#         A (possibly) merged black hole
#     """

#     R_s = calc_schwarzschild_radius(bh.mass)
#     bh_vel = bh.velocity
#     for child in tree.children: 
#         BHs = [other for other in child.enclosed_blackholes if other is not bh] # get all BHs in a child other than itself
        
#         if len(BHs) == 0: # there are no BHs in the candidate child, so nothing to merge
#             continue        
        
#         if comp_distance(bh, child, R_s, bh_vel): 
#             # now we descend child b/c it passes threshold
#             if len(BHs) == 1: # there is only a single BH in the candidate child, so we can merge
#                 merged = merge(bh, BHs[0])
#                 return merged, [bh, BHs[0]]
#             else:
#                 find_bh, consumed = find_merger_new(bh, child, child.radius) # there are multiple BHs in a child, 
#                                                                              # so we need to look at the children 
#                                                                              # of the child (since multiple BHs will 
#                                                                              # live in multiple different children)
#                 return find_bh, consumed
    
#     # no child passed threshold, so nothing to merge
#     return bh, [bh]

def find_merger_new(bh: BlackHole, tree: Node, radius: float):
    """
    Finds and performs black hole mergers within the given tree node.

    Returns
    -------
    merged_bh : BlackHole
        Either the original bh (no merge) or a new merged BH.
    consumed : list[BlackHole]
        All original BHs consumed to produce merged_bh.
    """

    R_s = calc_schwarzschild_radius(bh.mass)
    bh_vel = bh.velocity

    # At minimum, bh consumes itself
    consumed_total = [bh]

    for child in tree.children:

        # All possible partners except ourselves
        BHs = [other for other in child.enclosed_blackholes if other is not bh]
        if len(BHs) == 0:
            continue

        # Check the cross-section / distance threshold
        if not comp_distance(bh, child, R_s, bh_vel):
            continue

        # --- CASE 1: one BH in this node → simple merge ---
        if len(BHs) == 1:
            merged = merge(bh, BHs[0])
            return merged, [bh, BHs[0]]

        # --- CASE 2: multiple BHs → recurse into this child node ---
        merged_bh, consumed_sub = find_merger_new(bh, child, child.radius)

        # If recursion found a merge, return everything consumed
        if merged_bh is not bh:
            return merged_bh, consumed_total + consumed_sub

        # If recursion did NOT merge anything, keep searching siblings

    # No merges found anywhere in subtree
    return bh, consumed_total

# def find_merger(BHs: list[BlackHole], root: Node) -> BlackHole:
#     """
#     """

#     merged_candidates = []
#     for bh in BHs:
#         R_s = calc_schwarzschild_radius(bh.mass)
#         bh_vel = bh.velocity
#         merged_candidates.append(find_BH_candidates(bh, root, R_s, bh_vel))

#     if len(merged_candidates) == 0: # if there are no candidates to merge
#         return 
#     else:
#         return merge(BHs[0], merged_candidates[0]) # don't think this is right
    

# def merge(BH_1: BlackHole, BH_2: BlackHole) -> BlackHole:
#     """
#     Merge two black holes and return a NEW BlackHole object.
#     Defensive: ensures new object, uses center-of-mass position/velocity.
#     """
#     # Defensive copy of inputs (do NOT mutate inputs)
#     m1, m2 = float(BH_1.mass), float(BH_2.mass)
#     if not np.isfinite(m1) or not np.isfinite(m2):
#         raise ValueError("merge: non-finite mass encountered")

#     new_mass = m1 + m2
#     # compute center of mass position and velocity (works for numpy arrays)
#     new_pos = (m1 * np.array(BH_1.position) + m2 * np.array(BH_2.position)) / new_mass
#     new_vel = (m1 * np.array(BH_1.velocity) + m2 * np.array(BH_2.velocity)) / new_mass

#     # create a fresh BlackHole object (do not reuse BH_1 or BH_2)
#     merged_BH = BlackHole(new_mass, new_pos, new_vel, [1e-10, 1e-10, 1e-10])

#     # final sanity checks
#     if not np.all(np.isfinite(merged_BH.position)) or not np.all(np.isfinite(merged_BH.velocity)):
#         raise ValueError("merge produced non-finite position/velocity")

#     return merged_BH


def _uniq_preserve_order(seq):
    """Return list of unique items preserving order (by id for objects)."""
    seen = set()
    out = []
    for x in seq:
        ix = id(x)
        if ix not in seen:
            seen.add(ix)
            out.append(x)
    return out


# def find_merger_new(bh: BlackHole, tree: Node, radius: float):
#     """
#     Robust recursive merger finder.

#     Returns:
#       merged_bh: either the original `bh` (if no merge) or a NEW merged BlackHole
#       consumed_total: list of original BH objects that were consumed to produce merged_bh
#     """
#     R_s = calc_schwarzschild_radius(bh.mass)
#     bh_vel = np.array(bh.velocity)

#     # start with the BH itself in the consumed list
#     consumed_total = [bh]

#     # iterate children nodes
#     for child in tree.children:
#         # other BHs in this child (exclude `bh` itself)
#         BHs = [other for other in child.enclosed_blackholes if other is not bh]
#         if len(BHs) == 0:
#             continue

#         # quick distance / cross-section test
#         try:
#             if not comp_distance(bh, child, R_s, bh_vel):
#                 continue
#         except Exception:
#             # defensive: if something odd in comp_distance, skip this child
#             continue

#         # CASE 1: single partner -> merge directly
#         if len(BHs) == 1:
#             partner = BHs[0]
#             # ensure partner is a distinct object
#             if id(partner) == id(bh):
#                 continue
#             merged = merge(bh, partner)
#             consumed = [bh, partner]
#             consumed = _uniq_preserve_order(consumed)
#             return merged, consumed

#         # CASE 2: multiple BHs -> descend recursively
#         # We attempt to find a merge deeper inside this child; if found, propagate consumed list.
#         merged_bh, consumed_sub = find_merger_new(bh, child, child.radius)

#         # sanitize consumed_sub
#         consumed_sub = _uniq_preserve_order(consumed_sub)

#         if merged_bh is not bh:
#             # recursion produced a genuine merge
#             # combine consumed lists, removing duplicates
#             combined = consumed_total + consumed_sub
#             combined = _uniq_preserve_order(combined)
#             return merged_bh, combined

#         # otherwise no merge found in this child's subtree; continue to next sibling

#     # No merges anywhere in subtree -> return original bh and consumed_total
#     consumed_total = _uniq_preserve_order(consumed_total)
#     return bh, consumed_total
