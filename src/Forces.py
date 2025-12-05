#N-Body Gravitational Forces Calculations

import numpy as np
from .BlackHoles_Struct import BlackHole
# from .gravitree import Node, build_tree
from .merge_gravitree import Node, build_tree

# Gravitational constant (magnitude only, no units)
GG_new = 4.301047329348498e-06  # kpc * km^2 / (M_sun * s^2)


def _comp_acceleration(BH_i : BlackHole, BH_j : BlackHole):
    """ 
    Acceleration Function: Compute acceleration on BH_i due to BH_j
    -BH_i = blackhole on which the force is exerted
    -BH_j = blackhole exerting grav force 
    In this function we...
    - Calulate the position vector 
    - Calculate the vector magnitude --> np.linalg.norm() function calculates the magnitude vector (Euclidean norm)
    - Calculate the acceleration using our gravitational constant and converts to km/s**2
    Returns np.ndarray with acceleration vector for BH_i in km/s**2
    """

    pos_vec = BH_i.displacement(BH_j) 
    mag_vec = np.linalg.norm(pos_vec) 
    assert mag_vec != 0, "BHs cannot be at the same position - division by zero" #Checks that the magnitude vector is not zero (i.e. same BHs)
    accel = GG_new * BH_j.mass * (pos_vec)/(mag_vec)**3 

    accel *= 3.24078e-17  # conversion factor of km/kpc to result in an accel in km/s^2
    # accel = accel.to("km/s**2")  # converts acceleration to km/s^2

    return accel           

def _comp_jerk(BH_i : BlackHole, BH_j : BlackHole):
    """
    Computes the jerk value between two blackholes using the analytical expression for jerk.
    j_ij = G * m_j [(v_ij/ r_ij **3) - 3(r_ij dot v_ij)*r_ij/ r**5]
    Returns jerk in units units of km/s^3 
    """
    r_ij = BH_i.displacement(BH_j)         # Calls dposition vector function function from BlackHoles_Struct.py
    v_ij = (BH_j.velocity - BH_i.velocity)    # Difference in velocities (make sure that it is j minus i for calculation on i)
    r = np.linalg.norm(r_ij) #Magnitude 
    assert r != 0, "Black holes cannot be the same" #If the BHs are the same  
    rdotv = np.dot(r_ij, v_ij) #Dot product of position and velocity vectors
    jerk = GG_new * BH_j.mass * (v_ij / r**3 - 3 * rdotv * r_ij / r**5) 

    jerk *= 1.0502655e-33   # conversion factor of (km/kpc)^2 to result in a snap in km/s^3
    # return jerk.to("km/s^3")
    
    return jerk 

#Function to compute snap from jerk
def _comp_snap(BH_i: BlackHole, BH_j: BlackHole, previous_accel_i : list[float]):
    """
    Computes the snap value between two blackholes using the analytical expression for snap.
    s_ij= G * m_j [(a_ij/r_ij**3)- (6 * v_ij*(r_ij dot v_ij)/ r_ij**5) - (3 * r_ij * (v_ij**2 + r_ij dot a_ij)/ r_ij **5) + (15 * (r_i*(r_ij dot v_ij)**2)/ r_ij**7)]
    Returns snap in units km/s^4

    Note whether or not BH_j.accel is the new or old accel depends on if BH_j has had its accel computed before or after BH_i. I don't think this will cause issues, but something to be aware of.
    
    Parameters
    ----------
    BH_i : Blackhole
        blackhole whose snap is being computed
    BH_j : Blackhole
        source of the snap
    previous_accel_i : list[float]
        acceleration of BH_i at last timestep
    
    Returns
    -------
    list[float]
        snap of BH_i due to BH_j
    """

    r_ij = BH_i.displacement(BH_j)   #Calls position vector function from BlackHoles_Struct.py
    v_ij = (BH_j.velocity - BH_i.velocity)  #Difference in velocities 
    a_ij = BH_j.acceleration - previous_accel_i #Difference in accelerations

    r = np.linalg.norm(r_ij) #Magnitude 
    assert r != 0, "Black holes cannot be the same" #If the BHs are the same  

    rdotv = np.dot(r_ij, v_ij)#Dot product of position and velocity vectors
    v2 = np.dot(v_ij, v_ij) 
    rdota = np.dot(r_ij, a_ij)#Dot product of position and acceleration vectors

    km_by_kpc = 3.24078e-17
    kpc_by_km = 3.0856775814671916e16

    # different terms have different units in the expression below
    # hence two of them need to be multiplied by a factor of kpc_by_km
    snap = GG_new * BH_j.mass * (a_ij / r**3 * kpc_by_km
                                              - 6 * rdotv * v_ij / r**5  
                                              - 3 * (v2 + rdota * kpc_by_km) * r_ij / r**5  
                                              + 15 * (rdotv**2) * r_ij / r**7 )
    snap *= km_by_kpc**3
    return snap 

def recalculate_dynamics(BHs: list[BlackHole], use_tree : bool, use_dynamic_criterion : bool, ALPHA : float, THETA_0 : float):
    """
    Function for looping over black holes 
    Recalculates acceleration jerk and snap
    
    Parameters
    ----------
    BHs : list[BlackHole]
        the blackholes to have their dynamical values
    use_tree : bool, default False
        whether or not to use the tree for force calculation
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion
    """
    if use_tree:
        root, ignore_radius = build_tree(BHs)

    for BH_i in BHs: 
        #Reset acceleration before summing 
        previous_accel_i = BH_i.acceleration
        BH_i.acceleration = np.zeros(3)
        BH_i.jerk = np.zeros(3)
        BH_i.snap = np.zeros(3) 

        #Tree calculation
        if use_tree:
            _calculate_accel_with_tree(BH_i, root, previous_accel_i, use_dynamic_criterion, ALPHA, THETA_0)
            continue

        #brute force calculation
        for BH_j in BHs: 
            if BH_i == BH_j: 
                continue
            
            BH_i.acceleration += _comp_acceleration(BH_i, BH_j) 
            BH_i.jerk += _comp_jerk(BH_i, BH_j)
            BH_i.snap += _comp_snap(BH_i, BH_j, previous_accel_i)

def _calculate_accel_with_tree(bh : BlackHole, root : Node, previous_accel : list[float], use_dynamic_criterion : bool, ALPHA : float, THETA_0 : float):
    """
    Recursively calculates acceleration with tree

    Parameters
    ----------
    bh : BlackHole
        balckhole whose acceleration is being computed
    root : Node
        root node of tree from which accleration is being computed
    previous_accel : list[float]
        acceleration of bh at previous timestep
    use_dynamic_criterion : bool
        determines if dynamic or geometric criterion is used for node approximation
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion
    """
    #ignore empty nodes
    if len(root.enclosed_blackholes) == 0:
        return
    
    #ignore nodes that only contain self
    if len(root.enclosed_blackholes) == 1 and root.enclosed_blackholes[0] == bh:
        return
    
    #immediately compute nodes with only 1 BH
    if len(root.enclosed_blackholes) == 1:
        bh.acceleration += _comp_acceleration(bh, root)
        bh.jerk += _comp_jerk(bh, root)
        bh.snap += _comp_snap(bh, root, previous_accel)
        return
    
    #compute nodes which meet approximation criterion
    if node_is_approximatable(bh, root, np.linalg.norm(previous_accel), use_dynamic_criterion, ALPHA, THETA_0) and not root.is_inside(bh.position):
        bh.acceleration += _comp_acceleration(bh, root)
        bh.jerk += _comp_jerk(bh, root)
        bh.snap += _comp_snap(bh, root, previous_accel)
        return
    
    #descend tree further as node did not meet any end criterion
    for child in root.children:
        _calculate_accel_with_tree(bh, child, previous_accel, use_dynamic_criterion, ALPHA, THETA_0)
    
def node_is_approximatable(bh : BlackHole, node : Node, previous_accel : float, use_dynamic_criterion : bool, ALPHA : float, THETA_0 : float):
    """
    Determines if Node is safe to approximate or if tree needs to be further explored

    Parameters
    ----------
    bh : Blackhole
        the blackhole who's accel is being calculated
    node : Node
        the node whos approximation status is in question
    previous_accel : float
        magnitude of acceleration at previous timestep, only needed for dynamic criterion
    use_dynamic_criterion : bool
        determines if geometric criterion (default) or dynamic criterion should be used
    ALPHA : float
        accuracy parameter for dynamic criterion
    THETA_0 : float
        accuracy parameter for geometric criterion

    Returns
    bool
        if the node is sufficiently far to be approximated as a point mass
    """
    #dynamic criterion
    if use_dynamic_criterion:
        d = np.linalg.norm(bh.displacement(node))  # in kpc
        #GG_new is in kpc * km^2 / (M_sun * s^2)
        #This returns a dimensionless comparison
        return GG_new * node.mass * (node.crossection)**2 / (d**4) <= ALPHA * 1e-3 * previous_accel
    
    #geometric criterion
    return (node.crossection / np.linalg.norm(bh.displacement(node))) < THETA_0
