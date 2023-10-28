# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:16:02 2023

@author: Beom Chan Ryu

COSMO-SAC model.
"""
import numpy as np
from scipy.spatial import distance_matrix

# %% Global parameters
# atom radius [Å]
_rc = {"H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
       "C": 0.76,  # For sp3; sp2: 0.73, sp1: 0.69
       "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66,
       "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
       "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70,
       "Ti": 1.60, "V": 1.53, "Cr": 1.39,
       "Mn": 1.39,  # For l.s.; h.s.: 1.61
       "Fe": 1.32,  # For l.s.; h.s.: 1.52
       "Co": 1.26,  # For l.s.; h.s.: 1.50
       "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
       "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20,
       "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
       "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45,
       "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
       "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "La": 2.07,
       "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
       "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92,
       "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
       "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
       "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46,
       "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50, "Fr": 2.60,
       "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96,
       "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69}

_q0 = 79.53  # Area normalization parameter [Å**2]
_r0 = 66.69  # Volume normalization parameter [Å**3]
_z = 10  # Coordination number [ø]
_sighb = 0.0084  # H-bonding screening charge for psigA [e/Å**2]
_R = 1.987204258e-3  # Gas constant [kcal/K/mol]
_fdecay = 1/0.52928/0.52928  # Unit conversion parameter [ø]
_sig0 = 0.007  # H-bonding screening charge for W [e/Å**2]
_AES = 6525.69  # Electrostatic constant A [kcal*Å**4/mol/e**2]
_BES = 1.4859e8  # Electrostatic constant B [kcal*Å**4*K**2/mol/e**2]

# version-dependent global parameters
_aeff = {  # effective area [Å**2]
    2002: 7.5,
    2010: 7.25,
    2013: 7.25,
    2019: 7.25
}
_reff = {  # effective radius [Å]
    2002: np.sqrt(7.5/np.pi),  # see _aeff for 7.5
    2010: np.sqrt(7.25/np.pi),  # see _aeff for 7.25
    2013: np.sqrt(7.25/np.pi),
    2019: np.sqrt(7.25/np.pi)
}
_num_sp = {2002: 1, 2010: 3, 2013: 3, 2019: 4}  # number of sigma profiles [ø]
_chb = {  # hydrogen bonding parameter [kcal*Å**4/mol/e**2]
    2002: np.array([[85580]]),
    2010: np.array([[0,       0,       0],  # row and col: NHB, OH, OT
                    [0, 4013.78, 3016.43],
                    [0, 3016.43,  932.31]]),
    2013: np.array([[0,       0,       0],
                    [0, 4013.78, 3016.43],
                    [0, 3016.43,  932.31]]),
    2019: np.array([[0,       0,       0,       0],  # NHB, OH, OT, COOH
                    [0, 4013.78, 3016.43, 3020.18],
                    [0, 3016.43,  932.31, 1872.84],
                    [0, 3020.18, 1872.84, 2225.67]])
}
_cES = {  # electrostatic parameter [kcal*Å**4/mol/e**2]
    2002: lambda T: 8233.36,
    2010: lambda T: _AES + _BES/T/T,
    2013: lambda T: _AES + _BES/T/T,
    2019: lambda T: _AES + _BES/T/T
}


# %% Functions
def _is_in_version(version):
    """Check the version is valid.

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.

    Returns
    -------
    None.

    Raises
    ------
    ValueError : If the version is not one of the defined versions.
    """
    if version not in {2002, 2010, 2013, 2019}:
        raise ValueError(f"The COSMO-SAC version {version} is not supported.")


def _is_from_ms(file):
    """Find if the file is from the databases or Material Studio.

    The databases are VT (2006) and UC (2020).

    Parameters
    ----------
    file : str
        The name of the file.

    Returns
    -------
    bool
        True if the file is from Material Studio, False if it is from the
        databases.

    Raises
    ------
    ValueError
        If the file is not interpreted as cosmo file. It is determined by the
        first line of the file.
    """
    # Open file
    opened_file = open(file, "r")
    line = opened_file.readline()

    # Check the origin
    if "COSMO Results from DMol3" in line:  # Material Studio 2017
        return True

    elif "text" in line:  # VT database (2006), UD database (2020)
        return False

    else:
        raise ValueError(f"The file {file} is not interpreted as cosmo file.")


def _get_cosmo_from_ms(opened_file):
    """Get COSMO properties from the cosmo file in Material Studio.

    Parameters
    ----------
    opened_file : _io.TextIOWrapper
        Opened file.

    Returns
    -------
    A : float
        Cavity area.
    V : float
        Cavity volume.
    atom : numpy.ndarray of shape=(num_atom,)
        Atom symbols sorted by index in the cosmo file.
    coord : numpy.ndarray of shape=(num_atom, 3)
        The x, y, z coordinates of the atoms.
    seg : numpy.ndarray of shape=(num_seg, 6)
        The list of atom index, x, y, z position, segment area, and charge per
        segment area.
    """
    flag = "default"

    # COSMO data storage
    atom = []  # Atom symbols
    coord = []  # Atom coordinates
    seg = []  # Segment informations

    for line in opened_file:
        # Change flag
        if "$coordinates xyz [au]" in line:
            flag = "coordinate"
            continue

        if "n  atom        position (X, Y, Z) [au]" in line:
            flag = "segment"
            continue

        # Extract data
        if "Surface area of cavity" in line:  # Get cavity area
            line = line.split()
            A = float(line[6])  # [au**2]

        if "Total Volume of cavity" in line:  # Get cavity volume
            line = line.split()
            V = float(line[6])  # [au**3]

        if flag == "coordinate":  # Get atom informations
            if "$end" in line:
                flag = "default"

            else:
                line = line.split()
                atom.append(line[0])
                coord.append(list(map(float, line[1:4])))  # [au]

        if flag == "segment":  # Get segment informations
            line = line.split()
            if line:
                seg.append(
                    [int(line[1]) - 1] +
                    list(map(float, line[2:5] + line[6:8]))
                )  # [0], [au], [au], [au], [au**2], [e/au**2]

    # Change type from list to numpy.ndarray
    atom = np.array(atom)
    coord = np.array(coord)
    seg = np.array(seg)

    # Change unit from atomic unit (au) to angstrom (Å)
    ang_per_au = 0.52917721067  # angstrom per 1 atomic unit

    A = A*ang_per_au**2  # [Å**2]
    V = V*ang_per_au**3  # [Å**3]
    coord = coord*ang_per_au  # [Å]

    seg[:, 1:4] = seg[:, 1:4]*ang_per_au  # [Å], [Å], [Å]
    seg[:, 4] = seg[:, 4]*ang_per_au**2  # [Å**2]
    seg[:, 5] = seg[:, 5]/ang_per_au**2  # [e/Å**2]

    return A, V, atom, coord, seg


def _get_cosmo_from_not_ms(opened_file):
    """Get COSMO properties from the cosmo file in Material Studio.

    Parameters
    ----------
    opened_file : _io.TextIOWrapper
        Opened file.

    Returns
    -------
    A : float
        Cavity area.
    V : float
        Cavity volume.
    atom : numpy.ndarray of shape=(num_atom,)
        Atom symbols sorted by index in the cosmo file.
    coord : numpy.ndarray of shape=(num_atom, 3)
        The x, y, z coordinates of the atoms.
    seg : numpy.ndarray of shape=(num_seg, 6)
        The list of atom index, x, y, z position, segment area, and charge per
        segment area.
    """
    flag = "default"

    # COSMO data storage
    atom = []  # atom symbols
    coord = []  # atom coordinates
    seg = []  # segment informations

    for line in opened_file:
        # Change flag
        if "!DATE" in line:
            flag = "coordinate"
            continue

        if "n   atom        position (X, Y, Z) [au]" in line:
            flag = "segment"
            continue

        # Extract data
        if "Total surface area of cavity" in line:  # Get cavity area
            line = line.split()
            A = float(line[7])  # [Å**2]

        if "Total volume of cavity" in line:  # Get cavity volume
            line = line.split()
            V = float(line[6])  # [Å**3]

        if flag == "coordinate":  # Get atom informations
            if "end" in line:
                flag = "default"

            else:
                line = line.split()
                atom.append(line[7])
                coord.append(list(map(float, line[1:4])))  # [Å]

        if flag == "segment":  # Get segment informations
            line = line.split()
            if line:
                seg.append(
                    [int(line[1]) - 1] +
                    list(map(float, line[2:5] + line[6:8]))
                )
                # [0], [au], [au], [au], [Å**2], [e/Å**2]

    # Change type from list to numpy.ndarray
    atom = np.array(atom)
    coord = np.array(coord)
    seg = np.array(seg)

    # Change unit from atomic unit (au) to angstrom (Å)
    ang_per_au = 0.52917721067  # angstrom per 1 atomic unit

    seg[:, 1:4] = seg[:, 1:4]*ang_per_au  # [Å], [Å], [Å]

    return A, V, atom, coord, seg


def get_cosmo(file):
    """Get COSMO properties from the cosmo extension file.

    Parameters
    ----------
    file : str
        The name of the file.

    See Also
    --------
    _is_from_ms
        Function to check if the file is from databases or Material Studio.
    _get_cosmo_from_ms, _get_cosmo_from_not_ms
        Functions to get COSMO informations.
    """
    # Reading cosmo extension file
    opened_file = open(file, "r")
    is_ms = _is_from_ms(file)

    if is_ms:
        return _get_cosmo_from_ms(opened_file)

    else:
        return _get_cosmo_from_not_ms(opened_file)


def get_bond(atom, coord):
    """Get bond matrix.

    Parameters
    ----------
    atom : numpy.ndarray of shape=(num_atom,)
        Atom symbols sorted by index in the cosmo file.
    coord : numpy.ndarray of shape=(num_atom, 3)
        The x, y, z coordinates of the atoms.

    Returns
    -------
    bond : numpy.ndarray of shape=(num_atom, num_atom)
        The bond matrix. If two atoms are bonded, their entry is 1, else 0.
    """
    d_atom = distance_matrix(coord, coord)  # Distance between atoms
    rc = np.array([_rc[a] for a in atom])  # Radii of atoms

    mask = d_atom < 1.15*(rc[:, np.newaxis] + rc[np.newaxis, :])
    bond = np.where(mask, 1, 0)
    np.fill_diagonal(bond, 0)  # Atoms do not bond with themselves.

    return bond


def get_atom_type(version, atom, bond):
    """Get hybridization and sigma profile types for each atom.

    The dispersive natures are as below.
    DSP_WATER : WATER in this code. This indicates water.
    DSP_COOH : COOH in this code. This indicates a molecule with a carboxyl
    group.
    DSP_HB_ONLY_ACCEPTOR : HBOA in this code. The molecule contains any of the
    atoms O,N, or F but no H atoms bonded to any of these O, N, or F.
    DSP_HB_DONOR_ACCEPTOR : HBDA in this code. The molecule contains any of the
    functional groups NH, OH, or FH (but not OH of COOH or water).
    DSP_NHB : NHB in this code. This indicates that the molecule is non-
    hydrogen-bonding.

    The dispersion types are as below.
    C(sp3) : C bonded to 4 others.
    C(sp2) : C bonded to 3 others.
    C(sp) : C bonded to 2 others.
    N(sp3) : N bonded to three others.
    N(sp2) : N bonded to two others.
    N(sp) : N bonded to one other.
    -O- : O(sp3) in this code. O bonded to 2 others.
    =O : O(sp2) in this code. Double-bonded O.
    F : F bonded to one other.
    Cl : Cl bonded to one other.
    H(water) : H in water.
    H(OH) : H-O bond but not water.
    H(NH) : H bonded to N.
    H(other) : H otherwise.
    other : Undifined.

    The hydrogen-bonding types are as below.
    OH : if the atom is O and is bonded to an H, or vice versa.
    OT : if the atom is O and is bonded to an atom other than H, or if the atom
    is H and is bonded to N or F.
    COOH : if the atoms are C, O, H and are in the carboxyl group.
    NHB : otherwise.

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.
    atom : numpy.ndarray of shape=(num_atom,)
        Atom symbols sorted by index in the cosmo file.
    bond : numpy.ndarray of shape=(num_atom, num_atom)
        The bond matrix. If two atoms are bonded, their entry is 1, else 0.

    Returns
    -------
    dtype : list of shape=(num_atom,)
        The dispersion type for each atom.
    stype : list of shape=(num_atom,)
        The hydrogen-bonding type for each atom.
    dnatr : {"NHB", "HBOA", "HBDA", "WATER", "COOH"}
        The dispersive nature of the molecule.
    """
    dtype = ["other"]*len(atom)  # hybridization type
    stype = ["NHB"]*len(atom)  # sigma profile type
    dnatr = "NHB"  # dispersive nature of molecule
    dntype = set()  # dispersive nature type of atoms

    # no types for COSMO-SAC 2002
    if version == 2002:
        return dtype, stype, dnatr

    # {atom type: {bonded atoms: (dtype, stype, dntype), ...}, ...}
    # This assumes that all atoms are belong to NHB, OT and H(other).
    atom_prop = {
        "C": {
            2: ("C(sp)", "NHB", "NHB"),
            3: ("C(sp2)", "NHB", "NHB"),
            4: ("C(sp3)", "NHB", "NHB")
        },
        "O": {
            1: ("O(sp2)", "OT", "HBOA"),
            2: ("O(sp3)", "OT", "HBOA"),
        },
        "N": {
            1: ("N(sp)", "OT", "HBOA"),
            2: ("N(sp2)", "OT", "HBOA"),
            3: ("N(sp3)", "OT", "HBOA"),
        },
        "F": {1: ("F", "OT", "HBOA")},
        "Cl": {1: ("Cl", "NHB", "NHB")},
        "H": {1: ("H(other)", "NHB", "NHB")}
    }

    for i, atom_type in enumerate(atom):
        # Get dictionary of index and atom types that are bonded with atom i
        ard_i = {j: atom[j] for j in np.flatnonzero(bond[i])}

        # If the atom is in the difined properties
        if atom_type in atom_prop:
            # Get atom types, else get ("Undifined", 0)
            dtype[i], stype[i], dntype_i = atom_prop[atom_type].get(
                len(ard_i),
                ("other", "NHB", "NHB")
            )
            dntype.add(dntype_i)

        # Find H near N, and renew the types of H
        if atom_type == "H" and "N" in ard_i.values():
            dtype[i] = "H(NH)"
            stype[i] = "OT"
            dntype.add("HBDA")

        # Find H in HF, and renew the types of H
        if atom_type == "H" and "F" in ard_i.values():
            stype[i] = "OT"
            dntype.add("HBDA")

        # Find atom type for -OH, H2O, and COOH
        if atom_type == "H" and "O" in ard_i.values():
            # # Renew the typs of H and O in OH
            # Renew the types of H
            dtype[i] = "H(OH)"
            stype[i] = "OH"

            # Find the atom index of O in OH
            j = list(ard_i.keys())[0]
            ard_j = {k: atom[k] for k in np.flatnonzero(bond[j])}
            # Renew the types of O in -OH
            stype[j] = "OH"
            dntype.add("HBDA")

            # # Further find H-OH and CO-OH
            # if the O in -OH has not two bonds, stop searching
            if len(ard_j) != 2:
                break

            # Find the atom index of a neighber of O in -OH, but not H in -OH
            k = [k for k in ard_j.keys() if k != i][0]
            ard_k = {m: atom[m] for m in np.flatnonzero(bond[k])}

            # if the atom k is H, that is, if the molecule is water, renew the
            # dtype of the Hs in H2O and stop searching
            if atom[k] == "H":
                dtype[i] = "H(water)"
                dtype[k] = "H(water)"
                dntype.add("WATER")
                break

            # # Further find COOH
            # if the atom k is not the C in part of COOH, stop searching
            if not (atom[k] == "C" and len(ard_k) == 3 and
                    list(ard_k.values()).count("O") == 2):
                break

            # Find the O, neighber of C in -COH, but not in O in -COH
            m = [m for m in ard_k.keys() if (m != j and ard_k[m] == "O")][0]
            ard_m = {n: atom[n] for n in np.flatnonzero(bond[m])}

            # if the atom m is -O-, not =O, stop searching
            if len(ard_m) != 1:
                break

            # Renew i(H), j(O), k(C) and m(O) as the part of COOH
            dntype.add("COOH")
            if version == 2019:
                stype[i] = "COOH"
                stype[j] = "COOH"
                stype[m] = "COOH"

    # find the dispersive nature of the molecule
    if "HBOA" in dntype:
        dnatr = "HBOA"
    if "HBDA" in dntype:
        dnatr = "HBDA"
    if "WATER" in dntype:
        dnatr = "WATER"
    if "COOH" in dntype:
        dnatr = "COOH"

    return dtype, stype, dnatr


def get_sigma(version, atom, seg, stype):
    """Get sigma profiles.

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.
    atom : numpy.ndarray of shape=(num_atom,)
        Atom symbols sorted by index in the cosmo file.
    seg : numpy.ndarray of shape=(num_seg, 6)
        The list of atom index, x, y, z position, segment area, and charge per
        segment area.
    stype : list of shape=(num_atom,)
        The sigma profile type for each atom.

    Returns
    -------
    psigA : numpy.ndarray of shape=(num_sp, 51)
        The sigma profiles of the molecule. The number of sigma profiles is
        dependent on the version.
        {version: num_sp} = {2002: 1, 2010: 3, 2013: 3, 2019: 4}
    """
    # import global parameters
    reff = _reff[version]
    num_sp = _num_sp[version]

    # Set sigma profile types to integers
    type_mapping = {"NHB": 0, "OH": 1, "OT": 2, "COOH": 3}
    stype_int = np.array([type_mapping[element] for element in stype])

    # Define segment informations
    seg_atom_index = np.int32(seg[:, 0])
    seg_atom = atom[seg_atom_index]
    seg_stype = stype_int[seg_atom_index]
    seg_coord = seg[:, 1:4]
    seg_area = seg[:, 4]
    seg_charge = seg[:, 5]

    # Calculate radii of the segments and distances between the segments
    r = np.sqrt(seg_area/np.pi)
    d = distance_matrix(seg_coord, seg_coord)

    # Calculate averaged surface charges of the segments
    rcal = r**2*reff**2/(r**2 + reff**2)
    dcal = np.exp(-_fdecay*d**2/(r**2 + reff**2).reshape(-1, 1))

    upper = np.einsum("n,n,mn->m", seg_charge, rcal, dcal)
    lower = np.einsum("n,mn->m", rcal, dcal)

    seg_avg_charge = upper/lower

    # Decide sigma profile types
    sig_type = np.int32(np.zeros(len(seg)))  # Set all segments as NHB

    sig_type = np.where(  # Find OH sigma profile
        np.logical_and(seg_atom == "O", seg_stype == 1, seg_avg_charge > 0),
        1,
        sig_type
    )
    sig_type = np.where(
        np.logical_and(seg_atom == "H", seg_stype == 1, seg_avg_charge < 0),
        1,
        sig_type
    )

    sig_type = np.where(  # Find OT sigma profile
        np.logical_and(
            np.logical_or(seg_atom == "O", seg_atom == "N", seg_atom == "F"),
            seg_stype == 2,
            seg_avg_charge > 0
        ),
        2,
        sig_type
    )
    sig_type = np.where(
        np.logical_and(seg_atom == "H", seg_stype == 2, seg_avg_charge < 0),
        2,
        sig_type
    )

    sig_type = np.where(seg_stype == 3, 3, sig_type)  # Find COOH sigma profile

    # Calculate sigma profiles
    sig = np.linspace(-0.025, 0.025, 51)

    left = np.int32(np.floor((seg_avg_charge - sig[0])/0.001))
    w = (sig[left + 1] - seg_avg_charge)/0.001

    psigA = np.zeros((num_sp, 51))
    np.add.at(psigA, (sig_type, left), w*seg_area)
    np.add.at(psigA, (sig_type, left + 1), (1 - w)*seg_area)

    if version != 2002:
        phb = 1 - np.exp(-sig**2/2/_sig0**2)
        psigA[0] = psigA[0] + np.sum(psigA[1:], axis=0)*(1 - phb)
        psigA[1:] = psigA[1:]*phb

    return psigA


def get_dsp(version, dtype):
    """Get the dispersive nature of the molecule.

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.
    dtype : list of shape=(num_atom,)
        The dispersion type for each atom.

    Returns
    -------
    ek : float
        Dispersive parameter.
    """
    if version == 2002 or version == 2010 or ("other" in dtype):
        ek = None
        return ek

    # dispersive parameters
    ddict = {
        "C(sp3)": 115.7023,
        "C(sp2)": 117.4650,
        "C(sp)": 66.0691,
        "N(sp3)": 15.4901,
        "N(sp2)": 84.6268,
        "N(sp)": 109.6621,
        "O(sp3)": 95.6184,  # -O-
        "O(sp2)": -11.0549,  # =O
        "F": 52.9318,
        "Cl": 104.2534,
        "H(water)": 58.3301,
        "H(OH)": 19.3477,
        "H(NH)": 141.1709,
        "H(other)": 0
    }

    # calculate the dispersive parameter of the molecule
    ek = np.vectorize(ddict.get)(dtype)
    ek = np.sum(ek)/np.count_nonzero(ek)

    return ek


def cal_DW(version, T):
    """Calculate the exchange energy.

    The exchange energy has the values for each charge density combinations and
    sigma profile type combinations, therefore having the shape of (num_sp,
    num_sp, 51, 51).

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.
    T : float
        The system temperature.

    Returns
    -------
    DW : numpy.ndarray of shape=(num_sp, num_sp, 51, 51)
        The exchange energy.
    """
    # get global parameters
    chb = _chb[version]
    cES = _cES[version]
    num_sp = _num_sp[version]

    # calculate DW
    sig = np.linspace(-0.025, 0.025, 51)
    sigT = sig.reshape(-1, 1)

    DW = np.zeros((num_sp, num_sp, 51, 51))
    for i in range(num_sp):
        for j in range(i + 1):
            if version == 2002:
                acchb = np.maximum(sig, sigT) - _sighb
                donhb = np.minimum(sig, sigT) + _sighb

                maxacc = np.where(acchb > 0, acchb, 0)
                mindon = np.where(donhb < 0, donhb, 0)

                chb_part = -chb[i][j]*maxacc*mindon

            else:
                mask = (sig * sigT) < 0
                chb_part = np.where(mask, chb[i, j]*(sig - sigT)**2, 0)

            DW[i, j] = DW[j, i] = cES(T)*(sig + sigT)**2 - chb_part

    return DW


# %% calculating activity coefficients
def cal_ln_gam_comb(x, A, V):
    """Calculate log of combinatory activity coefficients.

    Parameters
    ----------
    x : numpy.ndarray of shape=(num_comp,)
        Mole fractions of components.
    A : numpy.ndarray of shape=(num_comp,)
        Cavity areas of components.
    V : numpy.ndarray of shape=(num_comp,)
        Cavity volumes of components.

    Returns
    -------
    ln_gam_comb : numpy.ndarray of shape=(num_comp,)
        Combinatory activity coefficients of components.
    """
    # calculate normalized areas and volumes
    q = A/_q0
    r = V/_r0
    L = (_z/2)*(r - q) - (r - 1)

    theta = q/np.sum(x*q)
    phi = r/np.sum(x*r)

    # calcualte combinatory activity coefficients
    ln_gam_comb = np.log(phi) + _z*q*np.log(theta/phi)/2 + L - phi*np.sum(x*L)
    return ln_gam_comb


def cal_psig_mix(x, A, psigA):
    """Calculate the mixture sigma profile of the mixture.

    Parameters
    ----------
    x : numpy.ndarray of shape=(num_comp,)
        The mole fractions of the components.
    A : numpy.ndarray of shape=(num_comp,)
        The cavity areas of the components.
    psigA : numpy.ndarray of shape=(num_comp, num_sp, 51)
        The sigma profiles of the components.

    Returns
    -------
    psig_mix : numpy.ndarray of shape=(num_sp, 51)
        The mixture sigma profiles.
    """
    psig_mix = np.einsum("i,itm->tm", x, psigA)/np.sum(x*A)
    return psig_mix


def cal_ln_gam_res(version, x, A, V, psigA, T, eps=1e-4, verbose=False):
    """Calculate residual activity coefficients.

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.
    x : numpy.ndarray of shape=(num_comp,)
        The mole fractions of the components.
    A : numpy.ndarray of shape=(num_comp,)
        The cavity areas of the components.
    V : numpy.ndarray of shape=(num_comp,)
        The cavity volumes of components.
    psigA : numpy.ndarray of shape=(num_comp, num_sp, 51)
        The sigma profiles of the components.
    T : float
        The system temperature.
    eps : float, optional
        The convergence criteria. The default is 1e-4.
    verbose : bool, optional
        If True, this function tells the failure of the convergence. The
        default is False.

    Returns
    -------
    ln_gam_res : numpy.ndarray of shape=(num_comp,)
        Residual activity coefficients of components.
    """
    # import version-dependent global parameters
    aeff = _aeff[version]

    # calculate intermediate terms
    psig = np.einsum("itm,i->itm", psigA, 1/A)
    psig_mix = cal_psig_mix(x, A, psigA)

    exp_DW = np.exp(-cal_DW(version, T)/_R/T)

    A_plus = np.einsum("stmn,isn->istmn", exp_DW, psig)  # A^(+)
    A_plus_mix = np.einsum("stmn,sn->stmn", exp_DW, psig_mix)  # A^(+)_mix

    # calculate the segment activity coefficients
    Gam = np.ones(np.shape(psig))
    Gam_mix = np.ones(np.shape(psig_mix))
    diff = 1

    for _ in range(500):
        Gam_old = Gam
        Gam_mix_old = Gam_mix

        Gam = 1 / np.einsum("istmn,isn->itm", A_plus, Gam)
        Gam_mix = 1 / np.einsum("stmn,sn->tm", A_plus_mix, Gam_mix)

        Gam = (1.618*Gam + Gam_old)/2.618
        Gam_mix = (1.618*Gam_mix + Gam_mix_old)/2.618

        # check convergence
        diff_Gam = np.max(np.abs((Gam - Gam_old)/Gam_old))
        diff_Gam_mix = np.max(np.abs((Gam_mix - Gam_mix_old)/Gam_mix_old))
        diff = np.max([diff_Gam, diff_Gam_mix])

        if diff <= eps:
            break

    else:
        if verbose:
            print("The convergence failed.")

    # calculate residual activity coefficients
    Gam_part = np.log(Gam_mix) - np.log(Gam)
    ln_gam_res = np.einsum("itm,itm->i", psigA, Gam_part)/aeff

    return ln_gam_res


def cal_ln_gam_dsp(x, ek, dnatr):
    """Calculate dispersive activity coefficients.

    Parameters
    ----------
    x : numpy.ndarray of shape=(num_comp,)
        The mole fractions of the components.
    ek : numpy.ndarray of shape=(num_comp,)
        The dispersive paramters of the components.
    dnatr : numpy.ndarray of shape=(num_comp,)
        The dispersive nature of the components.

    Returns
    -------
    ln_gam_dsp : numpy.ndarray of shape=(num_comp,)
        Dispersive activity coefficients of components.
    """
    num_mol = len(x)
    ekT = ek.reshape(-1, 1)

    if None in ek or None in dnatr:
        ln_gam_dsp = [None]*num_mol
        return ln_gam_dsp

    w = np.ones((num_mol, num_mol))*0.27027
    wpair = [
        {"WATER", "HBOA"},
        {"COOH", "NHB"},
        {"COOH", "HBDA"},
        {"WATER", "COOH"}
    ]
    for i in range(num_mol):
        for j in range(i):
            if {dnatr[i], dnatr[j]} in wpair:
                w[i][j] = w[j][i] = -0.27027

    A = w*(0.5*(ek + ekT) - np.sqrt(ek*ekT))  # not area

    ln_gam_dsp = np.zeros(num_mol)
    for i in range(num_mol):
        for j in range(num_mol):
            if i != j:
                ln_gam_dsp[i] = ln_gam_dsp[i] + x[j]*A[i, j]
            if j > i:
                ln_gam_dsp[i] = ln_gam_dsp[i] - x[i]*x[j]*A[i, j]

    return ln_gam_dsp


def cal_gam(version, x, T, A, V, psigA, ek, dnatr):
    """Calculate COSMO-SAC activity coefficients.

    Parameters
    ----------
    version : {2002, 2010, 2013, 2019}
        The COSMO-SAC version.
    x : numpy.ndarray of shape=(num_comp,)
        The mole fractions of the components.
    T : float
        The system temperature.
    A : numpy.ndarray of shape=(num_comp,)
        The cavity areas of the components.
    V : numpy.ndarray of shape=(num_comp,)
        The cavity volumes of components.
    psigA : numpy.ndarray of shape=(num_comp, num_sp, 51)
        The sigma profiles of the components.
    ek : numpy.ndarray of shape=(num_comp,)
        The dispersive paramters of the components.
    dnatr : numpy.ndarray of shape=(num_comp,)
        The dispersive nature of the components.

    Returns
    -------
    gam : numpy.ndarray of shape=(num_comp,)
        Activity coefficients of components.
    """
    # calculate log activity cofficients for each contribution
    ln_gam_comb = cal_ln_gam_comb(x, A, V)
    ln_gam_res = cal_ln_gam_res(version, x, A, V, psigA, T)
    ln_gam_dsp = cal_ln_gam_dsp(x, ek, dnatr)

    # check if dispersion activity coefficients are applicable
    if None in ln_gam_dsp:
        ln_gam = ln_gam_comb + ln_gam_res
    else:
        ln_gam = ln_gam_comb + ln_gam_res + ln_gam_dsp

    # calculate activity coefficients
    gam = np.exp(ln_gam)

    return gam


# %% Wrap-up functions
class COSMOMolecule:
    """A class of a molecule containing COSMO properties.

    Attributes
    ----------
    file : str
        The directory of the cosmo file.
    version : int, optional
        The COSMO-SAC version.
    name : str
        The component"s name.

    Methods
    -------
    None.
    """

    def __init__(self, file, version=None, name=None):
        self.file = file
        self.name = name

        # cosmo file 정보를 획득한다
        self.A, self.V, self.atom, self.coord, self.seg = get_cosmo(file)
        self.bond = get_bond(self.atom, self.coord)

        # version이 정해지면 COSMO 물성을 계산한다
        if version is not None:
            _is_in_version(version)
            self.version = version

    @property
    def version(self):  # getattr
        """Get COSMO-SAC version.

        Returns
        -------
        int
            The COSMO-SAC version.
        """
        return self.version

    @version.setter
    def version(self, version):  # setattr
        self.dtype, self.stype, self.dnatr = \
            get_atom_type(version, self.atom, self.bond)
        self.psigA = get_sigma(version, self.atom, self.seg, self.stype)
        self.ek = get_dsp(version, self.dtype)


class _COSMOMixture:
    """A class for combining the COSMO properties of several molecules.

    Attributes
    ----------
    COSMOMolecules : COSMOMolecule class
        The COSMO-calculated molecules with defined COSMO-SAC version.

    Methods
    -------
    None.
    """

    def __init__(self, *COSMOMolecules):
        for COSMOMolecule in COSMOMolecules:
            # 각 인스턴스의 변수를 합친다.
            for var_name, var_value in vars(COSMOMolecule).items():
                if not hasattr(self, var_name):
                    setattr(self, var_name, [var_value])
                else:
                    getattr(self, var_name).append(var_value)

        # 계산에 필요한 변수는 numpy.ndarray로 바꾼다.
        self.A = np.array(self.A)
        self.V = np.array(self.V)
        self.ek = np.array(self.ek)
        self.psigA = np.array(self.psigA)


class COSMOSAC:
    """A class for calculating COSMO-SAC activity coefficients.

    Attributes
    ----------
    COSMOMolecules : COSMOMolecule class
        The COSMO-calculated molecules in which the COSMO-SAC version is not
        defined.
    version : {2002, 2010, 2013, 2019}, optional
        The COSMO-SAC version. The default is 2019.

    Methods
    -------
    gam(x, T)
        Calculate the activity coefficients of the mixture.
    """

    def __init__(self, *COSMOMolecules, version=2019):
        # version의 유효성을 체크한다.
        _is_in_version(version)
        # COSMO-SAC version을 갱신해서 분자의 COSMO 물성을 계산한다.
        for COSMOMolecule in COSMOMolecules:
            COSMOMolecule.version = version

        self.version = version
        self.molecules = COSMOMolecules
        self._mixture = _COSMOMixture(*COSMOMolecules)

    def gam(self, x, T):
        """Calculate COSMO-SAC activity coefficients.

        Parameters
        ----------
        x : numpy.ndarray of shape=(num_comp,)
            The mole fractions of the components.
        T : float
            The system temperature.

        Returns
        -------
        gam : numpy.ndarray of shape=(num_comp,)
            Activity coefficients of components.

        See Also
        --------
        cal_gam : Main code for calculating COSMO-SAC activity coefficients
        """
        return cal_gam(
            self.version,
            x,
            T,
            self._mixture.A,
            self._mixture.V,
            self._mixture.psigA,
            self._mixture.ek,
            self._mixture.dnatr
        )


if __name__ == "__main__":
    # example
    # define the molecules
    mol1 = COSMOMolecule("KU1.cosmo", name="mol 1")
    mol2 = COSMOMolecule("KU2.cosmo", name="mol 2")
    mol2_with_version = COSMOMolecule("KU2.cosmo", version=2019, name="mol 2v")
    # mol2.psigA will raise error, because the version is not defined.
    # AttributeError: 'COSMOMolecule' object has no attribute 'psigA'
    # However, mol2_with_version.psigA gives the sigma profiles normally.

    # define COSMO-SAC model
    cosmo = COSMOSAC(mol1, mol2, version=2019)
    # Now mol2.psigA gives the sigma profiles.

    # calculate activity coefficients
    gam1 = cosmo.gam(np.array([0.0, 1.0]), 298.15)
    gam2 = cosmo.gam(np.array([0.5, 0.5]), 298.15)
    gam3 = cosmo.gam(np.array([1.0, 0.0]), 298.15)
