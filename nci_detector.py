"""
Non-Covalent Interaction (NCI) Detector for Known Protein-Ligand Complexes

This module detects various types of non-covalent interactions between
protein and ligand based on 3D coordinates from existing complex structures.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from collections import defaultdict


# Amino acid properties
AMINO_ACID_PROPERTIES = {
    'ALA': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
    'VAL': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
    'LEU': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
    'ILE': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
    'MET': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
    'PHE': {'type': 'hydrophobic', 'polar': False, 'aromatic': True},
    'TRP': {'type': 'hydrophobic', 'polar': False, 'aromatic': True},
    'PRO': {'type': 'hydrophobic', 'polar': False, 'aromatic': False},
    'SER': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
    'THR': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
    'CYS': {'type': 'polar', 'polar': True, 'hb_donor': True},
    'TYR': {'type': 'polar', 'polar': True, 'aromatic': True, 'hb_donor': True, 'hb_acceptor': True},
    'ASN': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
    'GLN': {'type': 'polar', 'polar': True, 'hb_donor': True, 'hb_acceptor': True},
    'LYS': {'type': 'charged', 'charge': '+', 'hb_donor': True},
    'ARG': {'type': 'charged', 'charge': '+', 'hb_donor': True},
    'HIS': {'type': 'charged', 'charge': '+', 'aromatic': True, 'hb_donor': True},
    'ASP': {'type': 'charged', 'charge': '-', 'hb_acceptor': True},
    'GLU': {'type': 'charged', 'charge': '-', 'hb_acceptor': True},
    'GLY': {'type': 'special', 'polar': False},
}


class ProteinStructure:
    """Protein structure parser for PDB files."""
    
    def __init__(self, pdb_file: str):
        self.pdb_file = pdb_file
        self.atoms = []
        self.residues = {}
        self._parse_pdb()
    
    def _parse_pdb(self):
        """Parse PDB file and extract atom information."""
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                
                try:
                    atom_info = {
                        'serial': int(line[6:11].strip()),
                        'name': line[12:16].strip(),
                        'res_name': line[17:20].strip(),
                        'chain': line[21].strip() or 'A',
                        'res_num': int(line[22:26].strip()),
                        'x': float(line[30:38].strip()),
                        'y': float(line[38:46].strip()),
                        'z': float(line[46:54].strip()),
                        'element': line[76:78].strip() or line[12:14].strip()[0]
                    }
                    
                    self.atoms.append(atom_info)
                    
                    # Group by residue
                    res_key = (atom_info['chain'], atom_info['res_num'], atom_info['res_name'])
                    if res_key not in self.residues:
                        self.residues[res_key] = []
                    self.residues[res_key].append(atom_info)
                    
                except (ValueError, IndexError):
                    continue


class NCIDetector:
    """
    Detector for non-covalent interactions between protein and ligand.
    Designed for analyzing existing protein-ligand complex structures.
    """
    
    # Distance thresholds (in Angstroms)
    HBOND_DISTANCE_MAX = 3.5
    PI_STACKING_DISTANCE_MAX = 5.0
    HYDROPHOBIC_DISTANCE_MAX = 4.5
    SALT_BRIDGE_DISTANCE_MAX = 4.5
    
    def __init__(self, protein_pdb: str):
        """
        Initialize NCI detector.
        
        Args:
            protein_pdb: Path to protein PDB file
        """
        self.protein = ProteinStructure(protein_pdb)
    
    def detect_all_interactions(
        self, 
        ligand_mol: Chem.Mol
    ) -> Dict[str, List[Dict]]:
        """
        Detect all types of interactions.
        
        Args:
            ligand_mol: RDKit molecule with 3D coordinates (from SDF file)
            
        Returns:
            Dictionary with interaction types as keys
        """
        if ligand_mol.GetNumConformers() == 0:
            raise ValueError("Ligand must have 3D coordinates. Load from SDF file with 3D coords.")
        
        interactions = {
            'hydrogen_bonds': self.detect_hydrogen_bonds(ligand_mol),
            'pi_stacking': self.detect_pi_stacking(ligand_mol),
            'hydrophobic': self.detect_hydrophobic_contacts(ligand_mol),
            'salt_bridges': self.detect_salt_bridges(ligand_mol)
        }
        
        return interactions
    
    def detect_hydrogen_bonds(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """
        Detect hydrogen bonds between protein and ligand.
        
        Criteria:
        - Donor-acceptor distance < 3.5Å
        - N-H...O, O-H...N, O-H...O patterns
        """
        hbonds = []
        ligand_conf = ligand_mol.GetConformer()
        
        # Get ligand donors and acceptors
        ligand_donors = self._get_hbond_donors(ligand_mol)
        ligand_acceptors = self._get_hbond_acceptors(ligand_mol)
        
        # Check each residue
        for (chain, res_num, res_name), res_atoms in self.protein.residues.items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            # Skip if residue cannot form H-bonds
            if not (res_props.get('hb_donor') or res_props.get('hb_acceptor')):
                continue
            
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
            res_id = f"{res_name}{res_num}:{chain}"
            
            # Ligand as donor, protein as acceptor
            if res_props.get('hb_acceptor'):
                for donor_idx in ligand_donors:
                    donor_pos = ligand_conf.GetAtomPosition(donor_idx)
                    donor_coord = np.array([donor_pos.x, donor_pos.y, donor_pos.z])
                    
                    # Find closest protein atom
                    distances = np.linalg.norm(res_coords - donor_coord, axis=1)
                    min_dist = np.min(distances)
                    
                    if min_dist < self.HBOND_DISTANCE_MAX:
                        hbonds.append({
                            'type': 'ligand_donor',
                            'ligand_atom': donor_idx,
                            'protein_residue': res_id,
                            'distance': float(min_dist),
                            'interaction': 'hydrogen_bond'
                        })
            
            # Ligand as acceptor, protein as donor
            if res_props.get('hb_donor'):
                for acceptor_idx in ligand_acceptors:
                    acceptor_pos = ligand_conf.GetAtomPosition(acceptor_idx)
                    acceptor_coord = np.array([acceptor_pos.x, acceptor_pos.y, acceptor_pos.z])
                    
                    distances = np.linalg.norm(res_coords - acceptor_coord, axis=1)
                    min_dist = np.min(distances)
                    
                    if min_dist < self.HBOND_DISTANCE_MAX:
                        hbonds.append({
                            'type': 'ligand_acceptor',
                            'ligand_atom': acceptor_idx,
                            'protein_residue': res_id,
                            'distance': float(min_dist),
                            'interaction': 'hydrogen_bond'
                        })
        
        return hbonds
    
    def detect_pi_stacking(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """
        Detect π-π stacking interactions.
        
        Criteria:
        - Distance between ring centers < 5.0Å
        """
        pi_stacks = []
        ligand_conf = ligand_mol.GetConformer()
        
        # Get aromatic rings in ligand
        ligand_rings = self._get_aromatic_rings(ligand_mol)
        
        # Check aromatic residues
        for (chain, res_num, res_name), res_atoms in self.protein.residues.items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            if not res_props.get('aromatic'):
                continue
            
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
            res_center = res_coords.mean(axis=0)
            res_id = f"{res_name}{res_num}:{chain}"
            
            # Compare with each ligand aromatic ring
            for ring_atoms in ligand_rings:
                ring_coords = np.array([
                    [ligand_conf.GetAtomPosition(i).x,
                     ligand_conf.GetAtomPosition(i).y,
                     ligand_conf.GetAtomPosition(i).z]
                    for i in ring_atoms
                ])
                ring_center = ring_coords.mean(axis=0)
                
                distance = np.linalg.norm(ring_center - res_center)
                
                if distance < self.PI_STACKING_DISTANCE_MAX:
                    pi_stacks.append({
                        'ligand_atoms': list(ring_atoms),
                        'protein_residue': res_id,
                        'distance': float(distance),
                        'interaction': 'pi_stacking'
                    })
        
        return pi_stacks
    
    def detect_hydrophobic_contacts(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """
        Detect hydrophobic contacts.
        
        Criteria:
        - Both atoms are hydrophobic (C not adjacent to polar groups)
        - Distance < 4.5Å
        """
        hydrophobic = []
        ligand_conf = ligand_mol.GetConformer()
        
        # Get hydrophobic atoms in ligand
        ligand_hydrophobic = self._get_hydrophobic_atoms(ligand_mol)
        
        # Check hydrophobic residues
        for (chain, res_num, res_name), res_atoms in self.protein.residues.items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            if res_props.get('type') != 'hydrophobic':
                continue
            
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
            res_id = f"{res_name}{res_num}:{chain}"
            
            # Check each hydrophobic ligand atom
            for lig_idx in ligand_hydrophobic:
                lig_pos = ligand_conf.GetAtomPosition(lig_idx)
                lig_coord = np.array([lig_pos.x, lig_pos.y, lig_pos.z])
                
                distances = np.linalg.norm(res_coords - lig_coord, axis=1)
                min_dist = np.min(distances)
                
                if min_dist < self.HYDROPHOBIC_DISTANCE_MAX:
                    hydrophobic.append({
                        'ligand_atom': lig_idx,
                        'protein_residue': res_id,
                        'distance': float(min_dist),
                        'interaction': 'hydrophobic'
                    })
        
        return hydrophobic
    
    def detect_salt_bridges(self, ligand_mol: Chem.Mol) -> List[Dict]:
        """
        Detect salt bridge interactions.
        
        Criteria:
        - Charged ligand atom near oppositely charged residue
        - Distance < 4.5Å
        """
        salt_bridges = []
        ligand_conf = ligand_mol.GetConformer()
        
        # Get charged atoms in ligand
        ligand_positive = self._get_positive_atoms(ligand_mol)
        ligand_negative = self._get_negative_atoms(ligand_mol)
        
        # Check charged residues
        for (chain, res_num, res_name), res_atoms in self.protein.residues.items():
            res_props = AMINO_ACID_PROPERTIES.get(res_name, {})
            
            if res_props.get('type') != 'charged':
                continue
            
            res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
            res_charge = res_props.get('charge')
            res_id = f"{res_name}{res_num}:{chain}"
            
            # Opposite charges
            target_atoms = ligand_negative if res_charge == '+' else ligand_positive
            
            for lig_idx in target_atoms:
                lig_pos = ligand_conf.GetAtomPosition(lig_idx)
                lig_coord = np.array([lig_pos.x, lig_pos.y, lig_pos.z])
                
                distances = np.linalg.norm(res_coords - lig_coord, axis=1)
                min_dist = np.min(distances)
                
                if min_dist < self.SALT_BRIDGE_DISTANCE_MAX:
                    salt_bridges.append({
                        'ligand_atom': lig_idx,
                        'protein_residue': res_id,
                        'distance': float(min_dist),
                        'interaction': 'salt_bridge'
                    })
        
        return salt_bridges
    
    # Helper methods
    def _get_hbond_donors(self, mol: Chem.Mol) -> List[int]:
        """Get hydrogen bond donor atoms (N-H, O-H)."""
        donors = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O']:
                if atom.GetTotalNumHs() > 0:
                    donors.append(atom.GetIdx())
        return donors
    
    def _get_hbond_acceptors(self, mol: Chem.Mol) -> List[int]:
        """Get hydrogen bond acceptor atoms (N, O with lone pairs)."""
        acceptors = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O', 'F']:
                acceptors.append(atom.GetIdx())
        return acceptors
    
    def _get_aromatic_rings(self, mol: Chem.Mol) -> List[Tuple[int, ...]]:
        """Get aromatic ring atom indices."""
        aromatic_rings = []
        ring_info = mol.GetRingInfo()
        
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic_rings.append(ring)
        
        return aromatic_rings
    
    def _get_hydrophobic_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get hydrophobic atoms (C not near polar groups)."""
        hydrophobic = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                # Check not adjacent to polar atoms
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if not any(n in ['N', 'O', 'F'] for n in neighbors):
                    hydrophobic.append(atom.GetIdx())
        return hydrophobic
    
    def _get_positive_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get positively charged atoms."""
        positive = []
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() > 0:
                positive.append(atom.GetIdx())
        return positive
    
    def _get_negative_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get negatively charged atoms."""
        negative = []
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() < 0:
                negative.append(atom.GetIdx())
        return negative


def map_interactions_to_fragments(
    interactions: Dict[str, List[Dict]],
    fragment_labels: List[int],
    fragment_smiles: List[str]
) -> Dict[str, List[Dict]]:
    """
    Map atom-level interactions to fragment-level interactions.
    
    Args:
        interactions: Output from NCIDetector.detect_all_interactions()
        fragment_labels: List mapping atom index to fragment index
        fragment_smiles: List of fragment SMILES strings
        
    Returns:
        Dictionary mapping fragment SMILES to their interactions
    """
    fragment_interactions = defaultdict(lambda: {
        'hydrogen_bonds': [],
        'pi_stacking': [],
        'hydrophobic': [],
        'salt_bridges': []
    })
    
    # Map hydrogen bonds
    for hb in interactions['hydrogen_bonds']:
        atom_idx = hb['ligand_atom']
        if atom_idx < len(fragment_labels):
            frag_idx = fragment_labels[atom_idx]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['hydrogen_bonds'].append({
                'residue': hb['protein_residue'],
                'distance': hb['distance'],
                'type': hb['type']
            })
    
    # Map pi-stacking
    for pi in interactions['pi_stacking']:
        ring_atoms = pi['ligand_atoms']
        if ring_atoms and ring_atoms[0] < len(fragment_labels):
            frag_idx = fragment_labels[ring_atoms[0]]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['pi_stacking'].append({
                'residue': pi['protein_residue'],
                'distance': pi['distance']
            })
    
    # Map hydrophobic contacts
    for hydro in interactions['hydrophobic']:
        atom_idx = hydro['ligand_atom']
        if atom_idx < len(fragment_labels):
            frag_idx = fragment_labels[atom_idx]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['hydrophobic'].append({
                'residue': hydro['protein_residue'],
                'distance': hydro['distance']
            })
    
    # Map salt bridges
    for salt in interactions['salt_bridges']:
        atom_idx = salt['ligand_atom']
        if atom_idx < len(fragment_labels):
            frag_idx = fragment_labels[atom_idx]
            frag_smi = fragment_smiles[frag_idx]
            fragment_interactions[frag_smi]['salt_bridges'].append({
                'residue': salt['protein_residue'],
                'distance': salt['distance']
            })
    
    return dict(fragment_interactions)


def format_fragment_interactions(fragment_interactions: Dict[str, Dict]) -> str:
    """
    Format fragment interactions into human-readable text for LLM.
    
    Args:
        fragment_interactions: Output from map_interactions_to_fragments()
        
    Returns:
        Formatted string describing interactions
    """
    lines = []
    
    for frag_idx, (frag_smi, interactions) in enumerate(fragment_interactions.items(), 1):
        lines.append(f"\n**Fragment {frag_idx}: {frag_smi}**")
        
        total_interactions = sum(len(v) for v in interactions.values())
        if total_interactions == 0:
            lines.append("  - No significant interactions detected")
            continue
        
        # Hydrogen bonds
        if interactions['hydrogen_bonds']:
            hb_by_residue = defaultdict(list)
            for hb in interactions['hydrogen_bonds']:
                hb_by_residue[hb['residue']].append(hb)
            
            lines.append(f"  - Hydrogen Bonds ({len(interactions['hydrogen_bonds'])} total):")
            for residue, hbs in sorted(hb_by_residue.items()):
                distances = [hb['distance'] for hb in hbs]
                avg_dist = np.mean(distances)
                lines.append(f"    • {residue}: {len(hbs)} bond(s), avg distance {avg_dist:.2f}Å")
        
        # Pi-stacking
        if interactions['pi_stacking']:
            lines.append(f"  - π-π Stacking ({len(interactions['pi_stacking'])} total):")
            for pi in interactions['pi_stacking']:
                lines.append(f"    • {pi['residue']}: {pi['distance']:.2f}Å")
        
        # Hydrophobic
        if interactions['hydrophobic']:
            hydro_by_residue = defaultdict(list)
            for h in interactions['hydrophobic']:
                hydro_by_residue[h['residue']].append(h['distance'])
            
            lines.append(f"  - Hydrophobic Contacts ({len(interactions['hydrophobic'])} total):")
            for residue, distances in sorted(hydro_by_residue.items()):
                lines.append(f"    • {residue}: {len(distances)} contact(s)")
        
        # Salt bridges
        if interactions['salt_bridges']:
            lines.append(f"  - Salt Bridges ({len(interactions['salt_bridges'])} total):")
            for salt in interactions['salt_bridges']:
                lines.append(f"    • {salt['residue']}: {salt['distance']:.2f}Å")
    
    return "\n".join(lines)


if __name__ == "__main__":
    print("✓ NCI Detector Module Loaded")
    print("\nUsage:")
    print("  detector = NCIDetector('protein.pdb')")
    print("  ligand_mol = Chem.SDMolSupplier('ligand.sdf')[0]")
    print("  interactions = detector.detect_all_interactions(ligand_mol)")
    print("  frag_interactions = map_interactions_to_fragments(interactions, labels, frags)")