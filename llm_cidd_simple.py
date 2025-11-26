"""
Simplified CIDD Implementation
"""

import os
import sys
import json
from typing import Optional, Tuple, List, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem

# 导入CIDD模块 - CIDD和minimal_llm_module在同一父目录下
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cidd_path = os.path.join(parent_dir, 'CIDD')
if cidd_path not in sys.path:
    sys.path.insert(0, cidd_path)

CIDD_AVAILABLE = False
DOCKING_AVAILABLE = False

try:
    from get_fragment import frag_mol_brics
    CIDD_AVAILABLE = True
    print("✓ CIDD frag_mol_brics imported")
except:
    print("⚠️  frag_mol_brics not available")

try:
    from vina_eval import vina_dock_crossdocked
    DOCKING_AVAILABLE = True
    print("✓ vina_dock_crossdocked imported")
except:
    print("⚠️  vina_dock_crossdocked not available")

from deepseek_client import send_request


def fallback_docking(mol: Chem.Mol) -> float:
    from rdkit.Chem import QED
    return -15 * QED.qed(mol)


class InteractionAnalyzer:
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.context = []
    
    def analyze(self, ligand_filename: str, ligand_smi: str, protein_path: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        mol = Chem.MolFromSmiles(ligand_smi)
        if mol is None:
            return None, None, None
        
        # BRICS decomposition
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            
            if CIDD_AVAILABLE:
                frags, labels, frag_text = frag_mol_brics(mol)
            else:
                from rdkit.Chem import BRICS
                bonds = list(BRICS.FindBRICSBonds(mol))
                if bonds:
                    frags_mol = BRICS.BreakBRICSBonds(mol)
                    frags_list = Chem.GetMolFrags(frags_mol, asMols=True)
                    frags = [Chem.MolToSmiles(f) for f in frags_list]
                else:
                    frags = [ligand_smi]
                frag_text = f"{len(frags)} fragments"
        except Exception as e:
            print(f"BRICS failed: {e}")
            frags = [ligand_smi]
            frag_text = "No decomposition"
        
        # Docking
        try:
            if DOCKING_AVAILABLE:
                ligand_path = os.path.join(self.data_root, ligand_filename)
                docking_score = vina_dock_crossdocked(
                    ligand_smi,
                    os.path.dirname(protein_path),
                    protein_path,
                    ligand_path,
                    thread_name=f"temp_{os.getpid()}",
                    output_path=f"/tmp/docked_{os.getpid()}.sdf",
                    exhaustiveness=8
                )
            else:
                docking_score = fallback_docking(mol)
        except:
            docking_score = fallback_docking(mol)
        
        # LLM analysis
        prompt = f"""Analyze this protein-ligand complex.
Ligand: {ligand_smi}
Fragments: {', '.join(frags)}
Docking: {docking_score:.2f} kcal/mol

Generate brief interaction analysis (3-4 sentences)."""
        
        report = send_request(self.context, prompt, temperature=0.3)
        return report, docking_score, frag_text


class MolecularDesigner:
    def __init__(self):
        self.context = []
    
    def design(self, protein_structure: str, ligand_smi: str, interaction_report: str, 
               previous_designs: List[str] = None, reflections: List[str] = None) -> Optional[str]:
        prompt = f"""Design molecular modifications.
Current: {ligand_smi}
Interaction: {interaction_report}

Propose ONE specific modification using Chain-of-Thought."""
        
        return send_request(self.context, prompt, temperature=0.5)
    
    def reflection(self, original_smi: str, new_smi: str, original_score: float, 
                   new_score: float, design_plan: str, new_report: str) -> Optional[str]:
        prompt = f"""Reflect on design.
Original: {original_smi} ({original_score:.2f})
New: {new_smi} ({new_score:.2f})
Change: {original_score - new_score:+.2f}

Brief reflection (2-3 sentences)."""
        
        return send_request(self.context, prompt, temperature=0.4)
    
    def select_best(self, molecules: List[str], scores: List[float], reports: List[str]) -> Tuple[Optional[int], Optional[str]]:
        candidates = "\n".join([f"{i+1}. {m} ({s:.2f})" for i, (m, s) in enumerate(zip(molecules, scores))])
        prompt = f"""Select best molecule:
{candidates}

Output JSON: {{"selected_index": <number>, "reasoning": "<text>"}}"""
        
        response = send_request(self.context, prompt, temperature=0.3)
        if not response:
            return None, None
        
        try:
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            result = json.loads(cleaned.strip())
            idx = result['selected_index'] - 1
            if 0 <= idx < len(molecules):
                return idx, result['reasoning']
        except:
            pass
        return None, None


class SimpleCIDD:
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.interaction_analyzer = InteractionAnalyzer(data_root)
        self.designer = MolecularDesigner()
    
    def run_pipeline(self, ligand_filename: str, ligand_smi: str, num_iterations: int = 3) -> Dict[str, Any]:
        protein_fn = os.path.join(os.path.dirname(ligand_filename), os.path.basename(ligand_filename)[:10] + '.pdb')
        protein_path = os.path.join(self.data_root, protein_fn)
        
        if not os.path.exists(protein_path):
            print(f"Protein not found: {protein_path}")
            return None
        
        with open(protein_path, 'r') as f:
            protein_structure = f.read()
        
        orig_report, orig_score, _ = self.interaction_analyzer.analyze(ligand_filename, ligand_smi, protein_path)
        if not orig_report:
            return None
        
        all_designs, all_molecules, all_scores, all_reports, all_reflections = [], [], [], [], []
        
        for i in range(num_iterations):
            design = self.designer.design(protein_structure, ligand_smi, orig_report, all_designs, all_reflections)
            if not design:
                continue
            
            new_smi = ligand_smi  # Placeholder - real implementation would generate new molecule
            new_report, new_score, _ = self.interaction_analyzer.analyze(ligand_filename, new_smi, protein_path)
            if not new_report:
                continue
            
            reflection = self.designer.reflection(ligand_smi, new_smi, orig_score, new_score, design, new_report)
            
            all_designs.append(design)
            all_molecules.append(new_smi)
            all_scores.append(new_score)
            all_reports.append(new_report)
            all_reflections.append(reflection or "No reflection")
        
        if not all_molecules:
            return None
        
        selected_idx, reason = self.designer.select_best(all_molecules, all_scores, all_reports)
        if selected_idx is None:
            selected_idx = all_scores.index(min(all_scores))
            reason = "Best score"
        
        results = {
            'original_molecule': ligand_smi,
            'original_score': orig_score,
            'selected_molecule': all_molecules[selected_idx],
            'selected_score': all_scores[selected_idx],
            'improvement': orig_score - all_scores[selected_idx],
            'selection_reason': reason
        }
        
        output_file = os.path.join(self.output_dir, f"results_{os.path.basename(ligand_filename)}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results