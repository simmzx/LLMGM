"""
Enhanced LLM-based Fragment Analyzer with NCI Detection

This module integrates CIDD's core NCI detection functionality with LLM analysis
for identifying key binding fragments in protein-ligand complexes.
"""

import os
import json
import sys
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

# Import CIDD's fragment decomposition (adjust path as needed)
try:
    sys.path.append('/data/home/luruiqiang/liu/zhangx/CIDD/src/generation')
    from get_fragment import frag_mol_brics
    BRICS_AVAILABLE = True
except:
    print("⚠️  CIDD's frag_mol_brics not available, using fallback")
    BRICS_AVAILABLE = False

# Import NCI detector (from the file we just created)
from nci_detector import NCIDetector, map_interactions_to_fragments, format_fragment_interactions

# Import DeepSeek client
from deepseek_client import DeepSeekClient


class EnhancedFragmentAnalyzer:
    """
    Enhanced analyzer that combines:
    1. BRICS fragment decomposition
    2. NCI detection
    3. Fragment-NCI mapping
    4. LLM-based critical fragment identification
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the enhanced analyzer.
        
        Args:
            api_key: DeepSeek API key
        """
        self.llm_client = DeepSeekClient(api_key)
    
    def decompose_ligand(self, ligand_mol: Chem.Mol) -> Tuple[List[str], List[int], str]:
        """
        Decompose ligand into fragments using BRICS.
        
        Args:
            ligand_mol: RDKit molecule object with 3D coordinates
            
        Returns:
            Tuple of (fragment_smiles_list, fragment_labels, formatted_text)
        """
        if BRICS_AVAILABLE:
            try:
                frags, labels, text = frag_mol_brics(ligand_mol)
                return frags, labels, text
            except Exception as e:
                print(f"⚠️  BRICS decomposition failed: {e}")
        
        # Fallback: use RDKit BRICS
        from rdkit.Chem import BRICS
        bonds = list(BRICS.FindBRICSBonds(ligand_mol))
        
        if bonds:
            frags_mol = BRICS.BreakBRICSBonds(ligand_mol)
            frags_list = Chem.GetMolFrags(frags_mol, asMols=True)
            frags = [Chem.MolToSmiles(f) for f in frags_list]
            
            # Simple labeling
            labels = []
            for atom_idx in range(ligand_mol.GetNumAtoms()):
                assigned = False
                for frag_idx, frag_mol in enumerate(frags_list):
                    if atom_idx < frag_mol.GetNumAtoms():
                        labels.append(frag_idx)
                        assigned = True
                        break
                if not assigned:
                    labels.append(0)
        else:
            # No BRICS bonds found - treat as single fragment
            frags = [Chem.MolToSmiles(ligand_mol)]
            labels = [0] * ligand_mol.GetNumAtoms()
        
        text = f"{len(frags)} fragments identified"
        return frags, labels, text
    
    def analyze_fragments_with_nci(
        self,
        protein_pdb: str,
        ligand_sdf: str,
        pocket_info: Optional[Dict] = None
    ) -> Optional[Dict[str, any]]:
        """
        Main analysis function: decompose ligand, detect NCIs, and identify key fragments.
        
        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file (with 3D coordinates)
            pocket_info: Optional pocket features dictionary
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print("🔬 Starting enhanced fragment analysis with NCI detection...")
        
        # Step 1: Load ligand
        print("📦 Loading ligand from SDF...")
        suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
        ligand_mol = next(iter(suppl))
        
        if ligand_mol is None:
            print("❌ Failed to load ligand from SDF")
            return None
        
        if ligand_mol.GetNumConformers() == 0:
            print("❌ Ligand has no 3D coordinates")
            return None
        
        print(f"   ✓ Loaded ligand: {Chem.MolToSmiles(ligand_mol)}")
        
        # Step 2: Decompose ligand
        print("🔪 Decomposing ligand into fragments...")
        frags, labels, text = self.decompose_ligand(ligand_mol)
        print(f"   ✓ Found {len(frags)} fragments")
        
        # Step 3: Detect NCIs
        print("🔍 Detecting non-covalent interactions...")
        try:
            detector = NCIDetector(protein_pdb)
            interactions = detector.detect_all_interactions(ligand_mol)
            
            total_interactions = sum(len(v) for v in interactions.values())
            print(f"   ✓ Detected {total_interactions} interactions:")
            print(f"     - Hydrogen bonds: {len(interactions['hydrogen_bonds'])}")
            print(f"     - π-π stacking: {len(interactions['pi_stacking'])}")
            print(f"     - Hydrophobic: {len(interactions['hydrophobic'])}")
            print(f"     - Salt bridges: {len(interactions['salt_bridges'])}")
        except Exception as e:
            print(f"❌ NCI detection failed: {e}")
            return None
        
        # Step 4: Map NCIs to fragments
        print("🗺️  Mapping interactions to fragments...")
        fragment_interactions = map_interactions_to_fragments(
            interactions, labels, frags
        )
        
        # Step 5: Build enhanced prompt with NCI data
        print("💬 Building LLM prompt with NCI data...")
        prompt = self._build_nci_aware_prompt(
            pocket_info, frags, fragment_interactions
        )
        
        # Step 6: Call LLM
        print("🤖 Calling DeepSeek API...")
        context = [
            {
                "role": "system",
                "content": "You are an expert medicinal chemist specializing in structure-based drug design. Always respond with valid JSON only."
            }
        ]
        
        response = self.llm_client.send_request(
            context, 
            prompt, 
            temperature=0.2,  # Lower temperature for more deterministic output
            max_tokens=2500
        )
        
        if not response:
            print("❌ LLM request failed")
            return None
        
        # Step 7: Parse response
        print("📊 Parsing LLM response...")
        try:
            # Clean response
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            
            # Add metadata
            analysis['all_fragments'] = frags
            analysis['fragment_labels'] = labels
            analysis['detected_interactions'] = {
                k: len(v) for k, v in interactions.items()
            }
            analysis['fragment_interactions'] = fragment_interactions
            
            print("✅ Analysis complete!")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Raw response:\n{response}")
            return None
    
    def _build_nci_aware_prompt(
        self,
        pocket_info: Optional[Dict],
        fragment_smiles: List[str],
        fragment_interactions: Dict[str, Dict]
    ) -> str:
        """
        Build enhanced prompt with actual NCI detection data.
        
        This is the key improvement over the original implementation:
        - Instead of asking LLM to guess interactions
        - We provide DETECTED interactions from structure
        """
        # Format pocket info
        pocket_section = ""
        if pocket_info:
            pocket_section = f"""
## Protein Pocket Characteristics

**Basic Properties:**
- Volume: {pocket_info.get('volume', 'N/A')} Ų
- Number of residues: {pocket_info.get('num_residues', 'N/A')}

**Chemical Composition:**
- Hydrophobic: {pocket_info.get('hydrophobic_ratio', 0):.1%}
- Polar: {pocket_info.get('polar_ratio', 0):.1%}
- Charged: {pocket_info.get('charged_ratio', 0):.1%}

**Key Residues:**
{', '.join(pocket_info.get('key_residues', [])[:10])}
"""
        
        # Format detected interactions
        interaction_section = format_fragment_interactions(fragment_interactions)
        
        # Build complete prompt
        prompt = f"""You are an expert medicinal chemist analyzing a known protein-ligand complex structure.
The ligand has been decomposed into {len(fragment_smiles)} fragments using BRICS algorithm, and we have 
DETECTED the actual non-covalent interactions between each fragment and the protein.

{pocket_section}

## Detected Fragment-Level Interactions

Based on 3D structural analysis, we detected the following interactions:

{interaction_section}

## Your Task

Based on the DETECTED interactions above (not speculation), identify the **TOP 3 most critical fragments** 
for protein-ligand binding.

**Evaluation Criteria:**
1. **Interaction Quantity**: How many interactions does the fragment form?
2. **Interaction Quality**: Are the interactions strong (e.g., H-bonds < 3.0Å)?
3. **Interaction Diversity**: Does it form multiple types of interactions?
4. **Strategic Position**: Does it interact with key residues (active site, specificity pockets)?

**IMPORTANT**: Base your ranking ONLY on the detected interactions provided above. Do not speculate 
about potential interactions that were not detected.

**Output Requirements:**
Respond ONLY with a valid JSON object in this EXACT format:
{{
  "critical_fragments": [
    {{
      "rank": 1,
      "fragment_smiles": "<exact SMILES from fragments above>",
      "interaction_summary": "Brief summary of detected interactions",
      "interaction_count": {{
        "hydrogen_bonds": 0,
        "pi_stacking": 0,
        "hydrophobic": 0,
        "salt_bridges": 0
      }},
      "key_residues": ["RES123:A", "RES456:B"],
      "rationale": "Why this fragment is most critical based on detected interactions"
    }},
    {{
      "rank": 2,
      "fragment_smiles": "...",
      "interaction_summary": "...",
      "interaction_count": {{}},
      "key_residues": [],
      "rationale": "..."
    }},
    {{
      "rank": 3,
      "fragment_smiles": "...",
      "interaction_summary": "...",
      "interaction_count": {{}},
      "key_residues": [],
      "rationale": "..."
    }}
  ],
  "overall_assessment": "Brief assessment of the overall binding mode based on detected interactions",
  "binding_hotspots": ["Description of 1-2 key binding regions identified"]
}}

CRITICAL RULES:
1. Your response must be ONLY the JSON object
2. NO markdown code blocks, NO explanations outside JSON
3. Use EXACT fragment SMILES from the list above
4. Base rankings ONLY on detected interactions, not speculation
"""
        return prompt


def validate_enhanced_analysis(analysis: Dict[str, any]) -> bool:
    """
    Validate enhanced analysis result.
    
    Args:
        analysis: Analysis dictionary from LLM
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['critical_fragments', 'overall_assessment']
    
    if not all(key in analysis for key in required_keys):
        print("❌ Missing required keys")
        return False
    
    if not isinstance(analysis['critical_fragments'], list):
        print("❌ 'critical_fragments' must be a list")
        return False
    
    if len(analysis['critical_fragments']) < 1:
        print("❌ Must have at least 1 critical fragment")
        return False
    
    # Check each fragment entry
    for frag in analysis['critical_fragments']:
        required_frag_keys = ['fragment_smiles', 'rationale', 'interaction_summary']
        if not all(key in frag for key in required_frag_keys):
            print(f"❌ Fragment missing required keys: {frag}")
            return False
    
    return True


if __name__ == "__main__":
    print("✓ Enhanced Fragment Analyzer Module Loaded")
    print("\nThis module integrates:")
    print("  1. BRICS fragment decomposition")
    print("  2. NCI detection (H-bonds, π-π, hydrophobic, salt bridges)")
    print("  3. Fragment-NCI mapping")
    print("  4. LLM-based analysis with real interaction data")