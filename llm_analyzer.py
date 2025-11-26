"""
LLM-based Fragment Analyzer for Drug Design with NCI Detection

This module uses LLM to analyze protein-ligand interactions and identify
key binding fragments, enhanced with actual NCI detection.
"""

import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

# Import CIDD's fragment decomposition (fallback to RDKit BRICS if not available)
BRICS_AVAILABLE = False
try:
    import sys
    sys.path.append('/data/home/luruiqiang/liu/zhangx/CIDD/src/generation')
    from get_fragment import frag_mol_brics
    BRICS_AVAILABLE = True
    print("✓ CIDD's frag_mol_brics loaded")
except:
    print("⚠️  CIDD's frag_mol_brics not available, using RDKit BRICS fallback")

# Import NCI detection (NEW)
NCI_AVAILABLE = False
try:
    from nci_detector import NCIDetector, map_interactions_to_fragments, format_fragment_interactions
    NCI_AVAILABLE = True
    print("✓ NCI detector loaded")
except ImportError:
    print("⚠️  NCI detector not available. Place nci_detector.py in the same directory.")


class DeepSeekLLMClient:
    """
    Client for DeepSeek API calls with structured output handling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (if None, reads from DEEPSEEK_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
        
        self.api_base = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def send_request(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Send request to DeepSeek API.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            Response content from LLM, or None if failed
        """
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


class FragmentAnalyzer:
    """
    Analyzes protein-ligand interactions using LLM to identify key binding fragments.
    Enhanced with actual NCI detection.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            api_key: DeepSeek API key
        """
        self.llm_client = DeepSeekLLMClient(api_key)
        self.conversation_history: List[Dict[str, str]] = []
    
    def decompose_ligand(self, ligand_mol: Chem.Mol) -> Tuple[List[str], List[int], str]:
        """
        Decompose ligand into fragments using BRICS.
        
        Args:
            ligand_mol: RDKit molecule object
            
        Returns:
            Tuple of (fragment_smiles_list, fragment_labels, formatted_text)
        """
        # Try CIDD's BRICS first
        if BRICS_AVAILABLE:
            try:
                frags, labels, text = frag_mol_brics(ligand_mol)
                return frags, labels, text
            except Exception as e:
                print(f"⚠️  CIDD BRICS failed: {e}, using RDKit fallback")
        
        # Fallback: RDKit BRICS
        try:
            from rdkit.Chem import BRICS
            bonds = list(BRICS.FindBRICSBonds(ligand_mol))
            
            if bonds:
                frags_mol = BRICS.BreakBRICSBonds(ligand_mol)
                frags_list = Chem.GetMolFrags(frags_mol, asMols=True)
                frags = [Chem.MolToSmiles(f) for f in frags_list]
                
                # Create simple labeling
                labels = [0] * ligand_mol.GetNumAtoms()
                atom_idx = 0
                for frag_idx, frag_mol in enumerate(frags_list):
                    for _ in range(frag_mol.GetNumAtoms()):
                        if atom_idx < len(labels):
                            labels[atom_idx] = frag_idx
                            atom_idx += 1
            else:
                # No BRICS bonds - single fragment
                frags = [Chem.MolToSmiles(ligand_mol)]
                labels = [0] * ligand_mol.GetNumAtoms()
            
            text = f"{len(frags)} fragments identified using BRICS"
            return frags, labels, text
            
        except Exception as e:
            print(f"BRICS decomposition failed: {e}")
            # Ultimate fallback: whole molecule
            smiles = Chem.MolToSmiles(ligand_mol)
            return [smiles], [0] * ligand_mol.GetNumAtoms(), "Unable to decompose"
    
    def build_analysis_prompt(
        self,
        pocket_info: Dict[str, any],
        ligand_fragments: List[str],
        fragment_text: str
    ) -> str:
        """
        Build structured prompt for LLM analysis (WITHOUT NCI data).
        
        This is the basic version without actual NCI detection.
        For NCI-enhanced analysis, use analyze_fragments_with_nci().
        
        Args:
            pocket_info: Dictionary containing pocket features
            ligand_fragments: List of fragment SMILES
            fragment_text: Formatted fragment information with coordinates
            
        Returns:
            Formatted prompt string
        """
        # Format pocket information
        pocket_desc = f"""
## Protein Pocket Characteristics

**Basic Properties:**
- Volume: {pocket_info.get('volume', 'N/A')} Ų
- Depth: {pocket_info.get('depth', 'N/A')} Å
- Number of residues: {pocket_info.get('num_residues', 'N/A')}

**Chemical Composition:**
- Hydrophobic residues: {pocket_info.get('hydrophobic_ratio', 0):.1%}
- Polar residues: {pocket_info.get('polar_ratio', 0):.1%}
- Charged residues: {pocket_info.get('charged_ratio', 0):.1%}

**Key Residues:**
{', '.join(pocket_info.get('key_residues', ['N/A']))}

**Potential Interaction Sites:**
- Hydrogen bond donors: {', '.join(pocket_info.get('hb_donors', ['N/A']))}
- Hydrogen bond acceptors: {', '.join(pocket_info.get('hb_acceptors', ['N/A']))}
- Aromatic residues: {', '.join(pocket_info.get('aromatic', ['N/A']))}
"""
        
        # Format fragments
        frag_list = "\n".join([f"{i+1}. {frag}" for i, frag in enumerate(ligand_fragments)])
        
        # Build complete prompt
        prompt = f"""You are an expert medicinal chemist specializing in structure-based drug design. 
Your task is to analyze a known ligand bound to a protein pocket and identify the most critical 
fragments for binding.

{pocket_desc}

## Known Ligand Fragments (BRICS Decomposition)

The ligand has been decomposed into the following fragments:
{frag_list}

## Detailed Fragment Information
{fragment_text if fragment_text != "Unknown" else "3D coordinates unavailable"}

## Task

Analyze the fragments and identify the **TOP 3 most critical fragments** for protein-ligand binding.
Consider:
1. Which fragments likely form hydrogen bonds with polar/charged residues
2. Which fragments occupy hydrophobic pockets
3. Which fragments provide key π-π stacking or aromatic interactions

**Output Requirements:**
Respond ONLY with a valid JSON object in this exact format:
{{
  "critical_fragments": [
    {{
      "rank": 1,
      "fragment_smiles": "fragment_smiles_here",
      "fragment_index": 0,
      "interaction_type": "hydrogen_bond | hydrophobic | pi_stacking | salt_bridge",
      "target_residues": ["RES123", "RES456"],
      "rationale": "Brief explanation of why this fragment is critical"
    }},
    {{
      "rank": 2,
      "fragment_smiles": "...",
      "fragment_index": 1,
      "interaction_type": "...",
      "target_residues": ["..."],
      "rationale": "..."
    }},
    {{
      "rank": 3,
      "fragment_smiles": "...",
      "fragment_index": 2,
      "interaction_type": "...",
      "target_residues": ["..."],
      "rationale": "..."
    }}
  ],
  "overall_assessment": "Brief overall assessment of the ligand's binding mode"
}}

CRITICAL: Your response must be ONLY the JSON object, with NO additional text, NO markdown code blocks, NO explanations outside the JSON.
"""
        return prompt
    
    def _build_nci_aware_prompt(
        self,
        pocket_info: Dict[str, any],
        fragment_smiles: List[str],
        fragment_interactions: Dict[str, Dict]
    ) -> str:
        """
        Build enhanced prompt with actual NCI detection data.
        
        This is the key improvement over the basic prompt:
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
    
    def analyze_fragments(
        self,
        pocket_info: Dict[str, any],
        ligand_mol: Chem.Mol
    ) -> Optional[Dict[str, any]]:
        """
        Basic analysis function without NCI detection.
        
        For NCI-enhanced analysis, use analyze_fragments_with_nci() instead.
        
        Args:
            pocket_info: Dictionary with pocket features
            ligand_mol: RDKit molecule object of the ligand
            
        Returns:
            Dictionary with analysis results, or None if failed
        """
        print("🔬 Starting basic fragment analysis (no NCI detection)...")
        
        # Step 1: Decompose ligand
        print("📦 Decomposing ligand into fragments...")
        frags, labels, text = self.decompose_ligand(ligand_mol)
        print(f"   Found {len(frags)} fragments")
        
        # Step 2: Build prompt
        print("💬 Building LLM prompt...")
        prompt = self.build_analysis_prompt(pocket_info, frags, text)
        
        # Step 3: Call LLM
        print("🤖 Calling DeepSeek API...")
        messages = [
            {
                "role": "system",
                "content": "You are an expert medicinal chemist. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm_client.send_request(messages, temperature=0.3)
        
        if not response:
            print("❌ LLM request failed")
            return None
        
        # Step 4: Parse response
        print("📊 Parsing LLM response...")
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            
            # Add original fragments to result
            analysis['all_fragments'] = frags
            analysis['fragment_labels'] = labels
            
            print("✅ Analysis complete!")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Raw response:\n{response}")
            return None
    
    def analyze_fragments_with_nci(
        self,
        protein_pdb: str,
        ligand_mol: Chem.Mol,
        pocket_info: Dict[str, any]
    ) -> Optional[Dict[str, any]]:
        """
        Enhanced analysis with actual NCI detection.
        
        This is the recommended method that implements CIDD's core functionality.
        
        Args:
            protein_pdb: Path to protein PDB file
            ligand_mol: RDKit molecule with 3D coordinates
            pocket_info: Pocket features dictionary
            
        Returns:
            Dictionary with comprehensive analysis including NCI data
        """
        if not NCI_AVAILABLE:
            print("⚠️  NCI detection not available, falling back to basic analysis")
            return self.analyze_fragments(pocket_info, ligand_mol)
        
        print("🔬 Starting NCI-enhanced fragment analysis...")
        
        # Step 1: Decompose ligand
        print("📦 Decomposing ligand...")
        frags, labels, text = self.decompose_ligand(ligand_mol)
        print(f"   Found {len(frags)} fragments")
        
        # Step 2: Detect NCIs
        print("🔍 Detecting non-covalent interactions...")
        try:
            detector = NCIDetector(protein_pdb)
            interactions = detector.detect_all_interactions(ligand_mol)
            
            total_int = sum(len(v) for v in interactions.values())
            print(f"   Detected {total_int} interactions:")
            print(f"     - Hydrogen bonds: {len(interactions['hydrogen_bonds'])}")
            print(f"     - π-π stacking: {len(interactions['pi_stacking'])}")
            print(f"     - Hydrophobic: {len(interactions['hydrophobic'])}")
            print(f"     - Salt bridges: {len(interactions['salt_bridges'])}")
        except Exception as e:
            print(f"❌ NCI detection failed: {e}")
            return self.analyze_fragments(pocket_info, ligand_mol)
        
        # Step 3: Map NCIs to fragments
        print("🗺️  Mapping interactions to fragments...")
        fragment_interactions = map_interactions_to_fragments(
            interactions, labels, frags
        )
        
        # Step 4: Build NCI-aware prompt
        print("💬 Building enhanced prompt...")
        prompt = self._build_nci_aware_prompt(
            pocket_info, frags, fragment_interactions
        )
        
        # Step 5: Call LLM
        print("🤖 Calling DeepSeek API...")
        messages = [
            {
                "role": "system",
                "content": "You are an expert medicinal chemist. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm_client.send_request(messages, temperature=0.2, max_tokens=2500)
        
        if not response:
            print("❌ LLM request failed")
            return None
        
        # Step 6: Parse response
        print("📊 Parsing response...")
        try:
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
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Raw response:\n{response}")
            return None


def validate_analysis_result(analysis: Dict[str, any]) -> bool:
    """
    Validate that the analysis result has required fields.
    
    Args:
        analysis: Analysis dictionary from LLM
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['critical_fragments', 'overall_assessment']
    
    if not all(key in analysis for key in required_keys):
        print("❌ Missing required keys in analysis")
        return False
    
    if not isinstance(analysis['critical_fragments'], list):
        print("❌ 'critical_fragments' must be a list")
        return False
    
    if len(analysis['critical_fragments']) < 1:
        print("❌ Must have at least 1 critical fragment")
        return False
    
    # Check each fragment entry
    for frag in analysis['critical_fragments']:
        required_frag_keys = ['fragment_smiles', 'rationale']
        if not all(key in frag for key in required_frag_keys):
            print(f"❌ Fragment missing required keys: {frag}")
            return False
    
    return True


if __name__ == "__main__":
    print("✓ Fragment Analyzer Module Loaded")
    print("\nFeatures:")
    print("  - BRICS fragment decomposition")
    if NCI_AVAILABLE:
        print("  - NCI detection (H-bonds, π-π, hydrophobic, salt bridges)")
        print("  - Fragment-NCI mapping")
        print("\nRecommended: Use analyze_fragments_with_nci() for best results")
    else:
        print("  - Basic analysis (install nci_detector.py for NCI detection)")