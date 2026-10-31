#!/usr/bin/env python3
"""
Data Quality Analysis for PK Extraction Results
"""

import os
import json
import sys
from collections import Counter, defaultdict

def analyze_pk_data(json_file: str):
    """Analyze the quality and coverage of extracted PK data"""
    
    print("=" * 80)
    print("PK DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    # Load data
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return
    
    print(f"\n✓ Loaded {len(data)} records from {json_file}\n")
    
    if not data:
        print("⚠️  No records to analyze")
        return
    
    # Basic stats
    print("=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    
    pmids = set(r.get('pmid') for r in data)
    print(f"Unique PMIDs: {len(pmids)}")
    print(f"Total records: {len(data)}")
    print(f"Records per paper: {len(data)/len(pmids):.1f}")
    
    # PK parameter coverage
    print("\n" + "=" * 80)
    print("PK PARAMETER COVERAGE")
    print("=" * 80)
    
    pk_params = {
        't_half_h': 'Half-life (h)',
        'AUC_value': 'AUC',
        'Cmax_value': 'Cmax',
        'Tmax_h': 'Tmax (h)',
        'Cmin_value': 'Cmin',
        'Ctrough_value': 'Ctrough',
        'CL_F_value': 'Clearance (CL/F)',
        'CLrenal_value': 'Renal Clearance',
        'Vd_value': 'Volume of Distribution',
        'bioavailability_pct': 'Bioavailability (%)',
        'protein_binding_pct': 'Protein Binding (%)',
        'renal_excretion_pct': 'Renal Excretion (%)'
    }
    
    coverage = {}
    for param, label in pk_params.items():
        count = sum(1 for r in data if r.get(param) is not None)
        pct = count / len(data) * 100
        coverage[param] = (count, pct)
        
        # Star rating
        stars = "⭐" * min(5, int(pct / 20))
        
        print(f"{label:25s} {count:4d}/{len(data)} ({pct:5.1f}%) {stars}")
    
    # Drug distribution
    print("\n" + "=" * 80)
    print("TOP 10 DRUGS")
    print("=" * 80)
    
    drug_counts = Counter(r.get('drug') for r in data if r.get('drug'))
    for drug, count in drug_counts.most_common(10):
        print(f"{drug:40s} {count:4d} records")
    
    # Study characteristics
    print("\n" + "=" * 80)
    print("STUDY CHARACTERISTICS")
    print("=" * 80)
    
    print("\nSpecies:")
    species_counts = Counter(r.get('species') for r in data if r.get('species'))
    for species, count in species_counts.most_common():
        print(f"  {species:20s} {count:4d} ({count/len(data)*100:.1f}%)")
    
    print("\nRoute:")
    route_counts = Counter(r.get('route') for r in data if r.get('route'))
    for route, count in route_counts.most_common():
        print(f"  {route:20s} {count:4d} ({count/len(data)*100:.1f}%)")
    
    print("\nStudy Phase:")
    phase_counts = Counter(r.get('study_phase') for r in data if r.get('study_phase'))
    for phase, count in phase_counts.most_common():
        print(f"  {phase:20s} {count:4d} ({count/len(data)*100:.1f}%)")
    
    print("\nFeeding State:")
    feeding_counts = Counter(r.get('feeding_state') for r in data if r.get('feeding_state'))
    for state, count in feeding_counts.most_common():
        print(f"  {state:20s} {count:4d} ({count/len(data)*100:.1f}%)")
    
    # Drug-drug interactions
    print("\n" + "=" * 80)
    print("DRUG-DRUG INTERACTIONS")
    print("=" * 80)
    
    with_ddi = sum(1 for r in data if r.get('concomitant_drugs'))
    print(f"Records with concomitant drugs: {with_ddi}/{len(data)} ({with_ddi/len(data)*100:.1f}%)")
    
    if with_ddi > 0:
        print("\nTop concomitant drugs:")
        ddi_counts = Counter(r.get('concomitant_drugs') for r in data if r.get('concomitant_drugs'))
        for drug, count in ddi_counts.most_common(10):
            print(f"  {drug[:50]:50s} {count:4d}")
    
    # Data quality
    print("\n" + "=" * 80)
    print("DATA QUALITY")
    print("=" * 80)
    
    confidence_counts = Counter(r.get('extraction_confidence') for r in data)
    print("\nExtraction Confidence:")
    for conf in ['high', 'medium', 'low']:
        count = confidence_counts.get(conf, 0)
        print(f"  {conf:10s} {count:4d} ({count/len(data)*100:.1f}%)")
    
    # Completeness per record
    print("\nRecord Completeness:")
    completeness = []
    for r in data:
        filled = sum(1 for p in pk_params.keys() if r.get(p) is not None)
        completeness.append(filled / len(pk_params) * 100)
    
    avg_completeness = sum(completeness) / len(completeness)
    print(f"  Average: {avg_completeness:.1f}% of PK parameters filled")
    
    # Records with critical params
    critical_params = ['t_half_h', 'AUC_value', 'Cmax_value']
    with_critical = sum(1 for r in data if any(r.get(p) for p in critical_params))
    print(f"  Records with at least one critical param (t½/AUC/Cmax): {with_critical}/{len(data)} ({with_critical/len(data)*100:.1f}%)")
    
    with_all_critical = sum(1 for r in data if all(r.get(p) for p in critical_params))
    print(f"  Records with all critical params: {with_all_critical}/{len(data)} ({with_all_critical/len(data)*100:.1f}%)")
    
    # ML readiness
    print("\n" + "=" * 80)
    print("MACHINE LEARNING READINESS")
    print("=" * 80)
    
    # Filter for ML-ready records
    ml_ready = [r for r in data if (
        r.get('t_half_h') is not None and
        r.get('dose_value') is not None and
        r.get('route') is not None
    )]
    
    print(f"\n✓ ML-ready records (has t½ + dose + route): {len(ml_ready)}/{len(data)} ({len(ml_ready)/len(data)*100:.1f}%)")
    
    if ml_ready:
        print("\nFeature availability in ML-ready subset:")
        features = {
            'CLrenal_value': 'Renal Clearance',
            'AUC_value': 'AUC',
            'Cmax_value': 'Cmax',
            'concomitant_drugs': 'Drug-Drug Interactions',
            'feeding_state': 'Food Effect',
            'population': 'Population Info'
        }
        
        for feat, label in features.items():
            count = sum(1 for r in ml_ready if r.get(feat) is not None)
            pct = count / len(ml_ready) * 100
            print(f"  {label:25s} {count:4d}/{len(ml_ready)} ({pct:5.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Successfully extracted {len(data)} records from {len(pmids)} papers")
    print(f"✅ Average {len(data)/len(pmids):.1f} records per paper")
    print(f"✅ {coverage['t_half_h'][0]} records with half-life ({coverage['t_half_h'][1]:.1f}%)")
    print(f"✅ {coverage['CLrenal_value'][0]} records with renal clearance ({coverage['CLrenal_value'][1]:.1f}%)")
    print(f"✅ {len(ml_ready)} ML-ready records for training")
    
    if avg_completeness >= 30:
        print(f"\n🎉 Good data quality: {avg_completeness:.1f}% average completeness")
    else:
        print(f"\n⚠️  Lower completeness: {avg_completeness:.1f}% - but this reflects what's in the literature")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default: look in output directory
        json_file = "pk_extraction_output/pk_extracted_multi.json"
        # Fallback to current directory
        if not os.path.exists(json_file):
            json_file = "pk_extracted_multi.json"
    
    analyze_pk_data(json_file)
