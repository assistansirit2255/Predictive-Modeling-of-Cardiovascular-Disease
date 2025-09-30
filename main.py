#!/usr/bin/env python3
"""
Main script to run the Cardiovascular Disease Risk Prediction analysis.

This script demonstrates the complete machine learning pipeline:
1. Dataset generation (68,000+ records)
2. Random Forest model training (targeting 90% accuracy)
3. SHAP analysis for feature importance
4. Identification of top 3 risk factors
"""

import os
import sys
sys.path.append('/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/src')

from generate_dataset import save_dataset
from cvd_model import CVDPredictor

def ensure_directories():
    """Ensure all necessary directories exist."""
    base_path = "/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease"
    directories = ['data', 'results', 'src']
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    print("Project directories verified.")

def main():
    """Run the complete CVD prediction pipeline."""
    print("🫀 CARDIOVASCULAR DISEASE RISK PREDICTION PROJECT")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Generate dataset
    print("\n📊 STEP 1: Generating cardiovascular disease dataset...")
    try:
        dataset = save_dataset()
        print("✅ Dataset generation completed successfully!")
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return
    
    # Step 2: Run ML analysis
    print("\n🤖 STEP 2: Running machine learning analysis...")
    try:
        predictor = CVDPredictor()
        data_path = "/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/data/cvd_dataset.csv"
        results = predictor.run_complete_analysis(data_path)
        
        print("✅ Machine learning analysis completed successfully!")
        
        # Display key results
        print(f"\n🎯 KEY RESULTS:")
        print(f"   • Model Accuracy: {results['accuracy']:.1%}")
        print(f"   • ROC AUC Score: {results['roc_auc']:.3f}")
        print(f"   • Dataset Size: {dataset.shape[0]:,} records")
        
        print(f"\n🔍 TOP 3 CARDIOVASCULAR RISK FACTORS:")
        for i, factor in enumerate(results['top_3_factors'], 1):
            print(f"   {i}. {factor.replace('_', ' ').title()}")
            
    except Exception as e:
        print(f"❌ Error in ML analysis: {e}")
        return
    
    print(f"\n📈 ANALYSIS SUMMARY:")
    print(f"   • Successfully analyzed {dataset.shape[0]:,} patient records")
    print(f"   • Achieved {results['accuracy']:.1%} prediction accuracy with Random Forest")
    print(f"   • Used SHAP for model interpretability")
    print(f"   • Identified key risk factors: {', '.join(results['top_3_factors'])}")
    print(f"   • Generated visualizations in results/ directory")
    
    print(f"\n✨ Project completed successfully!")
    print(f"📁 Check the results/ directory for detailed visualizations and analysis.")

if __name__ == "__main__":
    main()