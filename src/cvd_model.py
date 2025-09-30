"""
Cardiovascular Disease Risk Prediction Model

This module implements a Random Forest classifier to predict CVD risk
and uses SHAP for model interpretability to identify key risk factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

class CVDPredictor:
    """Cardiovascular Disease Risk Prediction Model with SHAP Interpretability."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.shap_values = None
        
    def load_data(self, data_path):
        """Load the cardiovascular disease dataset."""
        self.df = pd.read_csv(data_path)
        print(f"Dataset loaded: {self.df.shape[0]} records, {self.df.shape[1]} features")
        return self.df
    
    def preprocess_data(self):
        """Prepare data for modeling."""
        # Separate features and target
        X = self.df.drop('cvd_risk', axis=1)
        y = self.df['cvd_risk']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale numerical features (keeping original for interpretability)
        numerical_features = ['age', 'systolic_bp', 'cholesterol', 'diastolic_bp', 
                            'bmi', 'heart_rate', 'exercise_hours', 'sleep_hours']
        
        # Store unscaled data for SHAP
        self.X_train_unscaled = X_train.copy()
        self.X_test_unscaled = X_test.copy()
        
        # For Random Forest, we typically don't need scaling, but let's keep the option
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the Random Forest model."""
        # Initialize Random Forest with parameters to achieve high accuracy (targeting 90%)
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train the model
        print("Training Random Forest model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model performance."""
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Test Accuracy: {accuracy:.3f} ({accuracy:.1%})")
        print(f"ROC AUC Score: {roc_auc:.3f}")
        
        # Classification report
        print(f"\n=== CLASSIFICATION REPORT ===")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No CVD Risk', 'CVD Risk'],
                   yticklabels=['No CVD Risk', 'CVD Risk'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy, roc_auc
    
    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest's built-in method."""
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== FEATURE IMPORTANCE (Random Forest) ===")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance_df
    
    def shap_analysis(self):
        """Perform SHAP analysis for model interpretability."""
        print(f"\n=== SHAP ANALYSIS ===")
        print("Computing SHAP values...")
        
        # Initialize SHAP explainer
        # Use a subset of training data for efficiency
        sample_size = min(1000, len(self.X_train))
        X_train_sample = self.X_train.sample(n=sample_size, random_state=self.random_state)
        
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for test set sample
        test_sample_size = min(500, len(self.X_test))
        X_test_sample = self.X_test.sample(n=test_sample_size, random_state=self.random_state)
        
        # Convert to numpy array to ensure compatibility
        X_test_sample_array = X_test_sample.values
        self.shap_values = self.shap_explainer.shap_values(X_test_sample_array)
        
        # For binary classification, use the positive class SHAP values
        if isinstance(self.shap_values, list):
            shap_values_positive = self.shap_values[1]
        else:
            shap_values_positive = self.shap_values
        
        # Feature importance based on mean absolute SHAP values
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': np.abs(shap_values_positive).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        print("Top features by SHAP importance:")
        for idx, row in shap_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['shap_importance']:.4f}")
        
        # Identify top 3 risk factors as specified in requirements
        top_3_factors = shap_importance.head(3)['feature'].tolist()
        print(f"\n=== TOP 3 CARDIOVASCULAR RISK FACTORS ===")
        for i, factor in enumerate(top_3_factors, 1):
            importance_value = shap_importance[shap_importance['feature'] == factor]['shap_importance'].iloc[0]
            print(f"{i}. {factor.replace('_', ' ').title()}: {importance_value:.4f}")
        
        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_positive, X_test_sample_array, 
                         feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig('/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/results/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_positive, X_test_sample_array, 
                         feature_names=self.feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/results/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_importance, top_3_factors
    
    def generate_insights(self, top_3_factors):
        """Generate insights about the top risk factors."""
        insights = {
            'systolic_bp': "Systolic Blood Pressure is a critical indicator of cardiovascular health. "
                          "Higher systolic pressure indicates increased strain on arteries and heart.",
            'age': "Age is a non-modifiable risk factor. As people age, arteries naturally stiffen "
                   "and the risk of cardiovascular disease increases significantly.",
            'cholesterol': "Cholesterol levels, particularly LDL cholesterol, contribute to plaque buildup "
                          "in arteries, leading to atherosclerosis and increased CVD risk."
        }
        
        print(f"\n=== CLINICAL INSIGHTS ===")
        for i, factor in enumerate(top_3_factors, 1):
            if factor in insights:
                print(f"{i}. {factor.replace('_', ' ').title()}:")
                print(f"   {insights[factor]}")
            else:
                print(f"{i}. {factor.replace('_', ' ').title()}: Key risk factor identified by the model.")
        
        return insights
    
    def run_complete_analysis(self, data_path):
        """Run the complete CVD prediction analysis."""
        print("=== CARDIOVASCULAR DISEASE RISK PREDICTION ANALYSIS ===\n")
        
        # Load and preprocess data
        self.load_data(data_path)
        self.preprocess_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        accuracy, roc_auc = self.evaluate_model()
        
        # Feature importance analysis
        rf_importance = self.feature_importance_analysis()
        
        # SHAP analysis
        shap_importance, top_3_factors = self.shap_analysis()
        
        # Generate insights
        insights = self.generate_insights(top_3_factors)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Model achieved {accuracy:.1%} accuracy on test data")
        print(f"Key findings saved to results/ directory")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'top_3_factors': top_3_factors,
            'rf_importance': rf_importance,
            'shap_importance': shap_importance
        }

def main():
    """Main function to run the CVD prediction analysis."""
    # Initialize predictor
    predictor = CVDPredictor()
    
    # Run complete analysis
    data_path = "/home/runner/work/Predictive-Modeling-of-Cardiovascular-Disease/Predictive-Modeling-of-Cardiovascular-Disease/data/cvd_dataset.csv"
    results = predictor.run_complete_analysis(data_path)
    
    return results

if __name__ == "__main__":
    results = main()