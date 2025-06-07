from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from app.utils.logger import get_logger
from decimal import Decimal, ROUND_HALF_UP

logger = get_logger(__name__)

class DataValidator:
    """Validates and monitors financial data quality."""
    
    def __init__(self):
        # Validation rules for different metric types
        self.validation_rules = {
            "percentage": {
                "min": 0.0,
                "max": 100.0,
                "message": "Percentage value must be between 0 and 100"
            },
            "amount": {
                "min": -1e15,  # Allow negative values for losses
                "max": 1e15,
                "message": "Amount value is outside reasonable range"
            },
            "ratio": {
                "min": 0.0,
                "max": 1000.0,
                "message": "Ratio value is outside reasonable range"
            }
        }
        
        # Expected relationships between metrics
        self.metric_relationships = {
            "npa": {
                "gross_npa": ">=",
                "net_npa": "gross_npa"
            },
            "capital": {
                "tier1_capital": ">=",
                "tier2_capital": "tier1_capital"
            },
            "liquidity": {
                "lcr": ">=",
                "nsfr": "100"
            }
        }
        
        # Historical data for trend analysis
        self.historical_data = {}
        
        # Quality metrics
        self.quality_metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0
        }

    def validate_metrics(self, metrics: Dict[str, Dict[str, float]], pdf_path: Path) -> Dict[str, Any]:
        """Validate financial metrics and check relationships."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 0.0
        }
        
        try:
            # Check metric values against rules
            for category, category_metrics in metrics.items():
                for metric_name, value in category_metrics.items():
                    # Determine metric type
                    metric_type = self._get_metric_type(metric_name)
                    
                    # Validate value
                    if not self._validate_value(value, metric_type):
                        validation_results["errors"].append(
                            f"Invalid {metric_name} value: {value}"
                        )
                        validation_results["is_valid"] = False
                    
                    # Check relationships
                    if category in self.metric_relationships:
                        relationship_errors = self._check_relationships(
                            category, metric_name, value, metrics
                        )
                        validation_results["errors"].extend(relationship_errors)
                        if relationship_errors:
                            validation_results["is_valid"] = False
            
            # Check for missing required metrics
            missing_metrics = self._check_required_metrics(metrics)
            if missing_metrics:
                validation_results["warnings"].extend(
                    f"Missing required metric: {metric}" for metric in missing_metrics
                )
            
            # Calculate quality metrics
            self._calculate_quality_metrics(metrics, pdf_path)
            validation_results["quality_score"] = self._calculate_quality_score()
            
            # Update historical data
            self._update_historical_data(metrics, pdf_path)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating metrics: {str(e)}")
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")
            return validation_results

    def _get_metric_type(self, metric_name: str) -> str:
        """Determine the type of metric based on its name."""
        if any(x in metric_name.lower() for x in ["ratio", "rate", "margin"]):
            return "ratio"
        elif any(x in metric_name.lower() for x in ["growth", "change", "npa"]):
            return "percentage"
        else:
            return "amount"

    def _validate_value(self, value: float, value_type: str) -> bool:
        """Validate a numeric value against rules."""
        if value_type in self.validation_rules:
            rules = self.validation_rules[value_type]
            if not (rules["min"] <= value <= rules["max"]):
                logger.warning(f"{rules['message']}: {value}")
                return False
        return True

    def _check_relationships(self, category: str, metric_name: str, value: float, 
                           all_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Check relationships between related metrics."""
        errors = []
        
        if category in self.metric_relationships:
            relationships = self.metric_relationships[category]
            
            if metric_name in relationships:
                relationship = relationships[metric_name]
                
                if isinstance(relationship, str):
                    # Compare with another metric
                    if relationship in all_metrics.get(category, {}):
                        other_value = all_metrics[category][relationship]
                        if value < other_value:
                            errors.append(
                                f"{metric_name} ({value}) should be greater than or equal to "
                                f"{relationship} ({other_value})"
                            )
                else:
                    # Compare with fixed value
                    if value < float(relationship):
                        errors.append(
                            f"{metric_name} ({value}) should be greater than or equal to "
                            f"{relationship}"
                        )
        
        return errors

    def _check_required_metrics(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Check for missing required metrics."""
        required_metrics = {
            "asset_quality": ["gross_npa", "net_npa"],
            "profitability": ["net_profit", "roe"],
            "capital": ["car", "tier1_capital"],
            "liquidity": ["lcr", "nsfr"]
        }
        
        missing_metrics = []
        
        for category, required in required_metrics.items():
            if category in metrics:
                for metric in required:
                    if metric not in metrics[category]:
                        missing_metrics.append(f"{category}.{metric}")
        
        return missing_metrics

    def _calculate_quality_metrics(self, metrics: Dict[str, Dict[str, float]], pdf_path: Path):
        """Calculate data quality metrics."""
        # Completeness
        total_metrics = sum(len(category_metrics) for category_metrics in metrics.values())
        expected_metrics = 20  # Adjust based on your requirements
        self.quality_metrics["completeness"] = min(1.0, total_metrics / expected_metrics)
        
        # Accuracy (based on validation results)
        validation_errors = len(self._get_all_validation_errors(metrics))
        self.quality_metrics["accuracy"] = 1.0 - (validation_errors / total_metrics if total_metrics > 0 else 0)
        
        # Consistency (based on relationship checks)
        relationship_errors = len(self._get_all_relationship_errors(metrics))
        self.quality_metrics["consistency"] = 1.0 - (relationship_errors / total_metrics if total_metrics > 0 else 0)
        
        # Timeliness (based on file modification time)
        file_age = datetime.now().timestamp() - pdf_path.stat().st_mtime
        max_age = 30 * 24 * 3600  # 30 days in seconds
        self.quality_metrics["timeliness"] = 1.0 - (file_age / max_age)

    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score."""
        weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "consistency": 0.2,
            "timeliness": 0.2
        }
        
        return sum(
            self.quality_metrics[metric] * weight
            for metric, weight in weights.items()
        )

    def _update_historical_data(self, metrics: Dict[str, Dict[str, float]], pdf_path: Path):
        """Update historical data for trend analysis."""
        timestamp = datetime.fromtimestamp(pdf_path.stat().st_mtime)
        
        for category, category_metrics in metrics.items():
            if category not in self.historical_data:
                self.historical_data[category] = {}
            
            for metric_name, value in category_metrics.items():
                if metric_name not in self.historical_data[category]:
                    self.historical_data[category][metric_name] = []
                
                self.historical_data[category][metric_name].append({
                    "timestamp": timestamp,
                    "value": value
                })

    def _get_all_validation_errors(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Get all validation errors for metrics."""
        errors = []
        
        for category, category_metrics in metrics.items():
            for metric_name, value in category_metrics.items():
                metric_type = self._get_metric_type(metric_name)
                if not self._validate_value(value, metric_type):
                    errors.append(f"Invalid {metric_name} value: {value}")
        
        return errors

    def _get_all_relationship_errors(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Get all relationship errors for metrics."""
        errors = []
        
        for category, category_metrics in metrics.items():
            for metric_name, value in category_metrics.items():
                relationship_errors = self._check_relationships(
                    category, metric_name, value, metrics
                )
                errors.extend(relationship_errors)
        
        return errors

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        return {
            "quality_metrics": self.quality_metrics,
            "quality_score": self._calculate_quality_score(),
            "historical_trends": self._analyze_historical_trends(),
            "recommendations": self._generate_recommendations()
        }

    def _analyze_historical_trends(self) -> Dict[str, Any]:
        """Analyze trends in historical data."""
        trends = {}
        
        for category, metrics in self.historical_data.items():
            trends[category] = {}
            
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    # Calculate basic statistics
                    values_list = [v["value"] for v in values]
                    trends[category][metric_name] = {
                        "mean": np.mean(values_list),
                        "std": np.std(values_list),
                        "trend": self._calculate_trend(values_list),
                        "volatility": np.std(values_list) / np.mean(values_list) if np.mean(values_list) != 0 else 0
                    }
        
        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction and strength."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Calculate linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determine trend
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality metrics and trends."""
        recommendations = []
        
        # Check completeness
        if self.quality_metrics["completeness"] < 0.8:
            recommendations.append(
                "Consider adding more financial metrics to improve data completeness"
            )
        
        # Check accuracy
        if self.quality_metrics["accuracy"] < 0.9:
            recommendations.append(
                "Review data validation rules and ensure proper data cleaning"
            )
        
        # Check consistency
        if self.quality_metrics["consistency"] < 0.9:
            recommendations.append(
                "Verify relationships between related metrics"
            )
        
        # Check timeliness
        if self.quality_metrics["timeliness"] < 0.8:
            recommendations.append(
                "Update financial data more frequently"
            )
        
        # Check trends
        trends = self._analyze_historical_trends()
        for category, metrics in trends.items():
            for metric_name, stats in metrics.items():
                if stats["volatility"] > 0.5:
                    recommendations.append(
                        f"High volatility detected in {category}.{metric_name}, "
                        "consider investigating the cause"
                    )
        
        return recommendations 