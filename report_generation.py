import json
import pandas as pd
from datetime import datetime, timedelta
import re

def generate_structured_json_output(hierarchical_results, impact_results, masked_issues, date_column_name, full_df=None):
    """
    Generate structured JSON output for system integration
    """
    try:
        json_output = []
        
        # Get current period (latest date in YYYY-WXX format)
        if full_df is not None and date_column_name in full_df.columns:
            latest_date = full_df[date_column_name].max()
            if pd.notna(latest_date):
                # Convert to week format
                if hasattr(latest_date, 'isocalendar'):
                    year, week, _ = latest_date.isocalendar()
                    current_period = f"{year}-W{week:02d}"
                else:
                    current_period = str(latest_date)[:10]  # YYYY-MM-DD format
            else:
                current_period = datetime.now().strftime("%Y-W%U")
        else:
            current_period = datetime.now().strftime("%Y-W%U")
        
        # Process top impact results (limit to top 10 for JSON)
        top_results = impact_results.head(10) if not impact_results.empty else hierarchical_results.head(10)
        
        for _, row in top_results.iterrows():
            # Parse dimensions from Dimension_Combination
            dimensions = parse_dimension_combination(row.get('Dimension_Combination', 'Overall'))
            
            # Determine change type
            change_pct = row.get('Latest_WoW_Change', 0)
            change_type = "increase" if change_pct > 0 else "decrease" if change_pct < 0 else "stable"
            
            # Extract drivers from cross-metric analysis (simplified)
            drivers = extract_drivers_from_results(row, hierarchical_results)
            
            # Calculate estimated impact (use Impact_Score if available, otherwise estimate)
            estimated_impact = row.get('Impact_Score', abs(change_pct * row.get('Latest_Value', 0)))
            
            json_entry = {
                "metric": row.get('Metric', 'unknown'),
                "change_type": change_type,
                "change_pct": round(change_pct, 2),
                "period": current_period,
                "dimensions": dimensions,
                "drivers": drivers,
                "estimated_impact": round(estimated_impact, 0),
                "statistical_significance": row.get('Is_Statistically_Significant', False),
                "current_value": round(row.get('Latest_Value', 0), 2)
            }
            
            json_output.append(json_entry)
        
        return json.dumps(json_output, indent=2)
        
    except Exception as e:
        return json.dumps([{"error": f"Failed to generate JSON output: {str(e)}"}], indent=2)

def parse_dimension_combination(dimension_combination):
    """
    Parse dimension combination string into structured format
    """
    if dimension_combination == "Overall" or pd.isna(dimension_combination):
        return {"level": "overall"}
    
    dimensions = {"level": "segmented"}
    
    # Handle different formats like "device_type=mobile" or "device_type=mobile,geo=US"
    if "=" in dimension_combination:
        pairs = dimension_combination.split(",")
        for pair in pairs:
            if "=" in pair:
                key, value = pair.strip().split("=", 1)
                dimensions[key.strip()] = value.strip()
    else:
        # If no equals sign, treat as single dimension value
        dimensions["segment"] = dimension_combination
    
    return dimensions

def extract_drivers_from_results(target_row, all_results):
    """
    Extract potential drivers by finding correlated metrics
    """
    drivers = {}
    
    try:
        target_metric = target_row.get('Metric', '')
        target_dimension = target_row.get('Dimension_Combination', 'Overall')
        target_change = target_row.get('Latest_WoW_Change', 0)
        
        # Find other metrics in same dimension with significant changes
        same_dimension = all_results[
            (all_results['Dimension_Combination'] == target_dimension) & 
            (all_results['Metric'] != target_metric) &
            (abs(all_results['Latest_WoW_Change']) > 2)  # Only significant changes
        ]
        
        # Take top 3 most significant changes as potential drivers
        top_drivers = same_dimension.nlargest(3, 'Latest_WoW_Change')
        
        for _, driver_row in top_drivers.iterrows():
            driver_change = driver_row.get('Latest_WoW_Change', 0)
            driver_metric = driver_row.get('Metric', 'unknown')
            drivers[driver_metric] = f"{'+' if driver_change > 0 else ''}{driver_change:.1f}%"
        
        return drivers
        
    except Exception as e:
        return {"error": f"Driver extraction failed: {str(e)}"}

def generate_executive_summary_report(hierarchical_results, impact_results, masked_issues, business_context_summary="", ai_enabled=False):
    """
    Generate executive summary report in markdown format
    """
    try:
        # Get current date
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Analyze results for key insights
        top_5_takeaways = extract_top_5_takeaways(hierarchical_results, impact_results, masked_issues)
        executive_recommendations = generate_executive_recommendations(hierarchical_results, impact_results, masked_issues, ai_enabled)
        
        # Create markdown report
        report = f"""# Executive Summary Report
**Data Science Assessment Tool Analysis**  
*Generated on {current_date}*

---

## 📊 Analysis Overview

- **Total Metrics Analyzed**: {len(hierarchical_results['Metric'].unique()) if not hierarchical_results.empty else 0}
- **Significant Changes Detected**: {len(hierarchical_results[hierarchical_results.get('Is_Statistically_Significant', False) == True]) if not hierarchical_results.empty else 0}
- **Hidden Issues Found**: {len(masked_issues) if masked_issues else 0}
- **Analysis Method**: {"AI-Enhanced Analysis" if ai_enabled else "Statistical Analysis"}

---

## 🎯 Top 5 Key Takeaways

{chr(10).join([f"{i+1}. **{takeaway['title']}**: {takeaway['description']}" for i, takeaway in enumerate(top_5_takeaways)])}

---

## 💡 Executive Recommendations

{chr(10).join([f"### {i+1}. {rec['title']}{chr(10)}{rec['description']}{chr(10)}" for i, rec in enumerate(executive_recommendations)])}

---

*This report was generated by the Data Science Assessment Tool. For detailed analysis and interactive exploration, please refer to the full application interface.*
"""
        
        return report
        
    except Exception as e:
        return f"# Executive Summary Report\n\n**Error generating report**: {str(e)}"

def extract_top_5_takeaways(hierarchical_results, impact_results, masked_issues):
    """
    Extract top 5 key takeaways from analysis results
    """
    takeaways = []
    
    try:
        # Takeaway 1: Overall performance trend
        if not hierarchical_results.empty:
            avg_change = hierarchical_results['Latest_WoW_Change'].mean()
            positive_changes = len(hierarchical_results[hierarchical_results['Latest_WoW_Change'] > 0])
            total_changes = len(hierarchical_results)
            
            if avg_change > 2:
                trend = "strong positive growth"
            elif avg_change > 0:
                trend = "moderate positive growth"
            elif avg_change > -2:
                trend = "stable performance"
            else:
                trend = "declining performance"
            
            takeaways.append({
                "title": "Overall Business Performance",
                "description": f"Business shows {trend} with {positive_changes}/{total_changes} metrics showing positive changes (average: {avg_change:.1f}% WoW)."
            })
        
        # Takeaway 2: Most significant change
        if not impact_results.empty:
            top_impact = impact_results.iloc[0]
            takeaways.append({
                "title": "Highest Impact Change",
                "description": f"{top_impact['Metric']} in {top_impact['Dimension_Combination']} shows {top_impact['Latest_WoW_Change']:.1f}% change with impact score of {top_impact.get('Impact_Score', 'N/A')}."
            })
        
        # Takeaway 3: Statistical significance
        if not hierarchical_results.empty:
            significant_changes = len(hierarchical_results[hierarchical_results.get('Is_Statistically_Significant', False) == True])
            total_changes = len(hierarchical_results)
            takeaways.append({
                "title": "Statistical Reliability",
                "description": f"{significant_changes}/{total_changes} changes are statistically significant, indicating {('high' if significant_changes/total_changes > 0.3 else 'moderate')} confidence in observed patterns."
            })
        
        # Takeaway 4: Hidden issues
        if masked_issues:
            takeaways.append({
                "title": "Hidden Issues Detection",
                "description": f"Found {len(masked_issues)} masked issues where overall metrics appear stable but underlying segments show concerning patterns."
            })
        else:
            takeaways.append({
                "title": "Segment Alignment",
                "description": "No hidden issues detected - segment-level performance aligns well with overall metrics."
            })
        
        # Takeaway 5: Dimensional analysis
        if not hierarchical_results.empty:
            dimensions_analyzed = hierarchical_results['Level'].value_counts()
            most_volatile_level = hierarchical_results.groupby('Level')['Latest_WoW_Change'].std().idxmax()
            takeaways.append({
                "title": "Dimensional Insights",
                "description": f"Analysis across {len(dimensions_analyzed)} dimensional levels reveals highest volatility in {most_volatile_level} segments, suggesting targeted optimization opportunities."
            })
        
        return takeaways[:5]  # Ensure only top 5
        
    except Exception as e:
        return [{"title": "Analysis Error", "description": f"Could not extract takeaways: {str(e)}"}]

def generate_executive_recommendations(hierarchical_results, impact_results, masked_issues, ai_enabled):
    """
    Generate 3 executive recommendations
    """
    recommendations = []
    
    try:
        # Recommendation 1: Address top priority issue
        if not impact_results.empty:
            top_issue = impact_results.iloc[0]
            if top_issue['Latest_WoW_Change'] < 0:
                recommendations.append({
                    "title": "Immediate Action Required",
                    "description": f"**Priority 1**: Address the {top_issue['Latest_WoW_Change']:.1f}% decline in {top_issue['Metric']} for {top_issue['Dimension_Combination']}. This represents the highest business impact and requires immediate investigation and corrective action."
                })
            else:
                recommendations.append({
                    "title": "Amplify Success Factors",
                    "description": f"**Priority 1**: Scale the successful strategies driving {top_issue['Latest_WoW_Change']:.1f}% growth in {top_issue['Metric']} for {top_issue['Dimension_Combination']} to other segments."
                })
        
        # Recommendation 2: Hidden issues focus
        if masked_issues:
            recommendations.append({
                "title": "Investigate Hidden Issues",
                "description": f"**Priority 2**: Conduct deeper analysis of {len(masked_issues)} masked issues where segment-level problems are hidden by overall stability. These represent potential risks that could escalate if left unaddressed."
            })
        else:
            # Alternative recommendation for monitoring
            recommendations.append({
                "title": "Implement Proactive Monitoring",
                "description": "**Priority 2**: Establish regular monitoring dashboards for the key metrics showing highest volatility to catch emerging issues before they impact overall performance."
            })
        
        # Recommendation 3: Strategic focus
        if not hierarchical_results.empty:
            # Find the dimension level with most opportunities
            level_performance = hierarchical_results.groupby('Level').agg({
                'Latest_WoW_Change': ['mean', 'count'],
                'Is_Statistically_Significant': 'sum'
            }).round(2)
            
            recommendations.append({
                "title": "Strategic Optimization Focus",
                "description": f"**Priority 3**: Focus strategic optimization efforts on segment-level analysis where the data shows the highest potential for improvement. {'Leverage AI-enhanced insights' if ai_enabled else 'Consider implementing AI-powered analysis'} for deeper root cause understanding and predictive capabilities."
            })
        
        return recommendations[:3]  # Ensure only 3 recommendations
        
    except Exception as e:
        return [{"title": "Recommendation Error", "description": f"Could not generate recommendations: {str(e)}"}]

def generate_performance_highlights(hierarchical_results, highlight_type):
    """
    Generate performance highlights for positive or negative changes
    """
    try:
        if hierarchical_results.empty:
            return "No data available for analysis."
        
        if highlight_type == "positive":
            top_performers = hierarchical_results[hierarchical_results['Latest_WoW_Change'] > 0].nlargest(3, 'Latest_WoW_Change')
            if top_performers.empty:
                return "No positive changes detected in the analysis period."
            
            highlights = []
            for _, row in top_performers.iterrows():
                highlights.append(f"- **{row['Metric']}** ({row['Dimension_Combination']}): +{row['Latest_WoW_Change']:.1f}% WoW")
            return "\n".join(highlights)
        
        else:  # negative
            bottom_performers = hierarchical_results[hierarchical_results['Latest_WoW_Change'] < 0].nsmallest(3, 'Latest_WoW_Change')
            if bottom_performers.empty:
                return "No negative changes detected in the analysis period."
            
            highlights = []
            for _, row in bottom_performers.iterrows():
                highlights.append(f"- **{row['Metric']}** ({row['Dimension_Combination']}): {row['Latest_WoW_Change']:.1f}% WoW")
            return "\n".join(highlights)
        
    except Exception as e:
        return f"Error generating highlights: {str(e)}"

def generate_hidden_issues_summary(masked_issues):
    """
    Generate summary of hidden issues
    """
    if not masked_issues:
        return "✅ **No hidden issues detected** - All segment-level changes are reflected in overall metrics."
    
    summary = f"⚠️ **{len(masked_issues)} hidden issues detected:**\n\n"
    
    for i, issue in enumerate(masked_issues[:3], 1):  # Show top 3
        metric = issue.get('metric', 'Unknown')
        description = issue.get('description', 'No description available')
        summary += f"{i}. **{metric}**: {description}\n"
    
    if len(masked_issues) > 3:
        summary += f"\n*...and {len(masked_issues) - 3} additional issues detected.*"
    
    return summary

