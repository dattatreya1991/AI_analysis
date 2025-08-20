import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

def calculate_wow_yoy(df, date_col, metric_col):
    """Calculate week-over-week and year-over-year changes"""
    df = df.sort_values(by=date_col)
    df["WoW_Change"] = df[metric_col].pct_change(periods=7) * 100
    df["YoY_Change"] = df[metric_col].pct_change(periods=365) * 100
    return df

def perform_t_test(current_value, previous_values, alpha=0.05):
    """Perform a one-sample t-test to check if current_value is significantly different from previous_values mean.
    Returns True if significant, False otherwise.
    """
    if len(previous_values) < 2: #Needs at least 2 observations for t-test
        return False, np.nan
    
    try:
        t_statistic, p_value = stats.ttest_1samp(previous_values, current_value)
        return p_value < alpha, p_value
    except Exception:
        return False, np.nan

def perform_hierarchical_analysis_updated(df, dimensions, metrics, date_col, alpha=0.05, max_combinations=200):
    """Enhanced hierarchical analysis with statistical significance testing - supports up to Level 4"""
    results = []
    
    #Filter out user-level dimensions to avoid user-specific analysis
    user_keywords = ['user', 'user_id', 'userid', 'customer_id', 'customerid', 'customer', 'client_id', 'clientid']
    filtered_dimensions = [dim for dim in dimensions if not any(keyword in dim.lower() for keyword in user_keywords)]
    
    if len(filtered_dimensions) < len(dimensions):
        excluded_dims = [dim for dim in dimensions if dim not in filtered_dimensions]
        print(f"Excluded user-level dimensions: {excluded_dims}")
    
    dimensions = filtered_dimensions
    
    #Pre-compute overall data for all metrics at once
    overall_grouped = df.groupby(date_col)[metrics].sum().reset_index()
    
    #Start with overall aggregates - vectorized processing
    for metric in metrics:
        if len(overall_grouped) > 7:
            overall_df = calculate_wow_yoy(overall_grouped[[date_col, metric]], date_col, metric)
            latest_overall = overall_df.iloc[-1]  # Use iloc[-1] instead of sort
            previous_week_values = overall_df[metric].iloc[-8:-1]
            is_significant, p_value = perform_t_test(latest_overall[metric], previous_week_values, alpha)
            
            results.append({
                "Level": "Overall",
                "Dimension_Combination": "All",
                "Metric": metric,
                "Latest_WoW_Change": latest_overall["WoW_Change"] if not pd.isna(latest_overall["WoW_Change"]) else 0,
                "Latest_YoY_Change": latest_overall["YoY_Change"] if not pd.isna(latest_overall["YoY_Change"]) else 0,
                "Latest_Value": latest_overall[metric],
                "Is_Statistically_Significant": is_significant,
                "P_Value": p_value
            })

    #Single dimension analysis (Level 1) - enhanced to support up to 5 dimensions
    for dim in dimensions[:5]:  # Increased from 3 to 5 dimensions
        #Sample dimension values if too many (performance optimization)
        unique_values = df[dim].unique()
        if len(unique_values) > 20:  # Increased threshold
            # Sample top values by total volume to focus on most important segments
            top_values = df.groupby(dim)[metrics[0]].sum().nlargest(15).index.tolist()
            unique_values = top_values
        
        for metric in metrics:
            #Pre-group all data for this dimension-metric combination
            dim_grouped = df.groupby([date_col, dim])[metric].sum().reset_index()
            
            for dim_value in unique_values:
                dim_subset = dim_grouped[dim_grouped[dim] == dim_value]
                if len(dim_subset) > 7:
                    dim_subset = calculate_wow_yoy(dim_subset, date_col, metric)
                    latest_data = dim_subset.iloc[-1]  # Use iloc[-1] instead of sort
                    previous_week_values = dim_subset[metric].iloc[-8:-1]
                    is_significant, p_value = perform_t_test(latest_data[metric], previous_week_values, alpha)
                    
                    results.append({
                        "Level": f"Level 1 ({dim})",
                        "Dimension_Combination": str(dim_value),
                        "Metric": metric,
                        "Latest_WoW_Change": latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0,
                        "Latest_YoY_Change": latest_data["YoY_Change"] if not pd.isna(latest_data["YoY_Change"]) else 0,
                        "Latest_Value": latest_data[metric],
                        "Is_Statistically_Significant": is_significant,
                        "P_Value": p_value
                    })

    #Two dimension combinations (Level 2) - supports 20 combinations
    if len(dimensions) >= 2:
        for i, dim1 in enumerate(dimensions[:4]): 
            for dim2 in dimensions[i+1:5]:  
                for metric in metrics:
                    #Pre-compute combinations and sample intelligently
                    combo_grouped = df.groupby([date_col, dim1, dim2])[metric].sum().reset_index()
                    
                    #Get top combinations by latest value 
                    latest_combos = combo_grouped.groupby([dim1, dim2])[metric].last().nlargest(20) 
                    combo_count = 0
                    
                    for (val1, val2), _ in latest_combos.items():
                        if combo_count >= 20:  
                            break
                            
                        combo_subset = combo_grouped[
                            (combo_grouped[dim1] == val1) & 
                            (combo_grouped[dim2] == val2)
                        ]
                        
                        if len(combo_subset) > 7:
                            combo_subset = calculate_wow_yoy(combo_subset, date_col, metric)
                            latest_data = combo_subset.iloc[-1]  
                            previous_week_values = combo_subset[metric].iloc[-8:-1]
                            is_significant, p_value = perform_t_test(latest_data[metric], previous_week_values, alpha)
                            
                            combo_str = f"{val1} x {val2}"
                            results.append({
                                "Level": f"Level 2 ({dim1} x {dim2})",
                                "Dimension_Combination": combo_str,
                                "Metric": metric,
                                "Latest_WoW_Change": latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0,
                                "Latest_YoY_Change": latest_data["YoY_Change"] if not pd.isna(latest_data["YoY_Change"]) else 0,
                                "Latest_Value": latest_data[metric],
                                "Is_Statistically_Significant": is_significant,
                                "P_Value": p_value
                            })
                            combo_count += 1

    #Three dimension combinations (Level 3)
    if len(dimensions) >= 3:
        for i, dim1 in enumerate(dimensions[:3]):
            for j, dim2 in enumerate(dimensions[i+1:4]):
                for dim3 in dimensions[i+j+2:4]:
                    for metric in metrics:
                        #Pre-compute combinations and sample intelligently
                        combo_grouped = df.groupby([date_col, dim1, dim2, dim3])[metric].sum().reset_index()
                        
                        #Get top combinations by latest value 
                        latest_combos = combo_grouped.groupby([dim1, dim2, dim3])[metric].last().nlargest(10)
                        combo_count = 0
                        
                        for (val1, val2, val3), _ in latest_combos.items():
                            if combo_count >= 10: 
                                break
                                
                            combo_subset = combo_grouped[
                                (combo_grouped[dim1] == val1) & 
                                (combo_grouped[dim2] == val2) &
                                (combo_grouped[dim3] == val3)
                            ]
                            
                            if len(combo_subset) > 7:
                                combo_subset = calculate_wow_yoy(combo_subset, date_col, metric)
                                latest_data = combo_subset.iloc[-1]
                                previous_week_values = combo_subset[metric].iloc[-8:-1]
                                is_significant, p_value = perform_t_test(latest_data[metric], previous_week_values, alpha)
                                
                                combo_str = f"{val1} x {val2} x {val3}"
                                results.append({
                                    "Level": f"Level 3 ({dim1} x {dim2} x {dim3})",
                                    "Dimension_Combination": combo_str,
                                    "Metric": metric,
                                    "Latest_WoW_Change": latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0,
                                    "Latest_YoY_Change": latest_data["YoY_Change"] if not pd.isna(latest_data["YoY_Change"]) else 0,
                                    "Latest_Value": latest_data[metric],
                                    "Is_Statistically_Significant": is_significant,
                                    "P_Value": p_value
                                })
                                combo_count += 1

    #Four dimension combinations (Level 4) 
    if len(dimensions) >= 4:
        for i, dim1 in enumerate(dimensions[:2]):  
            for j, dim2 in enumerate(dimensions[i+1:3]):
                for k, dim3 in enumerate(dimensions[i+j+2:4]):
                    for dim4 in dimensions[i+j+k+3:4]:
                        for metric in metrics:
                            #Pre-compute combinations and sample intelligently
                            combo_grouped = df.groupby([date_col, dim1, dim2, dim3, dim4])[metric].sum().reset_index()
                            
                            #Get top combinations by latest value 
                            latest_combos = combo_grouped.groupby([dim1, dim2, dim3, dim4])[metric].last().nlargest(5)
                            combo_count = 0
                            
                            for (val1, val2, val3, val4), _ in latest_combos.items():
                                if combo_count >= 5: 
                                    break
                                    
                                combo_subset = combo_grouped[
                                    (combo_grouped[dim1] == val1) & 
                                    (combo_grouped[dim2] == val2) &
                                    (combo_grouped[dim3] == val3) &
                                    (combo_grouped[dim4] == val4)
                                ]
                                
                                if len(combo_subset) > 7:
                                    combo_subset = calculate_wow_yoy(combo_subset, date_col, metric)
                                    latest_data = combo_subset.iloc[-1]
                                    previous_week_values = combo_subset[metric].iloc[-8:-1]
                                    is_significant, p_value = perform_t_test(latest_data[metric], previous_week_values, alpha)
                                    
                                    combo_str = f"{val1} x {val2} x {val3} x {val4}"
                                    results.append({
                                        "Level": f"Level 4 ({dim1} x {dim2} x {dim3} x {dim4})",
                                        "Dimension_Combination": combo_str,
                                        "Metric": metric,
                                        "Latest_WoW_Change": latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0,
                                        "Latest_YoY_Change": latest_data["YoY_Change"] if not pd.isna(latest_data["YoY_Change"]) else 0,
                                        "Latest_Value": latest_data[metric],
                                        "Is_Statistically_Significant": is_significant,
                                        "P_Value": p_value
                                    })
                                    combo_count += 1

    return pd.DataFrame(results)

def rank_combinations(df, metric_col, change_col, top_n=5):
    """Rank top and bottom performing combinations - optimized"""
    df_clean = df[df[change_col].notna() & (df[change_col] != 0)]
    
    if len(df_clean[df_clean["Level"] != "Overall"]) >= top_n:
        df_clean = df_clean[df_clean["Level"] != "Overall"]
    
    ranked_top = df_clean.nlargest(top_n, change_col)
    ranked_bottom = df_clean.nsmallest(top_n, change_col)
    return ranked_top, ranked_bottom

def detect_significant_changes(df, metric_col, change_col, threshold=10, p_value_threshold=0.05):
    """Detect significant changes - vectorized operations for better performance"""
    df["Significant_Change_Flag"] = np.abs(df[change_col]) >= threshold
    df["Statistical_Significance_Flag"] = df["P_Value"] < p_value_threshold
    return df

def detect_masked_issues_improved(df, dimensions, metrics, date_col, change_type="WoW_Change"):
    """Optimized masked issue detection for large datasets"""
    masked_issues = []
    
    #Pre-compute overall metrics for all at once
    overall_grouped = df.groupby(date_col)[metrics].sum().reset_index()

    for metric in metrics:
        if len(overall_grouped) < 8:
            continue
            
        overall_metric_df = calculate_wow_yoy(overall_grouped[[date_col, metric]], date_col, metric)
        latest_overall_change = overall_metric_df[change_type].iloc[-1] if not overall_metric_df.empty else np.nan

        if pd.isna(latest_overall_change):
            continue

        #Only check if overall change is small (potential masking) 
        if abs(latest_overall_change) < 5:
            #Sample dimensions 
            for dim in dimensions[:2]:
                unique_values = df[dim].unique()
                if len(unique_values) > 15:
                    #Sample top values by volume
                    top_values = df.groupby(dim)[metric].sum().nlargest(10).index.tolist()
                    unique_values = top_values
                
                dim_grouped = df.groupby([date_col, dim])[metric].sum().reset_index()
                segment_changes = []
                
                for dim_value in unique_values:
                    dim_subset = dim_grouped[dim_grouped[dim] == dim_value]
                    if len(dim_subset) >= 8:
                        dim_subset = calculate_wow_yoy(dim_subset, date_col, metric)
                        latest_change = dim_subset[change_type].iloc[-1]
                        if not pd.isna(latest_change) and abs(latest_change) >= 2.0:
                            segment_changes.append({
                                'segment': dim_value,
                                'change': latest_change
                            })
                
                #Look for offsetting trends: at least one positive and one negative significant change
                positive_changes = [s for s in segment_changes if s['change'] > 0]
                negative_changes = [s for s in segment_changes if s['change'] < 0]
                
                #Check for masking - if segments have high variance but overall is stable for examples
                if positive_changes and negative_changes:
                    #sort to pick the most impactful positive and negative for the message
                    positive_changes.sort(key=lambda x: x['change'], reverse=True)
                    negative_changes.sort(key=lambda x: x['change'])
                    
                    masked_issues.append({
                        "Metric": metric,
                        "Dimension": dim,
                        "Positive_Segment": positive_changes[0]['segment'],
                        "Positive_Change": positive_changes[0]['change'],
                        "Negative_Segment": negative_changes[0]['segment'],
                        "Negative_Change": negative_changes[0]['change'],
                        "Overall_Change": latest_overall_change,
                        "Issue": f"Hidden offsetting trends in {metric} by {dim}: {positive_changes[0]['segment']} (+{positive_changes[0]['change']:.1f}%) vs {negative_changes[0]['segment']} ({negative_changes[0]['change']:.1f}%)"
                    })

    return masked_issues

def calculate_impact(df, metric_col, change_col, metric_value_col, business_criticality_weights):
    """Calculate business impact - vectorized for better performance"""
    df["Impact"] = 0.0
    
    # Create masks for valid data
    valid_mask = df[change_col].notna() & df[metric_value_col].notna() & (df[change_col] != 0)
    
    #Vectorized calculation for all rows 
    for metric, weight in business_criticality_weights.items():
        metric_mask = valid_mask & (df["Metric"] == metric)
        if metric_mask.any():
            df.loc[metric_mask, "Impact"] = (
                np.abs(df.loc[metric_mask, change_col] / 100) * 
                (df.loc[metric_mask, metric_value_col] / 1000) * 
                weight
            )
    
    #Set minimum impact for valid entries
    df.loc[valid_mask & (df["Impact"] == 0), "Impact"] = 0.01
    return df

def prioritize_findings(df, metrics_of_interest, top_n=5):
    """Prioritize findings by business impact - optimized"""
    prioritized_results = pd.DataFrame()
    
    for metric in metrics_of_interest:
        if metric in df["Metric"].values:
            metric_df = df[df["Metric"] == metric].nlargest(top_n, "Impact")
            prioritized_results = pd.concat([prioritized_results, metric_df])
    
    return prioritized_results.nlargest(top_n, "Impact")

def perform_multi_dimensional_breakdown_advanced(df, dimensions, target_metric, date_col, dimension_combination):
    """Advanced multi-dimensional breakdown analysis"""
    narrative = f"## 🔍 Multi-Dimensional Breakdown for {target_metric}\n\n"
    
    try:
        # Parse dimension combination to understand the context
        if dimension_combination == "All":
            narrative += f"**Overall Analysis:** Looking at {target_metric} across all segments.\n\n"
            
            # Show breakdown by top dimensions
            for dim in dimensions[:3]:
                dim_analysis = df.groupby(dim)[target_metric].agg(['sum', 'mean', 'count']).reset_index()
                dim_analysis = dim_analysis.sort_values('sum', ascending=False).head(5)
                
                narrative += f"**Top 5 {dim} segments by total {target_metric}:**\n"
                for _, row in dim_analysis.iterrows():
                    narrative += f"- {row[dim]}: {row['sum']:,.0f} total, {row['mean']:.1f} average\n"
                narrative += "\n"
        else:
            narrative += f"**Segment Analysis:** Analyzing {target_metric} for {dimension_combination}.\n\n"
            
            # Try to break down further if possible
            remaining_dims = [d for d in dimensions if d not in dimension_combination]
            if remaining_dims:
                for dim in remaining_dims[:2]:
                    try:
                        breakdown = df.groupby(dim)[target_metric].agg(['sum', 'mean']).reset_index()
                        breakdown = breakdown.sort_values('sum', ascending=False).head(3)
                        
                        narrative += f"**Breakdown by {dim}:**\n"
                        for _, row in breakdown.iterrows():
                            narrative += f"- {row[dim]}: {row['sum']:,.0f} total, {row['mean']:.1f} average\n"
                        narrative += "\n"
                    except:
                        continue
        
        # Add time-based insights
        if len(df) > 30:  # Ensure enough data
            recent_data = df.tail(30)
            older_data = df.head(30) if len(df) > 60 else df.head(len(df)//2)
            
            recent_avg = recent_data[target_metric].mean()
            older_avg = older_data[target_metric].mean()
            
            if recent_avg > older_avg:
                trend = "increasing"
                change_pct = ((recent_avg - older_avg) / older_avg) * 100
            else:
                trend = "decreasing"
                change_pct = ((older_avg - recent_avg) / older_avg) * 100
            
            narrative += f"**Trend Analysis:** {target_metric} has been {trend} over time "
            narrative += f"(~{change_pct:.1f}% change from earlier period).\n\n"
        
    except Exception as e:
        narrative += f"Unable to perform detailed breakdown due to data structure. Error: {str(e)}\n\n"
    
    return narrative

def perform_cross_metric_impact_analysis_advanced(df, metrics, target_metric, date_col, dimension_combination):
    """Advanced cross-metric impact analysis with correlation detection"""
    narrative = f"## 🔗 Cross-Metric Impact Analysis for {target_metric} in {dimension_combination}\n\n"
    narrative += "Understanding the upstream and downstream factors that influenced this change.\n\n"
    
    try:
        # Filter data for the specific dimension combination if not "All"
        if dimension_combination != "All":
            # This is a simplified filter - in practice, you'd need to parse the combination
            analysis_df = df.copy()
        else:
            analysis_df = df.copy()
        
        narrative += f"Let's analyze the factors contributing to the change in {target_metric} for {dimension_combination}.\n\n"
        
        # Calculate correlations with other metrics
        other_metrics = [m for m in metrics if m != target_metric and m in analysis_df.columns]
        
        if other_metrics:
            # Group by date to get time series for correlation
            time_series = analysis_df.groupby(date_col)[metrics].sum().reset_index()
            
            if len(time_series) > 4:  # Need minimum data points for correlation
                correlations = []
                
                for metric in other_metrics:
                    if metric in time_series.columns:
                        try:
                            corr = time_series[target_metric].corr(time_series[metric])
                            if not pd.isna(corr):
                                correlations.append({
                                    'metric': metric,
                                    'correlation': corr,
                                    'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak',
                                    'direction': 'positive' if corr > 0 else 'negative'
                                })
                        except:
                            continue
                
                if correlations:
                    # Sort by absolute correlation strength
                    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    
                    # Group by correlation strength
                    strong_pos = [c for c in correlations if c['correlation'] > 0.5]
                    strong_neg = [c for c in correlations if c['correlation'] < -0.5]
                    moderate_pos = [c for c in correlations if 0.3 <= c['correlation'] <= 0.5]
                    moderate_neg = [c for c in correlations if -0.5 <= c['correlation'] <= -0.3]
                    
                    narrative += "### 🔢 Correlation Insights:\n\n"
                    
                    if strong_pos:
                        narrative += "**Strong Positive Correlations** (move together):\n"
                        for c in strong_pos:
                            narrative += f"- **{c['metric']}**: r = {c['correlation']:.3f} (strong positive)\n"
                        narrative += "\n"
                    
                    if strong_neg:
                        narrative += "**Strong Negative Correlations** (move opposite):\n"
                        for c in strong_neg:
                            narrative += f"- **{c['metric']}**: r = {c['correlation']:.3f} (strong negative)\n"
                        narrative += "\n"
                    
                    if moderate_pos:
                        narrative += "**Moderate Positive Correlations**:\n"
                        for c in moderate_pos:
                            narrative += f"- **{c['metric']}**: r = {c['correlation']:.3f} (moderate positive)\n"
                        narrative += "\n"
                    
                    if moderate_neg:
                        narrative += "**Moderate Negative Correlations**:\n"
                        for c in moderate_neg:
                            narrative += f"- **{c['metric']}**: r = {c['correlation']:.3f} (moderate negative)\n"
                        narrative += "\n"
                    
                    narrative += "*Correlation Guide: r > 0.5 (strong), 0.3-0.5 (moderate), < 0.3 (weak)*\n\n"
                    
                    # Calculate recent changes for correlated metrics
                    if len(time_series) >= 8:
                        recent_changes = {}
                        for metric in other_metrics:
                            if metric in time_series.columns:
                                recent_val = time_series[metric].iloc[-1]
                                older_val = time_series[metric].iloc[-8] if len(time_series) >= 8 else time_series[metric].iloc[0]
                                if older_val != 0:
                                    change_pct = ((recent_val - older_val) / older_val) * 100
                                    recent_changes[metric] = change_pct
                        
                        if recent_changes:
                            # Categorize changes
                            increases = {k: v for k, v in recent_changes.items() if v > 2}
                            decreases = {k: v for k, v in recent_changes.items() if v < -2}
                            stable = {k: v for k, v in recent_changes.items() if -2 <= v <= 2}
                            
                            if increases:
                                narrative += "### 📈 Metrics that increased significantly:\n"
                                for metric, change in sorted(increases.items(), key=lambda x: x[1], reverse=True)[:5]:
                                    narrative += f"- **{metric}**: increased by {change:.1f}%\n"
                                narrative += "\n"
                            
                            if decreases:
                                narrative += "### 📉 Metrics that decreased significantly:\n"
                                for metric, change in sorted(decreases.items(), key=lambda x: x[1])[:5]:
                                    narrative += f"- **{metric}**: decreased by {abs(change):.1f}%\n"
                                narrative += "\n"
                            
                            if stable:
                                narrative += "### ➡️ Metrics that remained stable:\n"
                                for metric, change in list(stable.items())[:3]:
                                    narrative += f"- **{metric}**: changed by {change:+.1f}%\n"
                                narrative += "\n"
                else:
                    narrative += "No significant correlations found with other metrics in the dataset.\n\n"
            else:
                narrative += "Insufficient historical data for correlation analysis.\n\n"
        else:
            narrative += "No other metrics available for cross-metric analysis.\n\n"
            
    except Exception as e:
        narrative += f"Unable to perform cross-metric analysis. Error: {str(e)}\n\n"
    
    return narrative

def detect_hidden_issues_advanced(df, dimensions, metrics, date_col):
    """Advanced hidden issue detection"""
    hidden_issues = []
    
    try:
        for metric in metrics:
            for dim in dimensions[:2]:  # Limit to first 2 dimensions for performance
                # Look for segments with unusual patterns
                dim_grouped = df.groupby([date_col, dim])[metric].sum().reset_index()
                
                for dim_value in df[dim].unique()[:10]:  # Limit to top 10 values
                    dim_subset = dim_grouped[dim_grouped[dim] == dim_value]
                    
                    if len(dim_subset) >= 14:  # Need at least 2 weeks of data
                        dim_subset = calculate_wow_yoy(dim_subset, date_col, metric)
                        
                        # Look for volatility or unusual patterns
                        recent_changes = dim_subset['WoW_Change'].tail(4)  # Last 4 weeks
                        volatility = recent_changes.std()
                        
                        if volatility > 20:  # High volatility threshold
                            hidden_issues.append({
                                'type': 'High Volatility',
                                'metric': metric,
                                'dimension': dim,
                                'segment': dim_value,
                                'description': f"High volatility in {metric} for {dim_value} (std: {volatility:.1f}%)"
                            })
    except Exception as e:
        pass  # Fail silently for performance
    
    return hidden_issues

def get_business_context(metric, change_percentage):
    """Get business context and recommendations based on metric and change"""
    
    # Load business context from JSON file
    import json
    import os
    
    try:
        # Try to load business context from file
        context_file = os.path.join(os.path.dirname(__file__), 'business_context.json')
        if os.path.exists(context_file):
            with open(context_file, 'r') as f:
                business_contexts = json.load(f)
        else:
            # Fallback to default contexts
            business_contexts = {
                "revenue": {
                    "positive": {
                        "reasons": ["Successful marketing campaigns", "Product launches", "Market expansion", "Improved conversion rates"],
                        "recommendations": ["Scale successful campaigns", "Invest in high-performing channels", "Expand to similar markets"]
                    },
                    "negative": {
                        "reasons": ["Market competition", "Economic downturn", "Product issues", "Marketing inefficiencies"],
                        "recommendations": ["Review competitive positioning", "Optimize marketing spend", "Improve product offerings"]
                    }
                }
            }
    except:
        # Minimal fallback
        business_contexts = {}
    
    # Determine if change is positive or negative
    change_type = "positive" if change_percentage > 0 else "negative"
    change_magnitude = "significant" if abs(change_percentage) > 10 else "moderate"
    
    #Get context for the metric
    metric_context = business_contexts.get(metric, {})
    change_context = metric_context.get(change_type, {})
    
    #Build narrative
    narrative = f"## 💡 Business Context and Recommendations\n\n"
    narrative += f"Here are some possible business reasons and actionable recommendations for this change.\n\n"
    
    #Determine trend classification
    if abs(change_percentage) < 2:
        trend_class = "Stable Trend"
        trend_color = "🔵"
        trend_desc = "Minimal Change - Monitoring Recommended"
    elif change_percentage > 5:
        trend_class = "Positive Trend"
        trend_color = "🟢"
        trend_desc = "Growth Opportunity - Scale Up"
    elif change_percentage < -5:
        trend_class = "Negative Trend"
        trend_color = "🔴"
        trend_desc = "Requires Immediate Attention"
    else:
        trend_class = "Neutral Trend"
        trend_color = "🟡"
        trend_desc = "Further Investigation Recommended"
    
    narrative += f"{trend_color} **{trend_class} - {trend_desc}**\n\n"
    

    narrative += "**Possible Business Reasons:**\n\n"
    reasons = change_context.get("reasons", ["No specific reasons defined."])
    for reason in reasons:
        narrative += f"• {reason}\n"
    narrative += "\n"
    
    narrative += "**Actionable Recommendations:**\n\n"
    recommendations = change_context.get("recommendations", ["No specific recommendations defined."])
    for rec in recommendations:
        narrative += f"• {rec}\n"
    narrative += "\n"
    
    return narrative

