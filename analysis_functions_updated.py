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
                        "Issue": f"Masked issue in {metric}: Overall {change_type} is {latest_overall_change:.2f}%, but {dim} segments show offsetting changes: {positive_changes[0]['segment']} (+{positive_changes[0]['change']:.1f}%) vs {negative_changes[0]['segment']} ({negative_changes[0]['change']:.1f}%)"
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
            prioritized_results = pd.concat([prioritized_results, metric_df], ignore_index=True)
    
    if prioritized_results.empty:
        #If no specific metrics found, return top findings overall
        prioritized_results = df.nlargest(top_n, "Impact")
    
    return prioritized_results

def perform_root_cause_analysis(df, date_col, dimensions, metrics, selected_change, alpha):
    """Perform root cause analysis for a selected change."""
    st.header("🔬 Root Cause Analysis")
    st.markdown("Diving deeper to understand the drivers behind the selected change.")

    #Multi-Dimensional Breakdown
    perform_multi_dimensional_breakdown(df, date_col, dimensions, metrics, selected_change, alpha)

    #Cross-Metric Impact Analysis
    perform_cross_metric_impact_analysis(df, date_col, None, selected_change, metrics, dimensions)

def perform_multi_dimensional_breakdown(df, date_col, dimensions, metrics, selected_change_row, alpha):
    """Drill down into a specific change by analyzing related metrics across dimensions."""
    breakdown_results = []

    target_metric = selected_change_row["Metric"]
    target_level = selected_change_row["Level"]
    target_dim_combo = selected_change_row["Dimension_Combination"]

    st.subheader(f"🔍 Drilling Down: {target_metric} for {target_dim_combo}")
    st.markdown(f"Let's see what else changed  **{target_dim_combo}** that might explain the shift in **{target_metric}**.")

    filter_conditions = []
    if target_level == "Overall":
        pass
    elif target_level.startswith("Level 1"):
        dim_name = target_level.split("(")[1][:-1]
        filter_conditions.append((dim_name, target_dim_combo))
    elif target_level.startswith("Level 2"):
        dim_names = target_level.split("(")[1][:-1].split(" x ")
        dim_values = target_dim_combo.split(" x ")
        filter_conditions.append((dim_names[0], dim_values[0]))
        filter_conditions.append((dim_names[1], dim_values[1]))

    filtered_df = df.copy()
    for dim_name, dim_value in filter_conditions:
        filtered_df = filtered_df[filtered_df[dim_name] == dim_value]

    if filtered_df.empty:
        st.warning("Cant find data for the selected dimension combination for breakdown.")
        return pd.DataFrame()

    for metric in metrics:
        if metric == target_metric and target_level != "Overall":
            continue

        metric_df = filtered_df.groupby(date_col)[metric].sum().reset_index()
        if len(metric_df) > 7:
            metric_df = calculate_wow_yoy(metric_df, date_col, metric)
            latest_data = metric_df.sort_values(by=date_col, ascending=False).iloc[0]
            previous_week_values = metric_df[metric].iloc[-8:-1]
            is_significant, p_value = perform_t_test(latest_data[metric], previous_week_values, alpha)

            breakdown_results.append({
                "Metric": metric,
                "Latest_WoW_Change": latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0,
                "Latest_YoY_Change": latest_data["YoY_Change"] if not pd.isna(latest_data["YoY_Change"]) else 0,
                "Latest_Value": latest_data[metric],
                "Is_Statistically_Significant": is_significant,
                "P_Value": p_value
            })

    breakdown_df = pd.DataFrame(breakdown_results)
    if not breakdown_df.empty:
        st.dataframe(breakdown_df.style.format({
            "Latest_WoW_Change": "{:.2f}%", 
            "Latest_YoY_Change": "{:.2f}%",
            "Latest_Value": "{:,.2f}",
            "P_Value": "{:.3f}"
        }))
        st.markdown("**What this shows:** Changes in other numbers that might be connected to the main change you're looking at.")
    else:
        st.info("No related metric changes found for this breakdown.")

    return breakdown_df


def perform_cross_metric_impact_analysis(df, date_col, hierarchical_results, selected_change_row, metrics, dimensions):
    """Trace contributing factors for changes in key metrics (shoppers, revenue, profit)."""
    impact_analysis_results = []

    target_metric = selected_change_row["Metric"]
    target_level = selected_change_row["Level"]
    target_dim_combo = selected_change_row["Dimension_Combination"]
    target_change_percent = selected_change_row["Latest_WoW_Change"]

    st.subheader(f"🔗 Cross-Metric Impact: What Drove the Change in {target_metric} for {target_dim_combo}?")
    st.markdown(f"Let's explore how other metrics might have contributed to the **{target_change_percent:.2f}%** change in **{target_metric}** for **{target_dim_combo}**.")

    st.info("Cross-metric impact analysis is a complex feature. Can be improved still for more future enhancements.")

    related_metrics_to_show = []
    if target_metric in ["revenue", "profit"]:
        related_metrics_to_show = [m for m in ["shoppers", "avg_order_value", "conversion_rate"] if m in metrics and m != target_metric]
    elif target_metric == "shoppers":
        related_metrics_to_show = [m for m in ["spend", "impressions", "clicks"] if m in metrics and m != target_metric]
    
    if related_metrics_to_show:
        st.markdown("**Related Metrics:**")
        for related_metric in related_metrics_to_show:
            #Find the WoW change for the related metric at the same level and dimension combination
            related_change_row = hierarchical_results[
                (hierarchical_results["Metric"] == related_metric) &
                (hierarchical_results["Level"] == target_level) &
                (hierarchical_results["Dimension_Combination"] == target_dim_combo)
            ]
            if not related_change_row.empty:
                change = related_change_row["Latest_WoW_Change"].iloc[0]
                st.write(f"- **{related_metric}**: {change:.2f}% WoW Change")
    else:
        st.info("No directly related metrics found for cross-metric impact analysis in this simplified model.")

    return pd.DataFrame(impact_analysis_results)

def detect_hidden_issues_advanced(df, dimensions, metrics, date_col, hierarchical_results):
    """Advanced hidden issue detection, including specific scenarios like flat overall metrics masking declines/increases.
    This function will generate more detailed narratives for detected issues.
    """
    hidden_issues = []

    #Overall metric is flat, but segments show significant offsetting changes
    #This is an extension of masked issue detection, but with more narrative
    for metric in metrics:
        overall_change_row = hierarchical_results[
            (hierarchical_results["Metric"] == metric) &
            (hierarchical_results["Level"] == "Overall")
        ]
        if overall_change_row.empty:
            continue
        
        overall_wow_change = overall_change_row["Latest_WoW_Change"].iloc[0]
        overall_latest_value = overall_change_row["Latest_Value"].iloc[0]

        #If overall change is very small (e.g., less than 1% ,can be customized)
        if abs(overall_wow_change) < 1.0:
            for dim in dimensions:
                segment_changes = hierarchical_results[
                    (hierarchical_results["Metric"] == metric) &
                    (hierarchical_results["Level"] == f"Level 1 ({dim})")
                ]
                
                positive_segments = segment_changes[segment_changes["Latest_WoW_Change"] > 5.0] # > 5% increase
                negative_segments = segment_changes[segment_changes["Latest_WoW_Change"] < -5.0] # < -5% decrease

                if not positive_segments.empty and not negative_segments.empty:
                    #Sort by extent of change
                    top_positive = positive_segments.sort_values(by="Latest_WoW_Change", ascending=False).iloc[0]
                    top_negative = negative_segments.sort_values(by="Latest_WoW_Change", ascending=True).iloc[0]

                    #Checked if the segments are truly offsetting each other
                    #This is a heuristic: if the sum of absolute changes is much larger than overall change
                    if (abs(top_positive["Latest_WoW_Change"]) + abs(top_negative["Latest_WoW_Change"])) > (abs(overall_wow_change) * 5): #Arbitrary multiplier
                        
                        issue_narrative = f"**Hidden Issue Detected for {metric}**: Overall {metric} was flat ({overall_wow_change:.2f}% WoW change), but this masked significant offsetting trends! "
                        issue_narrative += f"For example, {top_positive['Dimension_Combination']} {metric} increased by {top_positive['Latest_WoW_Change']:.1f}%, while {top_negative['Dimension_Combination']} {metric} decreased by {abs(top_negative['Latest_WoW_Change']):.1f}%. "
                        
                        if metric == "revenue" and "avg_order_value" in metrics:
                            #Check AOV for the declining segment
                            declining_segment_aov = df[
                                (df[date_col] == df[date_col].max()) &
                                (df[dim] == top_negative['Dimension_Combination'])
                            ]["avg_order_value"].mean()
                            
                            if not pd.isna(declining_segment_aov):
                                issue_narrative += f"The decline in {top_negative['Dimension_Combination']} is particularly concerning as it typically has a higher Average Order Value ({declining_segment_aov:,.2f}). This means the overall flat trend might still represent a significant loss in high-value business." 

                        hidden_issues.append({
                            "Type": "Masked Offsetting Trend",
                            "Metric": metric,
                            "Narrative": issue_narrative
                        })

    return hidden_issues

def provide_business_context_and_recommendations(df, date_col, dimensions, metrics, hierarchical_results, impact_results, hidden_issues):
    """Provides business context and actionable recommendations for key findings."""
    st.header("💡 Business Insights & Recommendations")
    st.markdown("Translating the data into actionable insights for your business.")

    #Prioritize based on impact
    st.subheader("🚀 Top Actionable Insights (Prioritized by Impact)")
    st.markdown("These are the changes that matter most to your business, based on their potential impact.")

    if not impact_results.empty:
        #Sort by impact and take top N
        top_impactful_findings = impact_results.sort_values(by="Impact", ascending=False).head(5)
        
        for i, row in top_impactful_findings.iterrows():
            metric = row["Metric"]
            dim_combo = row["Dimension_Combination"]
            change = row["Latest_WoW_Change"]
            impact_score = row["Impact"]
            level = row["Level"]

            if change > 0:
                st.success(f"### 🎉 Opportunity: {metric} for {dim_combo} is UP {change:.1f}%! (Impact: {impact_score:.1f})")
                st.markdown(f"**What this means:** This is a positive trend! {dim_combo} is performing well in {metric}. This could be due to successful campaigns, improved customer satisfaction, or increased demand in this segment.")
                st.markdown("**Recommendation:**")
                st.markdown(f"- **Amplify Success:** Invest more resources (e.g., marketing spend, product focus) into {dim_combo} to further boost {metric}.")
                st.markdown(f"- **Learn & Replicate:** Analyze what's working well in {dim_combo} and try to apply those strategies to other underperforming segments or dimensions.")
                st.markdown(f"- **Monitor Closely:** Keep an eye on {dim_combo} to ensure this positive trend continues and identify any potential saturation points.")
            else:
                st.error(f"### 🚨 Alert: {metric} for {dim_combo} is DOWN {abs(change):.1f}%! (Impact: {impact_score:.1f})")
                st.markdown(f"**What this means:** This is a concerning decline! The {metric} for {dim_combo} is underperforming. This could be a sign of competitive pressure, operational issues, or changing customer preferences.")
                st.markdown("**Recommendation:**")
                st.markdown(f"- **Investigate Immediately:** Conduct a deeper dive into {dim_combo} to identify the precise root causes. Look at customer feedback, recent changes in operations, or competitor activity.")
                st.markdown(f"- **Mitigate Losses:** Consider short-term interventions like targeted promotions or customer retention efforts in {dim_combo}.")
                st.markdown(f"- **Strategic Review:** Evaluate if {dim_combo} needs a strategic shift or if resources should be reallocated to more promising areas.")
            st.markdown("--- ")
    else:
        st.info("No impactful findings to prioritize. Try adjusting business criticality weights or analysis thresholds.")

    # Hidden Issues Context
    if hidden_issues:
        st.subheader("🕵️‍♀️Hidden Issues: Don't Get Fooled by the overall Averages")
        st.markdown("Those masked issues we found? Here's why they're important and what to do about them.")
        for issue in hidden_issues:
            st.warning(f"### 🎭 Hidden Alert: {issue['Metric']} - {issue['Type']}")
            st.markdown(issue['Narrative'])
            st.markdown("**Recommendation:**")
            st.markdown("- **Segment-Specific Action:** Don't just look at overall numbers. Focus on the individual segments that are driving these offsetting trends. For the increasing segment, understand and amplify success. For the decreasing segment, investigate and mitigate the decline.")
            st.markdown("- **Granular Monitoring:** Implement more granular monitoring for these specific segments to catch future shifts quickly.")
            st.markdown("--- ")
    else:
        st.info("No hidden issues detected. Your overall numbers truly reflect what's happening in all segments.")

    st.subheader("✅ Next Steps for Your Data Journey")
    st.markdown("To get even more out of this tool:")
    st.markdown("- **Refine Dimensions:** Experiment with different combinations of categories (dimensions) to uncover new insights.")
    st.markdown("- **Adjust Thresholds:** Play with the 'significant change' percentage and 'statistical significance level' in the sidebar to fine-tune what gets flagged.")
    st.markdown("- **Update Weights:** Regularly review and adjust the 'business importance weights' to ensure the tool prioritizes what matters most to your current business goals.")
    st.markdown("- **Integrate More Data:** Connect this tool to live data sources for real-time insights.")

    st.markdown("--- ")
    st.markdown("### About This Analysis")
    st.markdown("**Disclaimer:** This tool provides data-driven insights and recommendations. Always combine these with your business expertise and judgment before making critical decisions.")



def perform_multi_dimensional_breakdown_advanced(df, selected_change_row, date_col, metrics, dimensions, alpha):
    """Drill down into a specific change by analyzing related metrics across dimensions, providing a narrative.
    This version is designed to be more robust and provide a clearer narrative.
    """
    breakdown_narrative = []
    
    target_metric = selected_change_row["Metric"]
    target_level = selected_change_row["Level"]
    target_dim_combo = selected_change_row["Dimension_Combination"]
    target_change_percent = selected_change_row["Latest_WoW_Change"]

    breakdown_narrative.append(f"Let's break down the **{target_change_percent:.2f}%** change in **{target_metric}** for **{target_dim_combo}**.")

    #Determine the filter conditions based on the selected change's level and dimension combination
    filter_conditions = []
    if target_level == "Overall":
        #No specific dimension filter for overall changes
        pass
    elif target_level.startswith("Level 1"):
        dim_name = target_level.split("(")[1][:-1] # e.g., 'region'
        filter_conditions.append((dim_name, target_dim_combo)) # e.g., ('region', 'North')
    elif target_level.startswith("Level 2"):
        #For Level 2, the combo is like 'dim_value1 x dim_value2'
        dim_names_raw = target_level.split("(")[1][:-1] # e.g's, 'program x business_unit'
        dim_names = [d.strip() for d in dim_names_raw.split(" x ")]
        dim_values = [v.strip() for v in target_dim_combo.split(" x ")]
        
        if len(dim_names) == len(dim_values):
            for i in range(len(dim_names)):
                filter_conditions.append((dim_names[i], dim_values[i]))
        else:
            breakdown_narrative.append("Could not parse dimension combination for Level 2 breakdown.")
            return breakdown_narrative

    filtered_df = df.copy()
    for dim_name, dim_value in filter_conditions:
        if dim_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[dim_name] == dim_value]
        else:
            breakdown_narrative.append(f"Warning: Dimension '{dim_name}' not found in data for filtering.")
            return breakdown_narrative

    if filtered_df.empty:
        breakdown_narrative.append("No data found for the selected dimension combination to perform breakdown.")
        return breakdown_narrative

    breakdown_narrative.append("Here's what we found about other related metrics:")
    
    #Analyze other metrics within the filtered segment
    for metric in metrics:
        if metric == target_metric: 
            continue

        metric_df = filtered_df.groupby(date_col)[metric].sum().reset_index()
        if len(metric_df) > 7: #Ensure enough data for WoW calculation
            metric_df = calculate_wow_yoy(metric_df, date_col, metric)
            latest_data = metric_df.sort_values(by=date_col, ascending=False).iloc[0]
            
            wow_change = latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0
            latest_value = latest_data[metric]

            if abs(wow_change) > 0.5: # Only report significant changes
                change_direction = "increased" if wow_change > 0 else "decreased"
                breakdown_narrative.append(
                    f"- **{metric}**: {change_direction} by **{wow_change:.2f}%** (from {latest_value / (1 + wow_change/100):,.0f} to {latest_value:,.0f})."
                )

    if len(breakdown_narrative) == 2: 
        breakdown_narrative.append("- No significant changes detected in other metrics for this segment.")
    
    return breakdown_narrative

def perform_cross_metric_impact_analysis_advanced(df, selected_change_row, date_col, metrics, dimensions, alpha):
    """Performs multivariate analysis to understand how other metrics correlate with and influence the target metric.
    This version analyzes all available metrics dynamically instead of using predefined relationships.
    """
    impact_narrative = []

    target_metric = selected_change_row["Metric"]
    target_level = selected_change_row["Level"]
    target_dim_combo = selected_change_row["Dimension_Combination"]
    target_change_percent = selected_change_row["Latest_WoW_Change"]

    impact_narrative.append(f"Let's analyze how other metrics correlate with the **{target_change_percent:.2f}%** change in **{target_metric}** for **{target_dim_combo}**.")

    # Determine the filter conditions based on the selected change
    filter_conditions = []
    if target_level == "Overall":
        pass
    elif target_level.startswith("Level 1"):
        dim_name = target_level.split("(")[1][:-1]
        filter_conditions.append((dim_name, target_dim_combo))
    elif target_level.startswith("Level 2"):
        dim_names_raw = target_level.split("(")[1][:-1]
        dim_names = [d.strip() for d in dim_names_raw.split(" x ")]
        dim_values = [v.strip() for v in target_dim_combo.split(" x ")]
        if len(dim_names) == len(dim_values):
            for i in range(len(dim_names)):
                filter_conditions.append((dim_names[i], dim_values[i]))

    filtered_df = df.copy()
    for dim_name, dim_value in filter_conditions:
        if dim_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[dim_name] == dim_value]
        else:
            impact_narrative.append(f"Warning: Dimension '{dim_name}' not found in data for filtering.")
            return impact_narrative

    if filtered_df.empty:
        impact_narrative.append("No data found for the selected dimension combination to perform impact analysis.")
        return impact_narrative

    #Analyze all other metrics and their changes
    other_metrics = [m for m in metrics if m != target_metric]
    if not other_metrics:
        impact_narrative.append("No other metrics available for cross-metric analysis.")
        return impact_narrative

    impact_narrative.append(f"Here's how other metrics changed alongside **{target_metric}**:")
    
    metric_changes = []
    for other_metric in other_metrics:
        try:
            other_metric_df = filtered_df.groupby(date_col)[other_metric].sum().reset_index()
            if len(other_metric_df) > 7:
                other_metric_df = calculate_wow_yoy(other_metric_df, date_col, other_metric)
                latest_data = other_metric_df.sort_values(by=date_col, ascending=False).iloc[0]
                
                wow_change = latest_data["WoW_Change"] if not pd.isna(latest_data["WoW_Change"]) else 0
                latest_value = latest_data[other_metric]
                
                metric_changes.append({
                    'metric': other_metric,
                    'change': wow_change,
                    'latest_value': latest_value
                })
        except Exception as e:
            continue

    #Sort by absolute change to show most significant changes first
    metric_changes.sort(key=lambda x: abs(x['change']), reverse=True)
    
    #Categorize changes for better narrative
    significant_increases = [m for m in metric_changes if m['change'] > 2.0]
    significant_decreases = [m for m in metric_changes if m['change'] < -2.0]
    stable_metrics = [m for m in metric_changes if -2.0 <= m['change'] <= 2.0]
    
    if significant_increases:
        impact_narrative.append("**📈 Metrics that increased significantly:**")
        for m in significant_increases[:3]:  # Show top 3
            change_direction = "increased"
            prev_value = m['latest_value'] / (1 + m['change']/100) if m['change'] != -100 else 0
            impact_narrative.append(
                f"- **{m['metric']}**: {change_direction} by **{m['change']:.2f}%** (from {prev_value:,.0f} to {m['latest_value']:,.0f})"
            )
    
    if significant_decreases:
        impact_narrative.append("**📉 Metrics that decreased significantly:**")
        for m in significant_decreases[:3]:  # Show top 3
            change_direction = "decreased"
            prev_value = m['latest_value'] / (1 + m['change']/100) if m['change'] != -100 else 0
            impact_narrative.append(
                f"- **{m['metric']}**: {change_direction} by **{abs(m['change']):.2f}%** (from {prev_value:,.0f} to {m['latest_value']:,.0f})"
            )
    
    if stable_metrics and len(significant_increases) + len(significant_decreases) < 3:
        impact_narrative.append("**📊 Metrics that remained relatively stable:**")
        for m in stable_metrics[:2]:  #Show top 2 stable ones
            impact_narrative.append(f"- **{m['metric']}**: {m['change']:+.2f}% change")
    
    #Add correlation insights with actual correlation coefficients
    if len(metric_changes) >= 2:
        impact_narrative.append("**🔗 Correlation Insights:**")
        
        #Calculate actual correlation coefficients
        try:
            #Prepare data for correlation analysis
            correlation_data = filtered_df.groupby(date_col)[metrics].sum().reset_index()
            if len(correlation_data) > 3:  
                
                target_values = correlation_data[target_metric].values
                correlations = []
                
                for other_metric in other_metrics:
                    if other_metric in correlation_data.columns:
                        other_values = correlation_data[other_metric].values
                        if len(other_values) == len(target_values) and len(other_values) > 1:
                            #Calculate Pearson correlation coefficient
                            correlation_coef = np.corrcoef(target_values, other_values)[0, 1]
                            if not np.isnan(correlation_coef):
                                correlations.append({
                                    'metric': other_metric,
                                    'correlation': correlation_coef,
                                    'strength': abs(correlation_coef)
                                })
                
                #Sort by correlation strength (absolute value)
                correlations.sort(key=lambda x: x['strength'], reverse=True)
                
                #Categorize correlations
                strong_positive = [c for c in correlations if c['correlation'] > 0.5]
                strong_negative = [c for c in correlations if c['correlation'] < -0.5]
                moderate_positive = [c for c in correlations if 0.3 <= c['correlation'] <= 0.5]
                moderate_negative = [c for c in correlations if -0.5 <= c['correlation'] <= -0.3]
                weak_correlations = [c for c in correlations if -0.3 < c['correlation'] < 0.3]
                
                if strong_positive:
                    impact_narrative.append("- **Strong Positive Correlations** (move together):")
                    for c in strong_positive[:3]:
                        impact_narrative.append(f"  - **{c['metric']}**: r = {c['correlation']:.3f} (strong positive)")
                
                if strong_negative:
                    impact_narrative.append("- **Strong Negative Correlations** (move opposite):")
                    for c in strong_negative[:3]:
                        impact_narrative.append(f"  - **{c['metric']}**: r = {c['correlation']:.3f} (strong negative)")
                
                if moderate_positive:
                    impact_narrative.append("- **Moderate Positive Correlations**:")
                    for c in moderate_positive[:2]:
                        impact_narrative.append(f"  - **{c['metric']}**: r = {c['correlation']:.3f} (moderate positive)")
                
                if moderate_negative:
                    impact_narrative.append("- **Moderate Negative Correlations**:")
                    for c in moderate_negative[:2]:
                        impact_narrative.append(f"  - **{c['metric']}**: r = {c['correlation']:.3f} (moderate negative)")
                
                if not strong_positive and not strong_negative and not moderate_positive and not moderate_negative:
                    if weak_correlations:
                        impact_narrative.append("- **Weak Correlations** (limited relationship):")
                        for c in weak_correlations[:2]:
                            strength_desc = "weak positive" if c['correlation'] > 0 else "weak negative"
                            impact_narrative.append(f"  - **{c['metric']}**: r = {c['correlation']:.3f} ({strength_desc})")
                    else:
                        impact_narrative.append(f"- No meaningful correlations detected with {target_metric}")
                
                # Add interpretation guide
                impact_narrative.append("- **Correlation Guide**: r > 0.5 (strong), 0.3-0.5 (moderate), < 0.3 (weak)")
                
            else:
                impact_narrative.append("- Not enough historical data for correlation analysis")
                
        except Exception as e:
            #Fallback to simple correlation analysis based on change directions
            target_direction = "increased" if target_change_percent > 0 else "decreased"
            correlated_metrics = []
            counter_correlated_metrics = []
            
            for m in metric_changes:
                if (target_change_percent > 0 and m['change'] > 1.0) or (target_change_percent < 0 and m['change'] < -1.0):
                    correlated_metrics.append(m['metric'])
                elif (target_change_percent > 0 and m['change'] < -1.0) or (target_change_percent < 0 and m['change'] > 1.0):
                    counter_correlated_metrics.append(m['metric'])
            
            if correlated_metrics:
                impact_narrative.append(f"- Metrics moving in the **same direction** as {target_metric}: {', '.join(correlated_metrics[:3])}")
            
            if counter_correlated_metrics:
                impact_narrative.append(f"- Metrics moving in the **opposite direction** to {target_metric}: {', '.join(counter_correlated_metrics[:3])}")
            
            if not correlated_metrics and not counter_correlated_metrics:
                impact_narrative.append(f"- No strong correlations detected with {target_metric} changes")

    if not metric_changes:
        impact_narrative.append("- No significant changes detected in other metrics for this segment.")

    return impact_narrative





def detect_hidden_issues_advanced(df, dimensions, metrics, date_col, change_type="WoW_Change"):
    """Enhanced hidden issue detection with more detailed narrative and cross-metric insights.
    This function aims to provide a narrative like(below are some random examples created to illustrate them):
    "UK revenue dropped 30% due to 50% reduction in clicks"
    "This was masked by 25% increase in US revenue from higher conversion"
    "Net result: overall revenue actually declined 5% despite stable aggregate metrics"
    """
    hidden_issues_narratives = []

    for metric in metrics:
        overall_metric_df = df.groupby(date_col)[metric].sum().reset_index()
        if len(overall_metric_df) < 8:
            continue
            
        overall_metric_df = calculate_wow_yoy(overall_metric_df, date_col, metric)
        latest_overall_change = overall_metric_df[change_type].iloc[-1] if not overall_metric_df.empty else np.nan

        if pd.isna(latest_overall_change):
            continue
            
        #Condition for masked issue: overall change is small, but segments have significant offsetting changes
        if abs(latest_overall_change) < 1.0: #Overall change must be very small (less than 1.0%)
            for dim in dimensions:
                if dim != date_col:
                    grouped_by_dim = df.groupby([date_col, dim])[metric].sum().reset_index()
                    
                    segment_changes = []
                    for segment_name in grouped_by_dim[dim].unique():
                        segment_df = grouped_by_dim[grouped_by_dim[dim] == segment_name]
                        if len(segment_df) >= 8:
                            segment_df = calculate_wow_yoy(segment_df, date_col, metric)
                            latest_segment_change = segment_df[change_type].iloc[-1]
                            if not pd.isna(latest_segment_change) and abs(latest_segment_change) >= 1.0: #Segment change must be at least 1%
                                segment_changes.append({
                                    'segment': segment_name,
                                    'change': latest_segment_change,
                                    'latest_value': segment_df[metric].iloc[-1]
                                })

                    positive_segments = [s for s in segment_changes if s['change'] > 0]
                    negative_segments = [s for s in segment_changes if s['change'] < 0]
                    
                    if positive_segments and negative_segments:
                        #Sort to pick the most impactful positive and negative for the message
                        positive_segments.sort(key=lambda x: x['change'], reverse=True)
                        negative_segments.sort(key=lambda x: x['change'])
                        
                        top_positive = positive_segments[0]
                        top_negative = negative_segments[0]

                        narrative = f"🚨 Hidden Issue Detected for {metric}! Overall {change_type} is {latest_overall_change:.2f}%, but...\n"
                        narrative += f"- **{top_negative['segment']} {metric}** dropped {abs(top_negative['change']):.1f}%\n"
                        narrative += f"- This was masked by **{top_positive['segment']} {metric}** increasing {top_positive['change']:.1f}%\n"
                        
                        #Add cross-metric insights for the negative segment - now dynamic
                        #Look for other metrics that might explain the decline
                        other_metrics = [m for m in metrics if m != metric]
                        if other_metrics:
                            negative_segment_df = df[df[dim] == top_negative['segment']]
                            for other_metric in other_metrics[:2]:  #Check top 2 other metrics
                                try:
                                    other_metric_data = negative_segment_df.groupby(date_col)[other_metric].sum().reset_index()
                                    if len(other_metric_data) > 7:
                                        other_metric_data = calculate_wow_yoy(other_metric_data, date_col, other_metric)
                                        other_change = other_metric_data['WoW_Change'].iloc[-1] if not other_metric_data.empty else np.nan
                                        if not pd.isna(other_change) and abs(other_change) > 2.0:
                                            direction = "increased" if other_change > 0 else "decreased"
                                            narrative += f"  - Likely related to **{other_metric}** in {top_negative['segment']} {direction} by {abs(other_change):.1f}%\n"
                                            break
                                except:
                                    continue

                        hidden_issues_narratives.append(narrative)

    return hidden_issues_narratives

import json

def get_business_context(metric, change_direction, context_file='business_context.json'):
    """Reads business context from a JSON file and provides reasons and recommendations."""
    try:
        with open(context_file, 'r') as f:
            context_data = json.load(f)
    except FileNotFoundError:
        return {"reasons": ["No business context file found."], "recommendations": ["Create 'business_context.json' to add custom insights."]}
    except json.JSONDecodeError:
        return {"reasons": ["Error decoding business context file."], "recommendations": ["Check 'business_context.json' for valid JSON format."]}

    metric_context = context_data.get(metric, {})
    direction_context = metric_context.get(change_direction, {})

    reasons = direction_context.get('reasons', ["No specific reasons defined."])
    recommendations = direction_context.get('recommendations', ["No specific recommendations defined."])
    flag = direction_context.get('flag', 'neutral') # 'attention', 'positive', 'neutral'

    return {"reasons": reasons, "recommendations": recommendations, "flag": flag}



