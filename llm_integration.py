try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

import pandas as pd
import json


def generate_enhanced_root_cause_analysis(selected_change_row, multi_dim_narrative, cross_metric_narrative, hierarchical_results, full_df_summary, full_df=None, date_column=None):
    """
    Generate enhanced root cause analysis using Google Gemini 2.5 Pro with forecasting capabilities
    """
    if not GEMINI_AVAILABLE:
        return "Google Generative AI package not available. Please install with: `pip install google-generativeai`"
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prepare context for LLM
        change_context = {
            'metric': selected_change_row['Metric'],
            'dimension': selected_change_row['Dimension_Combination'],
            'change_percent': selected_change_row['Latest_WoW_Change'],
            'level': selected_change_row['Level'],
            'value': selected_change_row['Latest_Value'],
            'is_significant': selected_change_row.get('Is_Statistically_Significant', False)
        }

        # Generate forecasting data if available
        forecast_context = ""
        if full_df is not None and date_column is not None:
            try:
                forecast_context = generate_forecast_context(full_df, date_column, change_context['metric'], change_context['dimension'])
            except Exception as e:
                forecast_context = f"Forecasting data unavailable: {str(e)}"

        prompt = f"""
        You are a data scientist conducting an in-depth root cause analysis with predictive capabilities. Based on the following detailed analysis, provide comprehensive insights into why this change occurred, what it means, and what the future holds.

        TARGET CHANGE:
        - Metric: {change_context['metric']}
        - Segment: {change_context['dimension']}
        - Change: {change_context['change_percent']:.2f}% (Week-over-Week)
        - Level: {change_context['level']}
        - Current Value: {change_context['value']:,.2f}
        - Statistically Significant: {change_context['is_significant']}

        MULTI-DIMENSIONAL ANALYSIS:
        {chr(10).join(multi_dim_narrative)}

        CROSS-METRIC ANALYSIS:
        {chr(10).join(cross_metric_narrative)}

        FORECASTING CONTEXT:
        {forecast_context}

        BROADER CONTEXT:
        - Total data points analyzed: {len(hierarchical_results)}
        - Similar patterns in other metrics: {len(hierarchical_results[abs(hierarchical_results['Latest_WoW_Change'] - change_context['change_percent']) < 2])}

        Please provide:
        1. **Root Cause Hypothesis** (2-3 sentences): Most likely explanation for this change
        2. **Supporting Evidence** (4-5 bullet points including MANDATORY future forecast):
           - What data supports this hypothesis?
           - **Future Forecast**: Based on current trend, provide 30-day forecast with:
             * If positive trend: Expected growth trajectory and opportunity value
             * If negative trend: Potential losses, missed opportunities, and range estimates
             * Business impact if current trend continues unchanged
        3. **Alternative Explanations** (2-3 bullet points): Other possible causes to investigate
        4. **Deeper Investigation** (3-4 bullet points): Specific analyses to confirm root cause
        5. **Business Impact Assessment** (2-3 sentences): What this change means for the business
        6. **Monitoring Recommendations** (2-3 bullet points): What to watch going forward

        - Format your response in clear markdown with appropriate headers and bullet points.
        - Focus on actionable insights that a business team can use to make decisions.
        - ENSURE the Future Forecast bullet point is included in Supporting Evidence with specific numbers and ranges.
        - Any Title header in the form of 'Comprehensive/Root Cause Analysis' or anything else with 'Root Cause Analysis' in the summary should not be included in the final output in the app.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating enhanced root cause analysis: {str(e)}"

def generate_forecast_context(df, date_column, target_metric, target_dimension):
    """
    Generate forecasting context for the LLM based on historical data
    """
    try:
        # Filter data for the specific metric and dimension
        if target_dimension == "Overall":
            metric_data = df.groupby(date_column)[target_metric].sum().reset_index()
        else:
            # Parse dimension combination (e.g., "device_type=mobile")
            if "=" in target_dimension:
                dim_col, dim_val = target_dimension.split("=", 1)
                filtered_df = df[df[dim_col] == dim_val]
                metric_data = filtered_df.groupby(date_column)[target_metric].sum().reset_index()
            else:
                metric_data = df.groupby(date_column)[target_metric].sum().reset_index()
        
        # Sort by date
        metric_data = metric_data.sort_values(date_column)
        
        # Get recent trend (last 8 weeks)
        recent_data = metric_data.tail(8)
        if len(recent_data) < 2:
            return "Insufficient historical data for forecasting"
        
        # Calculate trend metrics
        latest_value = recent_data[target_metric].iloc[-1]
        previous_value = recent_data[target_metric].iloc[-2]
        wow_change = ((latest_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
        
        # Calculate average weekly change over recent period
        values = recent_data[target_metric].values
        weekly_changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                weekly_changes.append((values[i] - values[i-1]) / values[i-1] * 100)
        
        avg_weekly_change = sum(weekly_changes) / len(weekly_changes) if weekly_changes else 0
        
        # Simple linear projection for 30 days (approximately 4 weeks)
        projected_4_week_value = latest_value * (1 + avg_weekly_change/100) ** 4
        
        # Calculate potential impact ranges
        conservative_projection = latest_value * (1 + (avg_weekly_change * 0.5)/100) ** 4
        optimistic_projection = latest_value * (1 + (avg_weekly_change * 1.5)/100) ** 4
        
        forecast_context = f"""
        Historical Trend Analysis for {target_metric} in {target_dimension}:
        - Latest Value: {latest_value:,.2f}
        - Current WoW Change: {wow_change:.2f}%
        - Average Weekly Change (8-week): {avg_weekly_change:.2f}%
        - 30-day Projection (base case): {projected_4_week_value:,.2f}
        - 30-day Range: {conservative_projection:,.2f} to {optimistic_projection:,.2f}
        - Trend Direction: {'Positive' if avg_weekly_change > 0 else 'Negative' if avg_weekly_change < 0 else 'Stable'}
        
        Business Impact Calculations:
        - If trend continues: {abs(projected_4_week_value - latest_value):,.2f} {'gain' if projected_4_week_value > latest_value else 'loss'} over 30 days
        - Risk/Opportunity Range: {abs(optimistic_projection - conservative_projection):,.2f}
        """
        
        return forecast_context
        
    except Exception as e:
        return f"Error generating forecast context: {str(e)}"

def create_analysis_summary(hierarchical_results, metrics, dimensions):
    """
    Create a summary of the analysis for LLM context
    """
    if hierarchical_results.empty:
        return "No analysis results available"
    
    summary = f"""
    Analysis Overview:
    - Metrics analyzed: {', '.join(metrics)}
    - Dimensions analyzed: {', '.join(dimensions)}
    - Total combinations analyzed: {len(hierarchical_results)}
    - Average WoW change: {hierarchical_results['Latest_WoW_Change'].mean():.2f}%
    - Largest positive change: {hierarchical_results['Latest_WoW_Change'].max():.2f}%
    - Largest negative change: {hierarchical_results['Latest_WoW_Change'].min():.2f}%
    - Statistically significant changes: {len(hierarchical_results[hierarchical_results.get('Is_Statistically_Significant', False) == True])}
    """
    
    return summary

def create_dataframe_summary(df):
    """
    Create a summary of the dataset for LLM context
    """
    return f"""
    Dataset Summary:
    - Total rows: {len(df)}
    - Date range: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}
    - Columns: {', '.join(df.columns.tolist())}
    """

