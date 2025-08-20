import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analysis_functions_updated import (
    calculate_wow_yoy, perform_hierarchical_analysis_updated, rank_combinations,
    detect_significant_changes, detect_masked_issues_improved, calculate_impact, prioritize_findings,
    perform_multi_dimensional_breakdown_advanced, perform_cross_metric_impact_analysis_advanced,
    detect_hidden_issues_advanced, get_business_context
)
import os
from report_generation import generate_structured_json_output, generate_executive_summary_report

#Optional import for Google Generative AI
try:
    import google.generativeai as genai
    from llm_integration import generate_enhanced_root_cause_analysis, create_analysis_summary, create_dataframe_summary
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

#Set page configuration
st.set_page_config(page_title="Data Science Assessment Tool", layout="wide")

#Title
st.title("📊 Bicycle Diagnostic Tool")
st.markdown("A **dataset-agnostic** analysis tool for detecting performance changes and identifying business drivers🕵️‍♀️")

#LLM Configuration in for sidebar
st.sidebar.header("🤖AI Enhancement")

if not GEMINI_AVAILABLE:
    st.sidebar.error("Google Generative AI package not installed. Install with: `pip install google-generativeai`")
    use_llm = False
else:
    use_llm = st.sidebar.toggle("Enable AI-Powered Insights", value=False, help="Use Google Gemini 2.5 Pro for enhanced business insights and root cause analysis")

    if use_llm:
        #Check API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.sidebar.error("GEMINI_API_KEY environment variable not found. Please set it to use AI features.")
            use_llm = False
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                st.sidebar.success("✅ AI features enabled")
            except Exception as e:
                st.sidebar.error(f"❌ Error configuring Gemini API: {str(e)}")
                use_llm = False

#Sidebar for file upload and configuration
st.sidebar.header("⚙️ Configuration")

#File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

#Initialize session state for analysis results
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'hierarchical_results' not in st.session_state:
    st.session_state.hierarchical_results = pd.DataFrame()
if 'impact_results' not in st.session_state:
    st.session_state.impact_results = pd.DataFrame()
if 'full_df_for_visualizations' not in st.session_state:
    st.session_state.full_df_for_visualizations = pd.DataFrame()
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
if 'dimensions' not in st.session_state:
    st.session_state.dimensions = []
if 'date_column' not in st.session_state:
    st.session_state.date_column = None

#Check if file is uploaded
if uploaded_file is None:
    #No file uploaded - show empty state
    st.markdown("---")
    st.markdown("### No Dataset Uploaded")
    st.info("👆 Please upload a CSV file using the file uploader in the sidebar to begin your analysis.")
    st.markdown("**What you can do:**")
    st.markdown("- Upload a CSV file with your business data")
    st.markdown("- The tool will automatically detect metrics (numbers) and dimensions (categories)")
    st.markdown("- Configure analysis parameters in the sidebar")
    st.markdown("- Run comprehensive analysis to find insights in your data")
    
    # Clear session state when no file is uploaded
    st.session_state.analysis_completed = False
    st.session_state.hierarchical_results = pd.DataFrame()
    st.session_state.impact_results = pd.DataFrame()
    st.session_state.full_df_for_visualizations = pd.DataFrame()
    st.session_state.metrics = []
    st.session_state.dimensions = []
    st.session_state.date_column = None
    
    st.stop()

#File is uploaded - proceed with analysis
df = pd.read_csv(uploaded_file)

#Performance optimization for large datasets
if len(df) > 500000:  # 5 lakh rows
    st.warning("Large dataset detected")
    
    #Sample data intelligently - keep recent data and sample older data
    #First, try to identify date column for intelligent sampling
    temp_date_col = None
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            temp_date_col = col
            break
        except:
            continue
    
    if temp_date_col:
        df[temp_date_col] = pd.to_datetime(df[temp_date_col])
        df_sorted = df.sort_values(by=temp_date_col, ascending=False)
        recent_data = df_sorted.head(100000)  # Keep recent 1 lakh rows
        older_data = df_sorted.tail(len(df) - 100000)
        
        #Sample older data if it's still large
        if len(older_data) > 200000:
            older_data_sampled = older_data.sample(n=200000, random_state=42)
            df = pd.concat([recent_data, older_data_sampled]).sort_values(by=temp_date_col)
            st.info(f"📊 Dataset optimized: Using {len(df):,} rows (kept all recent data + sampled older data)")
        else:
            df = df_sorted
    else:
        #If no date column found, random sample
        df = df.sample(n=300000, random_state=42)
        st.info(f"📊 Dataset optimized: Using {len(df):,} rows (random sample)")

#Display basic info about the dataset
st.header("Dataset Overview")
st.write(f"Your dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")
st.write("Sneak peek of your data:")
st.dataframe(df.head())

#Automatically identify metrics and dimensions
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

#Try to identify date column
date_column = None
for col in df.columns:
    try:
        pd.to_datetime(df[col])
        date_column = col
        break
    except:
        continue

if date_column:
    df[date_column] = pd.to_datetime(df[date_column])
    categorical_columns = [col for col in categorical_columns if col != date_column]

st.sidebar.header("🔍 Column Identification")
st.sidebar.write(f"**Date column:** `{date_column}`")
st.sidebar.write(f"**Metrics (numbers):** `{', '.join(numeric_columns)}`")
st.sidebar.write(f"**Dimensions (categories):** `{', '.join(categorical_columns)}`")

#Allow user to modify the identification
metrics = st.sidebar.multiselect("Which numbers do you want to analyze?", numeric_columns, default=numeric_columns)
dimensions = st.sidebar.multiselect("Which categories do you want to break down by?", categorical_columns, default=categorical_columns[:4] if len(categorical_columns) >= 4 else categorical_columns)

#Performance optimization for dimensions and metrics
if len(dimensions) > 5:
    st.sidebar.info("Performance optimization: Limiting to top 5 dimensions by data volume")
    #Select top dimensions by data volume
    dim_volumes = {}
    for dim in dimensions:
        dim_volumes[dim] = df[dim].nunique() * len(df)
    
    top_dimensions = sorted(dim_volumes.items(), key=lambda x: x[1], reverse=True)[:5]
    dimensions = [dim[0] for dim in top_dimensions]
    st.sidebar.write(f"**Selected dimensions:** {', '.join(dimensions)}")

#Limit metrics if too many
if len(metrics) > 8:
    st.sidebar.info("Performance optimization: Limiting to top 8 metrics")
    metrics = metrics[:8]
    st.sidebar.write(f"**Selected metrics:** {', '.join(metrics)}")

#Store in session state
st.session_state.metrics = metrics
st.session_state.dimensions = dimensions
st.session_state.date_column = date_column

#Configuration parameters
st.sidebar.header("⚙️ Analysis Parameters")
threshold = st.sidebar.slider("How big of a change is 'significant' (in %)?", 1, 50, 5)
alpha = st.sidebar.slider("Statistical Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01)
top_n = st.sidebar.slider("Show top/bottom how many results?", 3, 10, 5)

#Business criticality weights
st.sidebar.header("💰 Business Importance Weights")
st.sidebar.write("How important each number is for the business (higher = more important).")
business_weights = {}
for metric in metrics:
    # Use a unique key for each slider to ensure proper state management
    business_weights[metric] = st.sidebar.slider(f"Importance of **{metric}**", 0.1, 5.0, 1.0, key=f"weight_{metric}")

#Main analysis
if st.button("🚀 Run Analysis!", type="primary"):
    if date_column is None:
        st.error("Sorry! I can't find a 'Date' column. Please make sure your dataset has one to track changes over time.")
    elif len(metrics) == 0 or len(dimensions) == 0:
        st.error("Please select at least one 'number to analyze' (metric) and one 'category to break down by' (dimension) to start the analysis.")
    else:
        with st.spinner("🕵️‍♀️Analyzing your data... This might take a moment for big datasets!"):
            try:
                #Perform hierarchical analysis
                hierarchical_results = perform_hierarchical_analysis_updated(df, dimensions, metrics, date_column, alpha)
                
                #Store results in session state
                st.session_state.hierarchical_results = hierarchical_results
                st.session_state.full_df_for_visualizations = df.copy()
                st.session_state.analysis_completed = True
                st.session_state.threshold = threshold
                st.session_state.alpha = alpha
                st.session_state.top_n = top_n
                st.session_state.business_weights = business_weights
                
                if hierarchical_results.empty:
                    st.error("No analysis results generated. Please check your data and try again. Make sure your dataset has enough historical data for WoW/YoY calculations.")
                    
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {str(e)}")
                st.write("This might be due to data format issues or insufficient data for certain calculations. Please check your data and try again.")

#Display analysis results 
if st.session_state.analysis_completed and not st.session_state.hierarchical_results.empty:
    #Get stored parameters
    threshold = st.session_state.get('threshold', 5)
    alpha = st.session_state.get('alpha', 0.05)
    top_n = st.session_state.get('top_n', 5)
    stored_business_weights = st.session_state.get('business_weights', {})
    hierarchical_results = st.session_state.hierarchical_results
    
    # Check if business weights have changed and recalculate impact if needed
    weights_changed = False
    if stored_business_weights != business_weights:
        weights_changed = True
        st.session_state.business_weights = business_weights
        
        # Recalculate impact scores with new weights
        significant_results_temp = hierarchical_results.copy()
        significant_results_temp = detect_significant_changes(significant_results_temp, "Metric", "Latest_WoW_Change", threshold, alpha)
        impact_results_updated = calculate_impact(significant_results_temp, "Metric", "Latest_WoW_Change", "Latest_Value", business_weights)
        st.session_state.impact_results = impact_results_updated
    
    st.header("📈 Hierarchical Analysis: What's Changing?")
    st.markdown("Looking at how your numbers (metrics) are changing over time, both week-to-week and year-to-year, across different categories (dimensions).")
    
    st.subheader("📊 Multi-Level Analysis Results")
    st.markdown("""
    This analysis examines changes at multiple levels of granularity:
    - **Overall**: Everything combined across all dimensions
    - **Level 1**: Single dimension breakdowns (up to 5 dimensions)
    - **Level 2**: Two-dimension combinations (up to 20 combinations)
    - **Level 3**: Three-dimension combinations (up to 10 combinations) 
    - **Level 4**: Four-dimension combinations (up to 5 combinations)
    
    🔒 **Note**: User-level dimensions (user_id, customer_id, etc.) are automatically excluded to focus on business-level insights.
    """)
    
    #Show level distribution
    level_counts = hierarchical_results['Level'].value_counts()
    st.write("**Analysis Coverage:**")
    for level, count in level_counts.items():
        st.write(f"• {level}: {count} combinations analyzed")
    
    st.subheader("📋Detailed Results by Level")
    st.markdown("This table shows changes at different levels with statistical significance testing.")
    
    #Shows a sample of results with better formatting
    display_df = hierarchical_results.head(30)  # Show first 30 results to include more levels
    st.dataframe(display_df.style.format({
        "Latest_WoW_Change": "{:.2f}%", 
        "Latest_YoY_Change": "{:.2f}%",
        "Latest_Value": "{:,.2f}",
        "P_Value": "{:.3f}"
    }))
    
    if len(hierarchical_results) > 30:
        st.info(f"Showing first 30 results out of {len(hierarchical_results)} total results across all levels.")
    
    #Detect significant changes
    st.header("🚨 Significant Change Detection: Big Shifts!")
    st.markdown(f"Flagging changes that are bigger than **{threshold}%** or are statistically unusual (with a significance level of **{alpha*100:.0f}%**).")
    significant_results = hierarchical_results.copy()
    significant_results = detect_significant_changes(significant_results, "Metric", "Latest_WoW_Change", threshold, alpha)
    
    significant_changes = significant_results[significant_results["Significant_Change_Flag"] == True]
    statistically_significant_changes = significant_results[significant_results["Statistical_Significance_Flag"] == True]

    if not significant_changes.empty:
        st.subheader(f"🔥These are the big WoW changes found (more than {threshold}%):")
        st.dataframe(significant_changes[["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Is_Statistically_Significant", "P_Value"]].style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "P_Value": "{:.3f}"}))
        
        #Add explanations for significant changes
        st.markdown("**What this means:**")
        for _, row in significant_changes.head(3).iterrows():
            change_direction = "increased" if row["Latest_WoW_Change"] > 0 else "decreased"
            sig_text = "and is statistically significant!" if row["Is_Statistically_Significant"] else "but is NOT statistically significant."
            st.write(f"• **{row['Metric']}** for **{row['Dimension_Combination']}** {change_direction} by **{abs(row['Latest_WoW_Change']):.1f}%** this week! {sig_text}")
    else:
        st.info(f"Good news! No big week-over-week changes detected (above {threshold}%). Business seems stable!")
    
    if not statistically_significant_changes.empty:
        st.subheader(f"✨ These changes are statistically significant (p < {alpha:.2f}):")
        st.dataframe(statistically_significant_changes[["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Is_Statistically_Significant", "P_Value"]].style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "P_Value": "{:.3f}"}))
        st.markdown("**What this means:** These changes are unlikely to be due to random chance. They are real shifts in your data!")
    else:
        st.info(f"No statistically significant changes detected (p < {alpha:.2f}).")

    #Ranking top/bottom performers
    st.header("🏆Top/Bottom Performers: Who's Winning/Losing?")
    st.markdown(f"Here are the top and bottom **{top_n}** performers based on their week-over-week changes.")
    for metric in st.session_state.metrics:
        metric_data = hierarchical_results[hierarchical_results["Metric"] == metric]
        if not metric_data.empty:
            st.subheader(f"🎯 For **{metric}**:")
            top_performers, bottom_performers = rank_combinations(metric_data, "Metric", "Latest_WoW_Change", top_n)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🚀Top Performers (WoW Increase)**")
                if not top_performers.empty:
                    st.dataframe(top_performers[["Level", "Dimension_Combination", "Latest_WoW_Change"]].style.format({"Latest_WoW_Change": "{:.2f}%"}))
                    #Add explanation
                    best_performer = top_performers.iloc[0]
                    st.success(f"🌟Best: **{best_performer['Dimension_Combination']}** is up **{best_performer['Latest_WoW_Change']:.1f}%**!")
                else:
                    st.info("No top performers found for this metric.")
            with col2:
                st.write("**📉Bottom Performers (WoW Decrease)**")
                if not bottom_performers.empty:
                    st.dataframe(bottom_performers[["Level", "Dimension_Combination", "Latest_WoW_Change"]].style.format({"Latest_WoW_Change": "{:.2f}%"}))
                    # Add explanation
                    worst_performer = bottom_performers.iloc[0]
                    st.warning(f"⚠️Needs attention: **{worst_performer['Dimension_Combination']}** is down **{abs(worst_performer['Latest_WoW_Change']):.1f}%**!")
                else:
                    st.info("No bottom performers found for this metric.")

    #Masked issue detection
    st.header("🎭 Masked Issue Detection: Hidden Surprises in data!")
    st.markdown("Sometimes, the overall numbers look fine, but hidden problems are lurking underneath. Trying to find those scenario's")
    masked_issues = detect_masked_issues_improved(st.session_state.full_df_for_visualizations, st.session_state.dimensions, st.session_state.metrics, st.session_state.date_column)
    if masked_issues:
        st.subheader("🔍Found some hidden issues!")
        for issue in masked_issues:
            st.warning(f"🎭 **{issue['Issue']}**")
            with st.expander("Tell me more!"):
                st.write(f"- **Metric:** {issue['Metric']}")
                st.write(f"- **Category:** {issue['Dimension']}")
                st.write(f"- **Offsetting Groups:** {issue['Positive_Segment']} (+{issue['Positive_Change']:.1f}%) vs {issue['Negative_Segment']} ({issue['Negative_Change']:.1f}%)")
                st.write(f"- **Overall Change:** {issue['Overall_Change']:.2f}%")
                st.markdown("**Why this matters:** When overall numbers look stable, you might miss important trends happening in specific segments of your business!")
    else:
        st.info("Great! No masked issues detected. Overall numbers truly reflect what's happening in all segments.")
    
    #Impact-based prioritization
    st.header("💥Impact-Based Prioritization: What Matters Most?")
    st.markdown("From all the changes, showing the ones that have the biggest impact on business, based on how important each number is.")
    
    # Use updated impact results if weights changed, otherwise calculate fresh
    if 'impact_results' in st.session_state and not weights_changed:
        impact_results = st.session_state.impact_results
    else:
        significant_results = hierarchical_results.copy()
        significant_results = detect_significant_changes(significant_results, "Metric", "Latest_WoW_Change", threshold, alpha)
        impact_results = calculate_impact(significant_results, "Metric", "Latest_WoW_Change", "Latest_Value", business_weights)
        st.session_state.impact_results = impact_results
    
    #Filter for key metrics if they exist, but ensure diversity
    key_metrics_for_prioritization = [m for m in ["shoppers", "revenue", "profit"] if m in st.session_state.metrics]
    
    # Always show diverse metrics across all available metrics
    if not impact_results.empty:
        # Create a diverse top 5 by getting top finding from each metric first
        diverse_results = pd.DataFrame()
        
        # Get top 1 finding from each metric
        for metric in st.session_state.metrics:
            metric_data = impact_results[impact_results["Metric"] == metric]
            if not metric_data.empty:
                top_metric_finding = metric_data.nlargest(1, "Impact")
                diverse_results = pd.concat([diverse_results, top_metric_finding])
        
        # If we have fewer than top_n results, fill with remaining top findings
        if len(diverse_results) < top_n:
            remaining_results = impact_results[~impact_results.index.isin(diverse_results.index)]
            additional_results = remaining_results.nlargest(top_n - len(diverse_results), "Impact")
            diverse_results = pd.concat([diverse_results, additional_results])
        
        # Sort by impact and take top_n
        prioritized_results = diverse_results.nlargest(top_n, "Impact")
        
        if not prioritized_results.empty:
            st.subheader(f"🎯 Top {top_n} Most Impactful Findings:")
            display_cols = ["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Impact"]
            prioritized_display = prioritized_results[display_cols]
            st.dataframe(prioritized_display.style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "Impact": "{:.2f}"}))
            
            #Explanation for top impact
            top_impact = prioritized_display.iloc[0]
            st.info(f"💡 **Biggest Impact:** {top_impact['Dimension_Combination']} in {top_impact['Metric']} (Impact Score: {top_impact['Impact']:.1f})")
            
            # Show metric diversity
            metric_counts = prioritized_display['Metric'].value_counts()
            st.write(f"**Metrics represented:** {', '.join(metric_counts.index.tolist())}")
        else:
            st.info("No impactful findings available. Try adjusting weights or threshold.")
    else:
        st.info("No impact results available for prioritization.")

    # Advanced Analysis Section
    st.sidebar.header("🔬 Advanced Analysis")
    st.sidebar.write("Select a specific change to deep dive into its root causes.")
    
    # Create a list of changes for dropdown
    if not impact_results.empty:
        change_options = []
        for _, row in impact_results.head(20).iterrows():  # Top 20 for dropdown
            change_desc = f"{row['Metric']} for {row['Dimension_Combination']} ({row['Latest_WoW_Change']:+.2f}%) - {row['Level']}"
            change_options.append(change_desc)
        
        selected_change = st.sidebar.selectbox(
            "Choose a change to analyze:",
            change_options,
            key="advanced_analysis_selector"
        )
        
        if selected_change:
            # Find the corresponding row
            selected_index = change_options.index(selected_change)
            selected_change_row = impact_results.head(20).iloc[selected_index]
            
            # Store selected change in session state
            st.session_state.selected_change_row = selected_change_row
    
    
    # Visualizations Section
    st.header("📊 Visualizations: See the Story!")
    
    st.subheader("📈 How Big Are the Week-over-Week Changes?")
    st.markdown("This graph shows how often different sizes of week-over-week changes happen across all your data.")
    
    # Metric selector for WoW Change Distribution
    selected_metric_for_viz = st.selectbox("Select metric for WoW Change Distribution:", st.session_state.metrics)
    
    if selected_metric_for_viz:
        metric_results = hierarchical_results[hierarchical_results["Metric"] == selected_metric_for_viz]
        if not metric_results.empty:
            fig = px.histogram(
                metric_results,
                x="Latest_WoW_Change",
                nbins=20,
                title=f"Distribution of Week-over-Week Changes for {selected_metric_for_viz}",
                labels={"Latest_WoW_Change": "Week-over-Week Change (%)", "count": "Frequency"}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Change")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data available for {selected_metric_for_viz}")
    
    # Impact vs Change Scatter Plot
    st.subheader("💥 Impact vs Change: What's Worth Your Attention?")
    st.markdown("This shows the relationship between the size of change and business impact.")
    
    if not impact_results.empty:
        fig2 = px.scatter(
            impact_results.head(50),  # Top 50 for readability
            x="Latest_WoW_Change",
            y="Impact",
            color="Metric",
            size="Latest_Value",
            hover_data=["Dimension_Combination", "Level"],
            title="Business Impact vs Week-over-Week Change",
            labels={"Latest_WoW_Change": "Week-over-Week Change (%)", "Impact": "Business Impact Score"}
        )
        fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Change")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Level Distribution
    st.subheader("🎯 Analysis Level Distribution")
    st.markdown("Shows how many findings come from each level of analysis.")
    
    level_dist = hierarchical_results['Level'].value_counts().reset_index()
    level_dist.columns = ['Level', 'Count']
    
    fig3 = px.bar(
        level_dist,
        x='Level',
        y='Count',
        title="Number of Findings by Analysis Level",
        labels={"Count": "Number of Findings", "Level": "Analysis Level"}
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    
    # Display advanced analysis if a change is selected
    if 'selected_change_row' in st.session_state:
        selected_change_row = st.session_state.selected_change_row
        
        st.header("🔍 Root Cause Analysis: The Full Story!")
        st.markdown(f"Deep diving into **{selected_change_row['Metric']}** changes in **{selected_change_row['Dimension_Combination']}**")
        
        # Multi-Dimensional Breakdown
        # st.subheader("🔍 Multi-Dimensional Breakdown")
        multi_dim_narrative = perform_multi_dimensional_breakdown_advanced(
            st.session_state.full_df_for_visualizations, 
            st.session_state.dimensions, 
            selected_change_row['Metric'], 
            st.session_state.date_column,
            selected_change_row['Dimension_Combination']
        )
        st.markdown(multi_dim_narrative)
        
        # Cross-Metric Impact Analysis
        # st.subheader("🔗 Cross-Metric Impact Analysis")
        cross_metric_narrative = perform_cross_metric_impact_analysis_advanced(
            st.session_state.full_df_for_visualizations,
            st.session_state.metrics,
            selected_change_row['Metric'],
            st.session_state.date_column,
            selected_change_row['Dimension_Combination']
        )
        st.markdown(cross_metric_narrative)
        
        # Business Context and Recommendations
        ai_powered_text = " (powered by AI)" if use_llm else ""
        st.subheader(f"💡 Business Context and Recommendations{ai_powered_text}")
        
        if use_llm:
            # AI-Enhanced Analysis
            try:
                # Create summary of full dataframe for LLM context
                full_df_summary = create_dataframe_summary(st.session_state.full_df_for_visualizations)
                
                ai_analysis = generate_enhanced_root_cause_analysis(
                    selected_change_row,
                    multi_dim_narrative,
                    cross_metric_narrative,
                    hierarchical_results,
                    full_df_summary,
                    st.session_state.full_df_for_visualizations,
                    st.session_state.date_column
                )
                st.markdown(ai_analysis)
            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")
                # Fallback to standard business context
                business_context = get_business_context(
                    selected_change_row['Metric'],
                    selected_change_row['Latest_WoW_Change']
                )
                st.markdown(business_context)
        else:
            # Standard Business Context
            business_context = get_business_context(
                selected_change_row['Metric'],
                selected_change_row['Latest_WoW_Change']
            )
            st.markdown(business_context)

    # System Integration & Executive Summary Section
    st.header("📋 System Integration & Executive Summary")
    st.markdown("Generate structured outputs for system integration and executive reporting.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate JSON Output"):
            json_output = generate_structured_json_output(
                hierarchical_results, 
                impact_results, 
                masked_issues, 
                st.session_state.date_column,
                st.session_state.full_df_for_visualizations
            )
            
            st.subheader("🔗 Structured JSON for System Integration")
            st.json(json_output)
            
            # Download button for JSON
            import json
            json_str = json.dumps(json_output, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name="analysis_results.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📝 Generate Executive Report"):
            executive_report = generate_executive_summary_report(
                hierarchical_results,
                impact_results,
                masked_issues,
                business_weights,
                use_llm
            )
            
            st.subheader("📄 Executive Summary Report")
            with st.expander("View Full Report", expanded=True):
                st.markdown(executive_report)
            
            # Download button for report
            st.download_button(
                label="📥 Download Executive Report",
                data=executive_report,
                file_name="executive_summary.md",
                mime="text/markdown"
            )

    

    st.markdown("---")
    st.markdown("### 🎉 Analysis Complete!")
    st.markdown("You now have a comprehensive view of what's changing in your business data. Use the insights above to make data-driven decisions!")

