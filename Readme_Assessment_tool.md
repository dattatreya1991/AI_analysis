# 📊 Bicycle Diagnostic Tool

A comprehensive, dataset-agnostic analysis tool for detecting performance changes, identifying business drivers, and generating actionable insights with optional AI-powered enhancements.

## 🎯 Features

### 🔄 Core Analytics
- **Multi-level Analysis**: Week-over-Week (WoW) and Year-over-Year (YoY) changes
- **Hierarchical Analysis**: Drill down through dimension combinations
- **Statistical Significance Testing**: T-tests with p-values
- **Masked Issue Detection**: Hidden problems in segments
- **Impact-Based Prioritization**: Business-weighted importance scoring

### 🤖 AI-Enhanced Features (Optional)
- **AI-Powered Business Insights**: Comprehensive analysis summaries
- **Enhanced Root Cause Analysis**: Deep dive investigations with forecasting
- **Future Predictions**: 30-day forecasts with business impact assessment
- **Strategic Recommendations**: LLM-generated action items

### 📋 System Integration
- **Structured JSON Output**: API-ready format for downstream systems
- **Executive Summary Reports**: Downloadable markdown reports
- **Interactive Visualizations**: Dynamic charts and graphs

## 📋 Prerequisites

- **Python 3.8+** (check with `python3 --version`)
- **pip** package manager
- **Terminal/Command Prompt** access
- **Google Gemini API Key** (optional, for AI features)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Download all files** to a new folder (e.g., `analysis_tool`)
2. **Open terminal** and navigate to the folder:
   ```bash
   cd /path/to/your/analysis_tool
   ```
3. **Run the setup script**:
   ```bash
   chmod +x run_local.sh
   ./run_local.sh
   ```

The script will automatically:
- Create a Python virtual environment
- Install all dependencies
- Start the Streamlit application
- Open it in your browser

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Set Up AI Features (Optional)
If you want AI-enhanced analysis:

1. **Get Google Gemini API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated key

2. **Set Environment Variable**:
   ```bash
   # Option A: Set for current session
   export GEMINI_API_KEY="your_api_key_here"
   
   # Option B: Use the setup helper
   python3 setup_environment.py
   
   # Option C: Add to shell profile (permanent)
   echo 'export GEMINI_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc
   ```

#### Step 4: Run the Application
```bash
streamlit run streamlit_app_production.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 File Structure

```
analysis_tool/
├── streamlit_app_production.py         # Main Streamlit application 
├── analysis_functions_updated.py       # Core analysis functions 
├── llm_integration.py                  # Google Gemini AI integration 
├── report_generation.py                # JSON output and executive reports 
├── business_context.json               # Business context configuration 
├── requirements.txt                    # Python dependencies 
├── run_local.sh                        # Automated setup script 
├── setup_environment.py                # API key setup helper 
├── Synthetic_data_generator_Final.py   # Sample data generator 
├── metric_config.json                  # Metrics config file
└── Readme_Assessment_tool.md           # This documentation file
```

### **Core Application Files:**
- **`streamlit_app_production.py`** - Main application with UI and workflow
- **`analysis_functions_updated.py`** - Statistical analysis and data processing
- **`llm_integration.py`** - AI-powered insights and forecasting
- **`report_generation.py`** - JSON export and executive reporting

### **Configuration Files:**
- **`business_context.json`** - Business rules and context definitions
- **`requirements.txt`** - Python package dependencies
- **`metric_config.json`** - Metrics configuration

### **Setup & Utilities:**
- **`run_local.sh`** - One-command setup and launch script
- **`setup_environment.py`** - Interactive API key configuration
- **`generate_synthetic_data.py`** - Creates sample datasets for testing

## 🧪 How to Use

### 1. Upload Your Data
- Click **"Browse files"** in the sidebar
- Upload a CSV file with:
  - **Date column**: Any date format
  - **Metrics**: Numerical columns (revenue, clicks, conversions, etc.)
  - **Dimensions**: Categorical columns (device_type, geo, campaign, etc.)

### 2. Configure Analysis
- **Date Column**: Select your date column
- **Metrics**: Choose numerical columns to analyze
- **Dimensions**: Select categorical columns for segmentation
- **Threshold**: Set significance threshold (default: 10%)
- **Alpha**: Set statistical significance level (default: 0.05)
- **Business Weights**: Adjust importance of different metrics

### 3. Run Analysis
- Click **"🚀 Run Analysis!"**
- Wait for processing (may take 30-60 seconds for large datasets)
- Review the comprehensive results

### 4. Explore Advanced Features
- **Advanced Analysis**: Select specific changes for deep dive
- **AI Enhancement**: Toggle AI features if API key is configured
- **JSON Output**: Generate structured data for system integration
- **Executive Report**: Download markdown summary for stakeholders

### 5. Generate Sample Data (Optional)
If you want to test with sample data:
```bash
python3 generate_synthetic_data.py
```
This creates sample CSV files for testing the application.

## 🎯 Sample Data Format

Your CSV should look like this:

```csv
date,revenue,clicks,conversions,device_type,geo,campaign
2024-01-01,10000,5000,100,desktop,US,brand
2024-01-01,8000,4000,80,mobile,US,brand
2024-01-02,12000,6000,120,desktop,US,performance
...
```

**Key Requirements:**
- **Date column**: Any standard date format
- **Metrics**: Numerical values (revenue, clicks, etc.)
- **Dimensions**: Categories for segmentation
- **Minimum 2 weeks** of data for meaningful analysis

## 🔧 Troubleshooting

### Common Issues

**1. Python Version Error**
```bash
python3 --version  # Should be 3.8+
```

**2. Permission Denied**
```bash
chmod +x run_local.sh
```

**3. Virtual Environment Issues**
```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Streamlit Not Found**
```bash
source venv/bin/activate
pip install streamlit
```

**5. Google AI Package Missing**
```bash
pip install google-generativeai
```

**6. Port Already in Use**
```bash
streamlit run streamlit_app_production.py --server.port 8502
```

**7. Module Import Errors**
Make sure all files are in the same directory:
- `streamlit_app_production.py`
- `analysis_functions_updated.py`
- `llm_integration.py`
- `report_generation.py`
- `business_context.json`

### Data Issues

**1. Date Column Not Recognized**
- Ensure date column contains valid dates
- Try different date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)

**2. No Significant Changes Found**
- Lower the threshold percentage
- Check if you have enough historical data
- Verify metrics contain numerical values

**3. Analysis Takes Too Long**
- Reduce the number of dimensions
- Filter data to smaller date range
- Check for very large datasets (>1M rows)

## 🎛️ Configuration Options

### Analysis Parameters
- **Threshold**: Minimum change percentage to flag (default: 10%)
- **Alpha**: Statistical significance level (default: 0.05)
- **Top N**: Number of top results to show (default: 5)

### Business Weights
Adjust the relative importance of different metrics:
- **Revenue**: Default weight 1.0
- **Profit**: Default weight 1.2
- **Other metrics**: Default weight 1.0

### AI Settings
- **Toggle**: Enable/disable AI features
- **API Key**: Set via environment variable
- **Model**: Uses Google Gemini 2.5 Pro

## 📊 Understanding Results

### Key Metrics
- **WoW Change**: Week-over-week percentage change
- **YoY Change**: Year-over-year percentage change
- **Impact Score**: Business importance (higher = more critical)
- **P-value**: Statistical significance (< 0.05 = significant)

### Analysis Sections
1. **Performance Summary**: High-level overview
2. **Top/Bottom Performers**: Best and worst changes
3. **Masked Issues**: Hidden segment problems
4. **Visualizations**: Interactive charts
5. **Advanced Analysis**: Deep dive investigations
6. **System Integration**: JSON output and reports

## 🤖 AI Features

### Requirements
- Google Gemini API key
- `google-generativeai` package installed
- Internet connection

### Capabilities
- **Business Insights**: Comprehensive analysis summaries
- **Root Cause Analysis**: Deep investigations with hypotheses
- **Future Forecasting**: 30-day predictions with impact assessment
- **Strategic Recommendations**: Actionable business advice

### Setup
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable: `export GEMINI_API_KEY="your_key"`
3. Toggle AI features in the sidebar

## 📋 System Integration

### JSON Output
Generate structured JSON for API consumption:
```json
{
  "metric": "revenue",
  "change_type": "increase",
  "change_pct": 18.5,
  "period": "2025-W03",
  "dimensions": {...},
  "drivers": {...},
  "estimated_impact": 1250000
}
```

### Executive Reports
Download markdown reports with:
- Top 5 key takeaways
- 3 executive recommendations
- Performance highlights
- Next steps

## 🔄 Updates and Maintenance

### Updating the Tool
1. Download new files
2. Replace existing files
3. Update dependencies: `pip install -r requirements.txt --upgrade`
4. Restart the application

### Performance Optimization
- Use smaller datasets for faster processing
- Limit dimensions to most important ones
- Consider data sampling for very large files

## 🆘 Support

### Getting Help
1. Check troubleshooting section above
2. Verify all files are in correct location
3. Ensure Python and pip are up to date
4. Check terminal output for specific errors

### Common Solutions
- **Import errors**: Check virtual environment activation
- **API errors**: Verify Gemini API key setup
- **Memory issues**: Reduce dataset size or dimensions
- **Performance issues**: Check system resources

### File Verification
Ensure you have all required files:
```bash
ls -la analysis_tool/
# Should show all 10 files listed in File Structure section
```

## 🎉 Success Indicators

When everything works correctly:
- ✅ Streamlit app opens in browser
- ✅ File upload works without errors
- ✅ Analysis completes and shows results
- ✅ Visualizations display properly
- ✅ Advanced features respond correctly
- ✅ Downloads work as expected

---

**Happy analyzing! 🎯📊**

