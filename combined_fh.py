import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import re
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Hours Distribution Analyzer",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    .mode-selector {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================================
# HELPER FUNCTIONS - MONTHLY ANALYSIS
# ===================================

def convert_time_to_decimal(time_str):
    """Convert HH:MM format to decimal"""
    if pd.isna(time_str) or time_str == '':
        return 0.0
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            hours = float(parts[0])
            minutes = float(parts[1])
            return round(hours + (minutes / 60), 2)
        else:
            return float(time_str)
    except:
        return 0.0

def categorize_rank(rank):
    """Categorize rank into COCKPIT or CABIN"""
    if pd.isna(rank):
        return 'CABIN'
    rank_upper = str(rank).upper()
    cockpit_ranks = ['CPT', 'FO', 'CPT/FO']
    return 'COCKPIT' if rank_upper in cockpit_ranks else 'CABIN'

def determine_productivity_status_monthly(row):
    """Determine productivity status for monthly"""
    flight_hours = row['Flight Hours (Float)']
    upper_limit = row['UPPER LIMIT (105%*AVG)']
    lower_limit = row['LOWER LIMIT (95%*AVG)']
    
    if flight_hours > upper_limit:
        return 'OVER PROD'
    elif flight_hours < lower_limit:
        return 'LOWER PROD'
    else:
        return 'PROD'

def process_monthly_data(standardized_df, year_flight_hours_df):
    """Process data for monthly analysis"""
    with st.spinner('🔄 Processing monthly data...'):
        # Step 1: Create crew mapping
        crew_mapping = standardized_df.groupby('Crew ID').agg({
            'Crew Category': lambda x: x.mode()[0] if not x.mode().empty else '-',
            'Crew Status': lambda x: x.mode()[0] if not x.mode().empty else '-'
        }).reset_index()
        
        # Step 2: Merge
        year_flight_hours_df = year_flight_hours_df.merge(
            crew_mapping,
            left_on='ID',
            right_on='Crew ID',
            how='left'
        )
        
        year_flight_hours_df = year_flight_hours_df.drop(columns=['Crew ID'])
        
        # Step 3: Fill missing values
        year_flight_hours_df['Crew Category'] = year_flight_hours_df['Crew Category'].fillna('-')
        year_flight_hours_df['Crew Status'] = year_flight_hours_df['Crew Status'].fillna('-')
        
        # Step 4: Convert flight hours
        year_flight_hours_df['Flight Hours (Float)'] = year_flight_hours_df['FLIGHT HOURS'].apply(convert_time_to_decimal)
        
        # Step 5: Create Actual Rank
        year_flight_hours_df['Actual Rank'] = year_flight_hours_df['RANK'].apply(categorize_rank)
        
        # Step 6: Calculate AVG MONTHLY
        grouping_cols = ['COMPANY', 'Actual Rank', 'PERIOD (UTC TIME)', 'Crew Status']
        avg_monthly = year_flight_hours_df.groupby(grouping_cols)['Flight Hours (Float)'].transform('mean')
        year_flight_hours_df['AVG MONTHLY'] = avg_monthly.round(2)
        
        # Step 7: Calculate limits
        year_flight_hours_df['UPPER LIMIT (105%*AVG)'] = (year_flight_hours_df['AVG MONTHLY'] * 1.05).round(2)
        year_flight_hours_df['LOWER LIMIT (95%*AVG)'] = (year_flight_hours_df['AVG MONTHLY'] * 0.95).round(2)
        
        # Step 8: Determine productivity status
        year_flight_hours_df['PRODUCTIVITY STATUS'] = year_flight_hours_df.apply(
            determine_productivity_status_monthly,
            axis=1
        )
        
        # Step 9: Sort data
        crew_status_order = {'Ready Crew': 0, 'Not Ready Crew': 1}
        crew_category_order = {'Crew Strength': 0, 'Non Crew Strength': 1}
        actual_rank_order = {'COCKPIT': 0, 'CABIN': 1}
        
        year_flight_hours_df['_crew_status_sort'] = year_flight_hours_df['Crew Status'].map(crew_status_order)
        year_flight_hours_df['_crew_category_sort'] = year_flight_hours_df['Crew Category'].map(crew_category_order)
        year_flight_hours_df['_actual_rank_sort'] = year_flight_hours_df['Actual Rank'].map(actual_rank_order)
        
        year_flight_hours_df = year_flight_hours_df.sort_values(
            by=['COMPANY', '_crew_status_sort', '_crew_category_sort', '_actual_rank_sort', 'Flight Hours (Float)'],
            ascending=[True, True, True, True, False]
        ).reset_index(drop=True)
        
        year_flight_hours_df = year_flight_hours_df.drop(columns=['_crew_status_sort', '_crew_category_sort', '_actual_rank_sort'])
        
        return year_flight_hours_df

def create_monthly_distribution_report(year_flight_hours_df):
    """Create distribution report for monthly"""
    ready_crew_df = year_flight_hours_df[year_flight_hours_df['Crew Status'] == 'Ready Crew'].copy()
    all_periods = sorted(ready_crew_df['PERIOD (UTC TIME)'].unique())
    companies = ['JT', 'ID', 'IU', 'IW']
    ranks = ['COCKPIT', 'CABIN']
    
    report_data = []
    
    for company in companies:
        for rank in ranks:
            upper_row = {'COMPANY': company, 'Rank': rank, 'Metric': 'UPPER 5%'}
            lower_row = {'COMPANY': company, 'Rank': rank, 'Metric': 'LOWER 5%'}
            
            for period in all_periods:
                filtered_df = ready_crew_df[
                    (ready_crew_df['COMPANY'] == company) &
                    (ready_crew_df['Actual Rank'] == rank) &
                    (ready_crew_df['PERIOD (UTC TIME)'] == period)
                ]
                
                total_ready_crew = len(filtered_df)
                
                if total_ready_crew > 0:
                    over_prod_count = len(filtered_df[filtered_df['PRODUCTIVITY STATUS'] == 'OVER PROD'])
                    upper_pct = (over_prod_count / total_ready_crew) * 100
                    upper_row[period] = f"{upper_pct:.2f}%"
                    
                    lower_prod_count = len(filtered_df[filtered_df['PRODUCTIVITY STATUS'] == 'LOWER PROD'])
                    lower_pct = (lower_prod_count / total_ready_crew) * 100
                    lower_row[period] = f"{lower_pct:.2f}%"
                else:
                    upper_row[period] = "-"
                    lower_row[period] = "-"
            
            report_data.append(upper_row)
            report_data.append(lower_row)
    
    report_df = pd.DataFrame(report_data)
    period_columns = [col for col in report_df.columns if col not in ['COMPANY', 'Rank', 'Metric']]
    final_columns = ['COMPANY', 'Rank', 'Metric'] + sorted(period_columns)
    report_df = report_df[final_columns]
    
    return report_df

def new_to_excel_consecutive(processed_df, report_df):
    """Export consecutive data to Excel with 4 sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Complete Detailed Data
        processed_df.to_excel(writer, sheet_name='Complete Detailed Data', index=False)
        
        # Sheet 2: Productivity by Company
        prod_by_company = processed_df.groupby(['COMPANY', 'Actual Rank', 'PRODUCTIVITY STATUS']).size().reset_index(name='Count')
        prod_pivot = prod_by_company.pivot_table(
            index=['COMPANY', 'Actual Rank'],
            columns='PRODUCTIVITY STATUS',
            values='Count',
            fill_value=0
        ).reset_index()
        prod_pivot.to_excel(writer, sheet_name='Productivity by Company', index=False)
        
        # Sheet 3: Period Analysis
        period_analysis = processed_df.groupby(['PERIOD (UTC TIME)', 'COMPANY', 'Actual Rank', 'PRODUCTIVITY STATUS']).size().reset_index(name='Count')
        period_analysis.to_excel(writer, sheet_name='Period Analysis', index=False)
        
        # Sheet 4: Distribution Report
        report_df.to_excel(writer, sheet_name='Distribution Report', index=False)
    
    return output.getvalue()

# ===================================
# HELPER FUNCTIONS - 12 CONSECUTIVE MONTHS
# ===================================

def is_double_company(company_str):
    """Check if company string is double company"""
    if pd.isna(company_str):
        return False
    return '/' in str(company_str)

def get_first_company(company_str):
    """Extract first company from double company string"""
    if pd.isna(company_str):
        return None
    company_str = str(company_str).strip()
    first_company = company_str.split('/')[0].strip()
    return first_company

def extract_flight_numbers(activity_str, company_codes):
    """Extract flight numbers from activity string"""
    if pd.isna(activity_str) or activity_str in ['', '-', 'OFF', 'DO', 'GDO']:
        return []
    
    activity_str = str(activity_str)
    flight_parts = activity_str.split('/')
    valid_companies = []
    
    codes_pattern = '|'.join(company_codes)
    
    for part in flight_parts:
        part = part.strip()
        if '(D)' in part or '(d)' in part:
            continue
        match = re.match(f'^({codes_pattern})\\d+', part)
        if match:
            valid_companies.append(match.group(1))
    
    return valid_companies

def determine_company_for_day(activity_str, company_codes):
    """Determine company for single day"""
    flight_codes = extract_flight_numbers(activity_str, company_codes)
    
    if not flight_codes:
        return None
    
    counter = Counter(flight_codes)
    most_common = counter.most_common()
    
    if len(most_common) == 1:
        return most_common[0][0]
    
    max_count = most_common[0][1]
    tied_companies = [comp for comp, count in most_common if count == max_count]
    
    if len(tied_companies) > 1:
        return '&'.join(sorted(tied_companies))
    
    return most_common[0][0]

def analyze_crew_roster(roster_df, date_columns, flight_hour_df, company_codes):
    """Analyze roster for each crew"""
    crew_analysis = {}
    crew_company_map = dict(zip(flight_hour_df['Crew ID'], flight_hour_df['Company']))
    
    for idx, row in roster_df.iterrows():
        crew_id = row['Crew ID']
        company_original = crew_company_map.get(crew_id)
        
        daily_companies = []
        daily_breakdown = Counter()
        
        for date_col in date_columns:
            if date_col in row.index:
                activity = row[date_col]
                company = determine_company_for_day(activity, company_codes)
                
                if company:
                    daily_companies.append(company)
                    daily_breakdown[company] += 1
        
        if daily_breakdown:
            most_common = daily_breakdown.most_common()
            max_count = most_common[0][1]
            tied_companies = [comp for comp, count in most_common if count == max_count]
            
            if len(tied_companies) > 1:
                standardized_company = get_first_company(company_original)
            else:
                standardized_company = most_common[0][0]
        else:
            standardized_company = get_first_company(company_original)
        
        crew_analysis[crew_id] = {
            'standardized_company': standardized_company,
            'daily_breakdown': dict(daily_breakdown),
            'total_working_days': sum(daily_breakdown.values()),
            'company_original': company_original
        }
    
    return crew_analysis

def standardize_flight_hours(flight_hour_df, crew_analysis):
    """Update Company column based on crew analysis"""
    df = flight_hour_df.copy()
    df['Company_Original'] = df['Company']
    df['Is_Double_Company'] = df['Company'].apply(is_double_company)
    
    crew_company_mapping = {
        crew_id: analysis['standardized_company']
        for crew_id, analysis in crew_analysis.items()
        if analysis['standardized_company'] is not None
    }
    
    df['Company_Standardized'] = df.apply(
        lambda row: crew_company_mapping.get(row['Crew ID'], row['Company'])
        if row['Is_Double_Company']
        else row['Company'],
        axis=1
    )
    
    df['Company'] = df['Company_Standardized']
    df.drop(['Company_Standardized', 'Is_Double_Company'], axis=1, inplace=True)
    
    return df

def determine_productivity_status_consecutive(row, upper_threshold, lower_threshold):
    """Determine productivity status for 12 consecutive months"""
    flight_hours = row['Flight Hours (Float)']
    avg_monthly = row['AVG MONTHLY']
    upper_limit = avg_monthly * (upper_threshold / 100)
    lower_limit = avg_monthly * (lower_threshold / 100)
    
    if flight_hours > upper_limit:
        return 'OVER PROD'
    elif flight_hours < lower_limit:
        return 'LOWER PROD'
    else:
        return 'PROD'

def create_consecutive_productivity_report(standardized_df):
    """Create productivity report for 12 consecutive months"""
    ready_crew_df = standardized_df[standardized_df['Crew Status'] == 'Ready Crew'].copy()
    all_periods = sorted(ready_crew_df['Period'].unique())
    
    companies = ready_crew_df['Company'].unique()
    ranks = ['COCKPIT', 'CABIN']
    
    report_data = []
    
    for company in companies:
        for rank in ranks:
            upper_row = {'Company': company, 'Rank': rank, 'Metric': 'UPPER 10%'}
            lower_row = {'Company': company, 'Rank': rank, 'Metric': 'LOWER 10%'}
            
            for period in all_periods:
                filtered_df = ready_crew_df[
                    (ready_crew_df['Company'] == company) &
                    (ready_crew_df['Actual Rank'] == rank) &
                    (ready_crew_df['Period'] == period)
                ]
                
                total_ready_crew = len(filtered_df)
                
                if total_ready_crew > 0:
                    over_prod_count = len(filtered_df[filtered_df['PRODUCTIVITY STATUS'] == 'OVER PROD'])
                    upper_pct = (over_prod_count / total_ready_crew) * 100
                    upper_row[period] = f"{upper_pct:.2f}%"
                    
                    lower_prod_count = len(filtered_df[filtered_df['PRODUCTIVITY STATUS'] == 'LOWER PROD'])
                    lower_pct = (lower_prod_count / total_ready_crew) * 100
                    lower_row[period] = f"{lower_pct:.2f}%"
                else:
                    upper_row[period] = "-"
                    lower_row[period] = "-"
            
            report_data.append(upper_row)
            report_data.append(lower_row)
    
    report_df = pd.DataFrame(report_data)
    period_columns = [col for col in report_df.columns if col not in ['Company', 'Rank', 'Metric']]
    final_columns = ['Company', 'Rank', 'Metric'] + sorted(period_columns)
    report_df = report_df[final_columns]
    
    return report_df

def new_to_excel_monthly(standardized_df_raw, standardized_df, crew_analysis_df, report_df):
    """Export monthly data to Excel with 4 sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        standardized_df_raw.to_excel(writer, sheet_name='Standardized_Company', index=False)
        standardized_df.to_excel(writer, sheet_name='Standardized_Flight_Hours', index=False)
        crew_analysis_df.to_excel(writer, sheet_name='Crew_Analysis', index=False)
        report_df.to_excel(writer, sheet_name='Productivity_Report', index=False)
    
    return output.getvalue()

# ===================================
# MAIN APP - 12 CONSECUTIVE MONTHS
# ===================================

def new_render_consecutive_analysis():
    """Render 12 CONSECUTIVE MONTHS Interface"""
    st.header("📅 12 CONSECUTIVE MONTHS Mode")
    st.info("📊 Analyze flight hours distribution for 12 consecutive months period")
    
    # File Upload
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader(
            "**1. Monthly Report Flight Hours**",
            type=['xlsx', 'xls'],
            key='monthly_file1',
            help="Upload monthly_report_fh_distribution.xlsx (Sheet: Standardized_Company)"
        )
    
    with col2:
        file2 = st.file_uploader(
            "**2. Crew Consecutive Year Flight Hours**",
            type=['xlsx', 'xls'],
            key='monthly_file2',
            help="Upload Crew Consecutive Year Flight Hours.xlsx (Header on row 2)"
        )
    
    # Process Button
    if file1 and file2:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_button = st.button("🚀 Process Consecutive Data", type="primary", use_container_width=True)
    else:
        process_button = False
        st.warning("👆 Please upload both Excel files to continue")
    
    # Process data
    if process_button:
        try:
            standardized_df = pd.read_excel(file1, sheet_name='Standardized_Company')
            year_flight_hours_df = pd.read_excel(file2, header=1)
            
            processed_df = process_monthly_data(standardized_df, year_flight_hours_df)
            report_df = create_monthly_distribution_report(processed_df)
            
            st.session_state['monthly_processed_df'] = processed_df
            st.session_state['monthly_report_df'] = report_df
            st.session_state['monthly_processed'] = True
            
            st.success("✅ Consecutive data processed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.exception(e)
    
    # Display Results
    if 'monthly_processed' in st.session_state and st.session_state['monthly_processed']:
        df = st.session_state['monthly_processed_df']
        report_df = st.session_state['monthly_report_df']
        
        st.markdown("---")
        
        # Key Metrics
        st.subheader("📊 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Crew", f"{len(df):,}")
        with col2:
            ready_crew = len(df[df['Crew Status'] == 'Ready Crew'])
            st.metric("Ready Crew", f"{ready_crew:,}")
        with col3:
            avg_hours = df['Flight Hours (Float)'].mean()
            st.metric("Avg Flight Hours", f"{avg_hours:.1f}")
        with col4:
            over_prod_pct = (len(df[df['PRODUCTIVITY STATUS'] == 'OVER PROD']) / len(df)) * 100
            st.metric("Over Production", f"{over_prod_pct:.1f}%")
        with col5:
            lower_prod_pct = (len(df[df['PRODUCTIVITY STATUS'] == 'LOWER PROD']) / len(df)) * 100
            st.metric("Lower Production", f"{lower_prod_pct:.1f}%")
        
        st.markdown("---")
        
        # Company Breakdown
        st.subheader("🏢 Company Breakdown")
        company_stats = df.groupby(['COMPANY', 'Actual Rank']).agg({
            'ID': 'count',
            'Flight Hours (Float)': 'mean'
        }).reset_index()
        company_stats.columns = ['Company', 'Rank', 'Total Crew', 'Avg Flight Hours']
        company_stats['Avg Flight Hours'] = company_stats['Avg Flight Hours'].round(2)
        
        st.dataframe(company_stats, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prod_counts = df['PRODUCTIVITY STATUS'].value_counts()
            fig = px.pie(
                values=prod_counts.values,
                names=prod_counts.index,
                title="Productivity Status Distribution",
                color=prod_counts.index,
                color_discrete_map={
                    'PROD': '#10b981',
                    'OVER PROD': '#ef4444',
                    'LOWER PROD': '#f59e0b'
                },
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            crew_by_company = df.groupby(['COMPANY', 'Actual Rank']).size().reset_index(name='Count')
            fig = px.bar(
                crew_by_company,
                x='COMPANY',
                y='Count',
                color='Actual Rank',
                title="Crew Distribution by Company",
                barmode='group',
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Flight Hours Distribution
        fig = px.histogram(
            df,
            x='Flight Hours (Float)',
            color='PRODUCTIVITY STATUS',
            title="Distribution of Flight Hours",
            nbins=50,
            color_discrete_map={
                'PROD': '#10b981',
                'OVER PROD': '#ef4444',
                'LOWER PROD': '#f59e0b'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average Flight Hours
        avg_hours_df = df.groupby(['COMPANY', 'Actual Rank'])['Flight Hours (Float)'].mean().reset_index()
        fig = px.bar(
            avg_hours_df,
            x='COMPANY',
            y='Flight Hours (Float)',
            color='Actual Rank',
            title="Average Flight Hours by Company and Rank",
            barmode='group',
            text_auto='.1f'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed Data
        st.subheader("📋 Detailed Crew Data")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("🔍 Search by Name or ID", "", key='monthly_search')
        with col2:
            company_filter = st.selectbox("Filter by Company", ['ALL'] + list(df['COMPANY'].unique()), key='monthly_company')
        with col3:
            rank_filter = st.selectbox("Filter by Rank", ['ALL'] + list(df['Actual Rank'].unique()), key='monthly_rank')
        
        # Apply filters
        filtered_df = df.copy()
        if search_term:
            filtered_df = filtered_df[
                filtered_df['NAME'].str.contains(search_term, case=False, na=False) |
                filtered_df['ID'].astype(str).str.contains(search_term, case=False, na=False)
            ]
        if company_filter != 'ALL':
            filtered_df = filtered_df[filtered_df['COMPANY'] == company_filter]
        if rank_filter != 'ALL':
            filtered_df = filtered_df[filtered_df['Actual Rank'] == rank_filter]
        
        display_cols = [
            'ID', 'NAME', 'COMPANY', 'RANK', 'Actual Rank', 'Crew Status',
            'Flight Hours (Float)', 'AVG MONTHLY', 'PRODUCTIVITY STATUS'
        ]
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True, height=400)
        st.caption(f"Showing {len(filtered_df):,} of {len(df):,} crew members")
        
        st.markdown("---")
        
        # Distribution Report
        st.subheader("📈 Distribution Report (Ready Crew)")
        st.dataframe(report_df, use_container_width=True, hide_index=True, height=400)
        
        st.info("""
        **📌 Report Notes:**
        - **UPPER 5%**: Percentage of crew with flight hours > 105% of average (OVER PROD)
        - **LOWER 5%**: Percentage of crew with flight hours < 95% of average (LOWER PROD)
        - Report includes Ready Crew only
        """)
        
        st.markdown("---")
        
        # Download Button
        st.subheader("📥 Download Complete Report")
        excel_data = new_to_excel_consecutive(df, report_df)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Consecutive Report (Excel)",
                data=excel_data,
                file_name=f"consecutive_report_fh_distribution{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        
        st.caption("📊 Contains: Complete Detailed Data • Productivity by Company • Period Analysis • Distribution Report")

# ===================================
# MAIN APP - MONTHLY (PART 1)
# ===================================

def new_render_monthly_analysis_part1():
    """Render MONTHLY Analysis Interface - Part 1: Upload & Processing"""
    st.header("📆 MONTHLY Analysis Mode")
    st.info("📊 Standardize crew companies and analyze flight hours productivity for a month")
    
    # Default parameters
    upper_threshold = 110
    lower_threshold = 90
    company_codes = ['JT', 'ID', 'IU', 'IW']
    
    # File Upload
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader(
            "**1. Flight Hours File**",
            type=['xlsx', 'xls'],
            key='consecutive_file1',
            help="Upload Crew Readiness file (Header on row 2)"
        )
    
    with col2:
        file2 = st.file_uploader(
            "**2. Roster File**",
            type=['xlsx', 'xls'],
            key='consecutive_file2',
            help="Upload CR ALL AOC roster file (Header on row 2)"
        )
    
    # Process Button
    if file1 and file2:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_button = st.button("🚀 Process Monthly Data", type="primary", use_container_width=True)
    else:
        process_button = False
        st.warning("👆 Please upload both Excel files to continue")
    
    # Process data
    if process_button:
        try:
            with st.spinner("🔄 Processing files... Please wait."):
                # Load data
                flight_hour_df = pd.read_excel(file1, header=1)
                roster_df = pd.read_excel(file2, header=1)
                
                # Validation
                required_flight_cols = ['Crew ID', 'Company', 'Flight Hours', 'Rank', 'Crew Status', 'Crew Category', 'Period']
                missing_cols = [col for col in required_flight_cols if col not in flight_hour_df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns in Flight Hours file: {', '.join(missing_cols)}")
                    return
                
                if 'Crew ID' not in roster_df.columns:
                    st.error("❌ Missing 'Crew ID' column in Roster file")
                    return
                
                # Identify date columns
                date_columns = [i for i in range(1, 32) if i in roster_df.columns]
                
                # Filter double company crews
                double_company_crews = flight_hour_df[
                    flight_hour_df['Company'].apply(is_double_company)
                ]['Crew ID'].unique()
                
                roster_filtered = roster_df[roster_df['Crew ID'].isin(double_company_crews)]
                
                # Analyze roster
                crew_analysis = analyze_crew_roster(roster_filtered, date_columns, flight_hour_df, company_codes)
                
                # Standardize flight hours
                standardized_df_raw = standardize_flight_hours(flight_hour_df, crew_analysis)
                standardized_df = standardized_df_raw.copy()
                
                # Add supportive columns
                standardized_df['Flight Hours (Float)'] = standardized_df['Flight Hours'].apply(convert_time_to_decimal)
                standardized_df['Actual Rank'] = standardized_df['Rank'].apply(categorize_rank)
                
                # Calculate AVG MONTHLY
                grouping_cols = ['Company', 'Actual Rank', 'Period', 'Crew Status']
                avg_monthly = standardized_df.groupby(grouping_cols)['Flight Hours (Float)'].transform('mean')
                standardized_df['AVG MONTHLY'] = avg_monthly.round(2)
                
                # Calculate limits
                standardized_df['UPPER LIMIT (110%*AVG)'] = (standardized_df['AVG MONTHLY'] * (upper_threshold / 100)).round(2)
                standardized_df['LOWER LIMIT (90%*AVG)'] = (standardized_df['AVG MONTHLY'] * (lower_threshold / 100)).round(2)
                
                # Determine productivity status
                standardized_df['PRODUCTIVITY STATUS'] = standardized_df.apply(
                    lambda row: determine_productivity_status_consecutive(row, upper_threshold, lower_threshold),
                    axis=1
                )
                
                # Sort data
                crew_status_order = {'Ready Crew': 0, 'Not Ready Crew': 1}
                crew_category_order = {'Crew Strength': 0, 'Non Crew Strength': 1}
                actual_rank_order = {'COCKPIT': 0, 'CABIN': 1}
                
                standardized_df['_crew_status_sort'] = standardized_df['Crew Status'].map(crew_status_order)
                standardized_df['_crew_category_sort'] = standardized_df['Crew Category'].map(crew_category_order)
                standardized_df['_actual_rank_sort'] = standardized_df['Actual Rank'].map(actual_rank_order)
                
                standardized_df = standardized_df.sort_values(
                    by=['Company', '_crew_status_sort', '_crew_category_sort', '_actual_rank_sort', 'Flight Hours (Float)'],
                    ascending=[True, True, True, True, False]
                ).reset_index(drop=True)
                
                standardized_df = standardized_df.drop(columns=['_crew_status_sort', '_crew_category_sort', '_actual_rank_sort'])
                
                # Create crew analysis dataframe
                crew_analysis_df = pd.DataFrame.from_dict(crew_analysis, orient='index')
                crew_analysis_df.index.name = 'Crew ID'
                crew_analysis_df.reset_index(inplace=True)
                
                # Create productivity report
                report_df = create_consecutive_productivity_report(standardized_df)
                
                # Calculate changes
                changes = (standardized_df['Company'] != standardized_df['Company_Original']).sum()
                
                # Store in session state
                st.session_state['consecutive_standardized_df_raw'] = standardized_df_raw
                st.session_state['consecutive_standardized_df'] = standardized_df
                st.session_state['consecutive_crew_analysis_df'] = crew_analysis_df
                st.session_state['consecutive_report_df'] = report_df
                st.session_state['consecutive_double_company_crews'] = double_company_crews
                st.session_state['consecutive_changes'] = changes
                st.session_state['consecutive_crew_analysis'] = crew_analysis
                st.session_state['consecutive_processed'] = True
                
                st.success("✅ Monthly data processed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.exception(e)

# ===================================
# MAIN APP - MONTHLY (PART 2)
# Results Display
# ===================================

def new_render_monthly_analysis_part2():
    """Render MONTHLY Analysis Results - Part 2"""
    
    if 'consecutive_processed' in st.session_state and st.session_state['consecutive_processed']:
        
        standardized_df = st.session_state['consecutive_standardized_df']
        standardized_df_raw = st.session_state['consecutive_standardized_df_raw']
        crew_analysis_df = st.session_state['consecutive_crew_analysis_df']
        report_df = st.session_state['consecutive_report_df']
        double_company_crews = st.session_state['consecutive_double_company_crews']
        changes = st.session_state['consecutive_changes']
        crew_analysis = st.session_state['consecutive_crew_analysis']
        
        st.markdown("---")
        
        # Summary Statistics
        st.subheader("📊 Processing Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Records", len(standardized_df))
        with col2:
            st.metric("Double Company", len(double_company_crews))
        with col3:
            st.metric("Records Changed", changes)
        with col4:
            st.metric("Crews Analyzed", len(crew_analysis))
        with col5:
            change_rate = (changes / len(double_company_crews) * 100) if len(double_company_crews) > 0 else 0
            st.metric("Change Rate", f"{change_rate:.1f}%")
        
        st.markdown("---")
        
        # Company Distribution Comparison
        st.subheader("🏢 Company Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Standardization**")
            before_dist = standardized_df['Company_Original'].value_counts().sort_values(ascending=False)
            
            fig_before = px.bar(
                x=before_dist.index,
                y=before_dist.values,
                labels={'x': 'Company', 'y': 'Count'},
                title='Company Distribution - Before',
                color=before_dist.values,
                color_continuous_scale='Blues'
            )
            fig_before.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Company",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_before, use_container_width=True)
            
            with st.expander("View Details"):
                st.dataframe(before_dist.reset_index().rename(columns={'Company_Original': 'Count', 'index': 'Company'}))
        
        with col2:
            st.markdown("**After Standardization**")
            after_dist = standardized_df['Company'].value_counts().sort_values(ascending=False)
            
            fig_after = px.bar(
                x=after_dist.index,
                y=after_dist.values,
                labels={'x': 'Company', 'y': 'Count'},
                title='Company Distribution - After',
                color=after_dist.values,
                color_continuous_scale='Greens'
            )
            fig_after.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Company",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_after, use_container_width=True)
            
            with st.expander("View Details"):
                st.dataframe(after_dist.reset_index().rename(columns={'Company': 'Count', 'index': 'Company'}))
        
        st.markdown("---")
        
        # Productivity Status Distribution
        st.subheader("📈 Productivity Status Distribution")
        
        prod_status = standardized_df['PRODUCTIVITY STATUS'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            over_prod = prod_status.get('OVER PROD', 0)
            st.metric("OVER PROD", over_prod, f"{over_prod/len(standardized_df)*100:.1f}%")
        with col2:
            prod = prod_status.get('PROD', 0)
            st.metric("PROD", prod, f"{prod/len(standardized_df)*100:.1f}%")
        with col3:
            lower_prod = prod_status.get('LOWER PROD', 0)
            st.metric("LOWER PROD", lower_prod, f"{lower_prod/len(standardized_df)*100:.1f}%")
        
        # Plotly Bar Chart
        fig_prod = px.bar(
            x=prod_status.index,
            y=prod_status.values,
            labels={'x': 'Productivity Status', 'y': 'Count'},
            title='Productivity Status Distribution',
            color=prod_status.index,
            color_discrete_map={
                'OVER PROD': '#ef4444',
                'PROD': '#22c55e',
                'LOWER PROD': '#f97316'
            }
        )
        fig_prod.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Status",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_prod, use_container_width=True)
        
        # Pie Chart and Stacked Bar
        col_pie1, col_pie2 = st.columns(2)
        
        with col_pie1:
            fig_pie = px.pie(
                values=prod_status.values,
                names=prod_status.index,
                title='Productivity Status Proportion',
                color=prod_status.index,
                color_discrete_map={
                    'OVER PROD': '#ef4444',
                    'PROD': '#22c55e',
                    'LOWER PROD': '#f97316'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_pie2:
            # Company vs Productivity Status
            company_prod = pd.crosstab(
                standardized_df['Company'],
                standardized_df['PRODUCTIVITY STATUS']
            )
            
            fig_stacked = px.bar(
                company_prod,
                title='Productivity Status by Company',
                labels={'value': 'Count', 'variable': 'Status'},
                color_discrete_map={
                    'OVER PROD': '#ef4444',
                    'PROD': '#22c55e',
                    'LOWER PROD': '#f97316'
                },
                barmode='stack'
            )
            fig_stacked.update_layout(
                height=400,
                xaxis_title="Company",
                yaxis_title="Count",
                legend_title="Status"
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
        
        st.markdown("---")
        
        # Data Preview with Filters
        st.subheader("🔍 Data Preview & Filters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_company = st.selectbox("Company", ['All'] + sorted(standardized_df['Company'].unique().tolist()), key='cons_company')
        with col2:
            filter_rank = st.selectbox("Rank", ['All', 'COCKPIT', 'CABIN'], key='cons_rank')
        with col3:
            filter_period = st.selectbox("Period", ['All'] + sorted(standardized_df['Period'].unique().tolist()), key='cons_period')
        with col4:
            filter_status = st.selectbox("Crew Status", ['All', 'Ready Crew', 'Not Ready Crew'], key='cons_status')
        
        # Apply filters
        filtered_df = standardized_df.copy()
        if filter_company != 'All':
            filtered_df = filtered_df[filtered_df['Company'] == filter_company]
        if filter_rank != 'All':
            filtered_df = filtered_df[filtered_df['Actual Rank'] == filter_rank]
        if filter_period != 'All':
            filtered_df = filtered_df[filtered_df['Period'] == filter_period]
        if filter_status != 'All':
            filtered_df = filtered_df[filtered_df['Crew Status'] == filter_status]
        
        st.info(f"📋 Showing {len(filtered_df)} of {len(standardized_df)} records")
        
        # Display tables
        st.markdown("#### 📋 Standardized Flight Hours")
        st.dataframe(
            filtered_df[['Crew ID', 'Crew Name', 'Company', 'Company_Original', 'Rank', 'Actual Rank', 
                        'Flight Hours (Float)', 'AVG MONTHLY', 'PRODUCTIVITY STATUS']],
            use_container_width=True,
            height=400
        )
        
        st.markdown("#### 👥 Crew Analysis Details")
        st.dataframe(
            crew_analysis_df,
            use_container_width=True,
            height=400
        )
        
        st.markdown("#### 📊 Productivity Report")
        st.dataframe(
            report_df,
            use_container_width=True,
            height=400
        )
        
        st.info("""
        **📌 Report Notes:**
        - **UPPER 10%**: Percentage of crew with flight hours > 110% of average (OVER PROD)
        - **LOWER 10%**: Percentage of crew with flight hours < 90% of average (LOWER PROD)
        - Report includes Ready Crew only
        """)
        
        st.markdown("---")
        
        # Download Button
        st.subheader("📥 Download Complete Report")
        excel_data = new_to_excel_monthly(standardized_df_raw, standardized_df, crew_analysis_df, report_df)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Monthly Report (Excel)",
                data=excel_data,
                file_name=f"monthly_report_fh_distribution_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        
        st.caption("📊 Contains: Standardized Company • Standardized Flight Hours • Crew Analysis • Productivity Report")

# ===================================
# MAIN FUNCTION
# ===================================

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Flight Hours Distribution Analyzer</h1>
        <p>Comprehensive crew flight hours analysis system - Monthly & 12 Consecutive Months</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    st.subheader("📊 Select Analysis Mode")
    
    analysis_mode = st.radio(
        "Choose your analysis type:",
        options=["Monthly Analysis", "12 Consecutive Months Analysis"],
        horizontal=True,
        help="Select the type of analysis you want to perform"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Route to appropriate analysis mode
    if analysis_mode == "Monthly Analysis":
        new_render_monthly_analysis_part1()
        new_render_monthly_analysis_part2()
    else:  # 12 Consecutive Months Analysis
        new_render_consecutive_analysis ()

# ===================================
# RUN APP
# ===================================

if __name__ == "__main__":
    main()