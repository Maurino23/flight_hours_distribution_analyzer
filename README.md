# ✈️ Flight Hours Distribution Analyzer

Comprehensive crew flight hours analysis system with two analysis modes:
- **Monthly Analysis** (12 Consecutive Months)
- **12 Consecutive Months Analysis** (Company Standardization)

---

## 🚀 Installation

### 1. Install Required Libraries

```bash
pip install streamlit pandas numpy plotly openpyxl
```

### 2. Save Complete Code

Gabungkan semua sections (Section 1-7) ke dalam satu file bernama `app.py`

**Urutan menggabungkan:**
1. Section 1: Imports & Config
2. Section 2: Monthly Helper Functions
3. Section 3: 12 Consecutive Months Helper Functions
4. Section 4: Main App Monthly Interface
5. Section 5: 12 Consecutive Months Interface Part 1
6. Section 6: 12 Consecutive Months Interface Part 2
7. Section 7: Main Function & Runner

### 3. Run Application

```bash
streamlit run app.py
```

---

## 📋 How to Use

### **Monthly Analysis Mode**

#### Required Files:
1. **Combined Reports Monthly** (.xlsx)
   - Must contain sheet: `Standardized_Company`
   - Contains crew category and status data

2. **Crew Consecutive Year Flight Hours** (.xlsx)
   - Header starts at row 2
   - Contains flight hours data for 12 months

#### Steps:
1. Select "Monthly Analysis" mode
2. Upload both required files
3. Click "Process Monthly Data"
4. Review visualizations and data
5. Download Excel report (4 sheets)

#### Output Sheets:
- Complete Detailed Data
- Productivity by Company
- Period Analysis
- Distribution Report

---

### **12 Consecutive Months Analysis Mode**

#### Required Files:
1. **Flight Hours File** (.xlsx)
   - Crew Readiness file
   - Header on row 2
   - Required columns: Crew ID, Company, Flight Hours, Rank, Crew Status, Crew Category, Period

2. **Roster File** (.xlsx)
   - CR ALL AOC roster file
   - Header on row 2
   - Must contain Crew ID and date columns (1-31)

#### Steps:
1. Select "12 Consecutive Months Analysis" mode
2. Upload both required files
3. Click "Process 12 Months Data"
4. Review company standardization results
5. Analyze productivity distributions
6. Download Excel report (4 sheets)

#### Output Sheets:
- Standardized Company
- Standardized Flight Hours
- Crew Analysis
- Productivity Report

---

## 📊 Features

### Common Features:
- ✅ Interactive data upload
- ✅ Real-time data processing
- ✅ Key performance indicators (KPIs)
- ✅ Interactive visualizations (Plotly)
- ✅ Search and filter capabilities
- ✅ Excel export with multiple sheets

### Monthly Analysis Features:
- Flight hours distribution analysis
- Productivity status calculation (105%/95% thresholds)
- Period-based analysis
- Company and rank breakdowns

### 12 Consecutive Months Features:
- Double company standardization
- Roster-based company determination
- Crew activity analysis
- Productivity status calculation (110%/90% thresholds)
- Before/after comparison charts

---

## 🎨 Visualizations

### Monthly Analysis:
1. Productivity Status Pie Chart
2. Crew Distribution by Company (Bar)
3. Flight Hours Distribution (Histogram)
4. Average Hours Comparison (Bar)

### 12 Consecutive Months:
1. Company Distribution Before/After (Bar)
2. Productivity Status Distribution (Bar)
3. Productivity Status Proportion (Pie)
4. Productivity by Company (Stacked Bar)

---

## 📝 Notes

### Productivity Status:
- **OVER PROD**: Flight hours > threshold (105% monthly, 110% consecutive)
- **PROD**: Flight hours within threshold range
- **LOWER PROD**: Flight hours < threshold (95% monthly, 90% consecutive)

### File Requirements:
- All files must be Excel format (.xlsx or .xls)
- Headers must be on specified rows
- Required columns must be present
- Date format should be consistent

---

## 🆘 Troubleshooting

### Common Issues:

**1. Missing Column Error**
- Verify Excel file has all required columns
- Check header row position

**2. Processing Error**
- Ensure files are not corrupted
- Check file format (.xlsx or .xls)
- Verify data types are correct

**3. Visualization Not Showing**
- Refresh the page
- Check if data was processed successfully
- Verify Plotly is installed correctly

---

## 💡 Tips

1. **Large Files**: Processing may take time for large datasets
2. **Filters**: Use search and filters to focus on specific crew members
3. **Export**: Download reports for offline analysis
4. **Comparison**: Use both modes for comprehensive analysis

---

## 📞 Support

For issues or questions, please check:
1. File format and structure
2. Required columns are present
3. Data types are correct
4. All dependencies are installed

---

## 🔄 Version

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Python**: 3.7+  
**Streamlit**: 1.28+

---

## 📜 License

Internal use only - Crew Management & Operations Analysis System
