import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .professional-header {
        background: linear-gradient(135deg, #a8c0ff 0%, #c5a3ff 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .header-subtitle {
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f6f4;
    }
    
    /* Multiselect styling - ochre/tan theme */
    [data-baseweb="tag"] {
        background-color: #d4a574 !important;
        border: 1px solid #b8936a !important;
        color: white !important;
    }
    
    [data-baseweb="tag"] span {
        color: white !important;
    }
    
    /* Remove button (X) styling */
    [data-baseweb="tag"] svg {
        fill: white !important;
    }
    
    /* Sidebar section headers */
    .sidebar-section {
        background: linear-gradient(135deg, #d4a574 0%, #c9965e 100%);
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: white;
        font-weight: 600;
        font-size: 0.95em;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="professional-header">
        <div class="header-title">Clinical Trial Analytics Dashboard</div>
        <div class="header-subtitle">Phase III Multi-Center Efficacy and Safety Analysis</div>
        <div style="margin-top: 15px; opacity: 0.8;">
            Protocol: CT-2022-001 | Indication: Chronic Disease Management
        </div>
    </div>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    sites = pd.read_csv('trial_sites.csv')
    patients = pd.read_csv('trial_patients.csv')
    efficacy = pd.read_csv('trial_efficacy.csv')
    adverse_events = pd.read_csv('trial_adverse_events.csv')
    laboratory = pd.read_csv('trial_laboratory.csv')
    
    # Convert dates
    patients['enrollment_date'] = pd.to_datetime(patients['enrollment_date'])
    patients['dropout_date'] = pd.to_datetime(patients['dropout_date'])
    efficacy['visit_date'] = pd.to_datetime(efficacy['visit_date'])
    adverse_events['onset_date'] = pd.to_datetime(adverse_events['onset_date'])
    adverse_events['resolution_date'] = pd.to_datetime(adverse_events['resolution_date'])
    laboratory['visit_date'] = pd.to_datetime(laboratory['visit_date'])
    
    return sites, patients, efficacy, adverse_events, laboratory

try:
    sites_df, patients_df, efficacy_df, ae_df, labs_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.title("Analysis Filters")

st.sidebar.markdown('<div class="sidebar-section">Treatment Arms</div>', unsafe_allow_html=True)
selected_arms = st.sidebar.multiselect(
    "Select treatment groups to analyze",
    options=patients_df['treatment_arm'].unique().tolist(),
    default=patients_df['treatment_arm'].unique().tolist(),
    label_visibility="collapsed",
    key="treatment_filter"
)

st.sidebar.markdown('<div class="sidebar-section">Research Sites</div>', unsafe_allow_html=True)
selected_sites = st.sidebar.multiselect(
    "Select research sites to include",
    options=sites_df['site_name'].unique().tolist(),
    default=sites_df['site_name'].unique().tolist(),
    label_visibility="collapsed",
    key="sites_filter"
)

# Filter data
if selected_arms:
    patients_filtered = patients_df[patients_df['treatment_arm'].isin(selected_arms)]
else:
    patients_filtered = patients_df

if selected_sites:
    site_ids = sites_df[sites_df['site_name'].isin(selected_sites)]['site_id'].tolist()
    patients_filtered = patients_filtered[patients_filtered['site_id'].isin(site_ids)]

efficacy_filtered = efficacy_df[efficacy_df['patient_id'].isin(patients_filtered['patient_id'])]
ae_filtered = ae_df[ae_df['patient_id'].isin(patients_filtered['patient_id'])]
labs_filtered = labs_df[labs_df['patient_id'].isin(patients_filtered['patient_id'])]

# ========== TRIAL OVERVIEW KPIs ==========
st.subheader("Trial Overview")

col1, col2, col3, col4, col5 = st.columns(5)

total_enrolled = len(patients_filtered)
completed = patients_filtered['completed_trial'].sum()
dropout_rate = ((total_enrolled - completed) / total_enrolled * 100) if total_enrolled > 0 else 0

total_ae = len(ae_filtered)
serious_ae = ae_filtered['is_serious'].sum()

with col1:
    st.metric("Enrolled Patients", total_enrolled,
              delta=f"{len(sites_df)} sites")

with col2:
    st.metric("Completion Rate", f"{(completed/total_enrolled*100):.1f}%",
              delta=f"{completed} completed")

with col3:
    st.metric("Dropout Rate", f"{dropout_rate:.1f}%",
              delta="Within expected range" if dropout_rate < 20 else "Monitor closely",
              delta_color="inverse")

with col4:
    st.metric("Total AEs", total_ae,
              delta=f"{serious_ae} serious")

with col5:
    avg_age = patients_filtered['age'].mean()
    st.metric("Mean Age", f"{avg_age:.1f} yrs",
              delta=f"{patients_filtered['age'].std():.1f} SD")

st.markdown("---")

# ========== ENROLLMENT TRACKER ==========
st.subheader("Enrollment Timeline")

enrollment_by_date = patients_filtered.groupby(
    patients_filtered['enrollment_date'].dt.to_period('M')
).size().reset_index()
enrollment_by_date.columns = ['Month', 'Patients']
enrollment_by_date['Month'] = enrollment_by_date['Month'].astype(str)
enrollment_by_date['Cumulative'] = enrollment_by_date['Patients'].cumsum()

fig_enrollment = go.Figure()

fig_enrollment.add_trace(go.Bar(
    x=enrollment_by_date['Month'],
    y=enrollment_by_date['Patients'],
    name='Monthly Enrollment',
    marker_color='#a8c0ff'
))

fig_enrollment.add_trace(go.Scatter(
    x=enrollment_by_date['Month'],
    y=enrollment_by_date['Cumulative'],
    name='Cumulative Enrollment',
    yaxis='y2',
    line=dict(color='#5dae8b', width=3),
    mode='lines+markers'
))

fig_enrollment.update_layout(
    title='Patient Enrollment Over Time',
    xaxis_title='Month',
    yaxis_title='Patients Enrolled',
    yaxis2=dict(
        title='Cumulative Enrollment',
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_enrollment, use_container_width=True)

st.markdown("---")

# ========== EFFICACY ANALYSIS ==========
st.subheader("Efficacy Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Mean Efficacy Score Over Time**")
    
    # Calculate mean and confidence intervals by treatment arm
    efficacy_summary = efficacy_filtered.merge(
        patients_filtered[['patient_id', 'treatment_arm']], 
        on='patient_id'
    )
    
    efficacy_by_arm = efficacy_summary.groupby(['treatment_arm', 'visit_week']).agg({
        'efficacy_score': ['mean', 'std', 'count']
    }).reset_index()
    
    efficacy_by_arm.columns = ['treatment_arm', 'visit_week', 'mean', 'std', 'count']
    efficacy_by_arm['se'] = efficacy_by_arm['std'] / np.sqrt(efficacy_by_arm['count'])
    efficacy_by_arm['ci_lower'] = efficacy_by_arm['mean'] - 1.96 * efficacy_by_arm['se']
    efficacy_by_arm['ci_upper'] = efficacy_by_arm['mean'] + 1.96 * efficacy_by_arm['se']
    
    fig_efficacy = go.Figure()
    
    colors = {'Placebo': '#e8daef', 'Low Dose': '#d4f1f4', 
              'Medium Dose': '#d5f4e6', 'High Dose': '#fdebd0'}
    
    for arm in efficacy_by_arm['treatment_arm'].unique():
        arm_data = efficacy_by_arm[efficacy_by_arm['treatment_arm'] == arm]
        
        # Add confidence interval
        fig_efficacy.add_trace(go.Scatter(
            x=arm_data['visit_week'],
            y=arm_data['ci_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_efficacy.add_trace(go.Scatter(
            x=arm_data['visit_week'],
            y=arm_data['ci_lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor=colors.get(arm, '#cccccc'),
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add mean line
        fig_efficacy.add_trace(go.Scatter(
            x=arm_data['visit_week'],
            y=arm_data['mean'],
            mode='lines+markers',
            name=arm,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig_efficacy.update_layout(
        title='Efficacy Score by Treatment Arm (with 95% CI)',
        xaxis_title='Week',
        yaxis_title='Mean Efficacy Score',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_efficacy, use_container_width=True)

with col2:
    st.markdown("**Response Rate by Treatment Arm**")
    
    # Get latest assessment for each patient
    latest_efficacy = efficacy_filtered.sort_values('visit_week').groupby('patient_id').tail(1)
    latest_efficacy = latest_efficacy.merge(
        patients_filtered[['patient_id', 'treatment_arm']], 
        on='patient_id'
    )
    
    response_summary = latest_efficacy.groupby(
        ['treatment_arm', 'response_category']
    ).size().reset_index(name='count')
    
    # Calculate percentages
    total_by_arm = response_summary.groupby('treatment_arm')['count'].sum()
    response_summary['percentage'] = response_summary.apply(
        lambda row: (row['count'] / total_by_arm[row['treatment_arm']] * 100), 
        axis=1
    )
    
    fig_response = px.bar(
        response_summary,
        x='treatment_arm',
        y='percentage',
        color='response_category',
        title='Response Distribution by Treatment Arm',
        labels={'percentage': 'Percentage (%)', 'treatment_arm': 'Treatment Arm'},
        color_discrete_map={
            'Complete Response': '#5dae8b',
            'Partial Response': '#a8c0ff',
            'Stable Disease': '#fdebd0',
            'Progressive Disease': '#fadbd8'
        },
        height=400
    )
    
    fig_response.update_layout(barmode='stack')
    st.plotly_chart(fig_response, use_container_width=True)

st.markdown("---")

# ========== WATERFALL PLOT ==========
st.subheader("Best Response Waterfall Plot")

best_response = efficacy_filtered.groupby('patient_id')['change_from_baseline'].max().reset_index()
best_response = best_response.merge(
    patients_filtered[['patient_id', 'treatment_arm']], 
    on='patient_id'
)
best_response = best_response.sort_values('change_from_baseline', ascending=False).reset_index(drop=True)

# Add response category colors
best_response['color'] = best_response['change_from_baseline'].apply(
    lambda x: '#5dae8b' if x >= 50 else '#a8c0ff' if x >= 30 else '#fdebd0' if x >= 10 else '#fadbd8'
)

fig_waterfall = go.Figure()

for arm in best_response['treatment_arm'].unique():
    arm_data = best_response[best_response['treatment_arm'] == arm]
    
    fig_waterfall.add_trace(go.Bar(
        x=arm_data.index,
        y=arm_data['change_from_baseline'],
        name=arm,
        marker_color=arm_data['color'],
        showlegend=False,
        hovertemplate='Patient: %{x}<br>Best Response: %{y:.1f}%<extra></extra>'
    ))

fig_waterfall.add_hline(y=30, line_dash="dash", line_color="green", 
                        annotation_text="Partial Response Threshold")
fig_waterfall.add_hline(y=50, line_dash="dash", line_color="darkgreen", 
                        annotation_text="Complete Response Threshold")

fig_waterfall.update_layout(
    title='Best Response from Baseline (Waterfall Plot)',
    xaxis_title='Patient (sorted by response)',
    yaxis_title='Change from Baseline (%)',
    height=400
)

st.plotly_chart(fig_waterfall, use_container_width=True)

st.markdown("---")

# ========== ADVERSE EVENTS ANALYSIS ==========
st.subheader("Safety Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Adverse Events by Treatment Arm**")
    
    ae_by_arm = ae_filtered.merge(
        patients_filtered[['patient_id', 'treatment_arm']], 
        on='patient_id'
    )
    
    ae_summary = ae_by_arm.groupby(['treatment_arm', 'severity']).size().reset_index(name='count')
    
    fig_ae = px.bar(
        ae_summary,
        x='treatment_arm',
        y='count',
        color='severity',
        title='Adverse Events by Severity',
        labels={'count': 'Number of Events', 'treatment_arm': 'Treatment Arm'},
        color_discrete_map={
            'Mild': '#d5f4e6',
            'Moderate': '#fdebd0',
            'Severe': '#fadbd8'
        },
        height=400
    )
    
    st.plotly_chart(fig_ae, use_container_width=True)

with col2:
    st.markdown("**Most Common Adverse Events**")
    
    ae_types = ae_filtered['ae_type'].value_counts().head(10).reset_index()
    ae_types.columns = ['AE Type', 'Count']
    
    fig_ae_types = px.bar(
        ae_types,
        x='Count',
        y='AE Type',
        orientation='h',
        title='Top 10 Adverse Events',
        color='Count',
        color_continuous_scale='Reds',
        height=400
    )
    
    fig_ae_types.update_layout(showlegend=False)
    st.plotly_chart(fig_ae_types, use_container_width=True)

# AE Heatmap
st.markdown("**Adverse Event Incidence Heatmap**")

ae_heatmap_data = ae_by_arm.groupby(['treatment_arm', 'ae_type']).size().reset_index(name='count')
ae_pivot = ae_heatmap_data.pivot(index='ae_type', columns='treatment_arm', values='count').fillna(0)

fig_ae_heatmap = go.Figure(data=go.Heatmap(
    z=ae_pivot.values,
    x=ae_pivot.columns,
    y=ae_pivot.index,
    colorscale='Reds',
    text=ae_pivot.values,
    texttemplate='%{text:.0f}',
    textfont={"size": 10},
    colorbar=dict(title="Event Count")
))

fig_ae_heatmap.update_layout(
    title='Adverse Event Incidence by Type and Treatment',
    xaxis_title='Treatment Arm',
    yaxis_title='Adverse Event Type',
    height=500
)

st.plotly_chart(fig_ae_heatmap, use_container_width=True)

st.markdown("---")

# ========== DEMOGRAPHICS ==========
st.subheader("Patient Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Age Distribution**")
    
    fig_age = px.histogram(
        patients_filtered,
        x='age',
        nbins=20,
        title='Age Distribution',
        labels={'age': 'Age (years)', 'count': 'Number of Patients'},
        color_discrete_sequence=['#a8c0ff']
    )
    
    fig_age.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.markdown("**Gender Distribution**")
    
    gender_counts = patients_filtered['gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    fig_gender = px.pie(
        gender_counts,
        values='Count',
        names='Gender',
        title='Gender Distribution',
        color_discrete_sequence=['#a8c0ff', '#fdebd0']
    )
    
    fig_gender.update_layout(height=300)
    st.plotly_chart(fig_gender, use_container_width=True)

with col3:
    st.markdown("**Baseline Severity**")
    
    severity_counts = patients_filtered['baseline_severity'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    
    fig_severity = px.bar(
        severity_counts,
        x='Severity',
        y='Count',
        title='Baseline Disease Severity',
        color='Count',
        color_continuous_scale='Mint'
    )
    
    fig_severity.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_severity, use_container_width=True)

st.markdown("---")

# ========== LABORATORY ANALYSIS ==========
st.subheader("Laboratory Safety Monitoring")

# Merge with patient data
labs_enhanced = labs_filtered.merge(
    patients_filtered[['patient_id', 'treatment_arm']], 
    on='patient_id'
)

# ALT over time by treatment arm
alt_summary = labs_enhanced.groupby(['treatment_arm', 'visit_week']).agg({
    'alt_ul': ['mean', 'std']
}).reset_index()
alt_summary.columns = ['treatment_arm', 'visit_week', 'mean_alt', 'std_alt']

fig_alt = go.Figure()

for arm in alt_summary['treatment_arm'].unique():
    arm_data = alt_summary[alt_summary['treatment_arm'] == arm]
    
    fig_alt.add_trace(go.Scatter(
        x=arm_data['visit_week'],
        y=arm_data['mean_alt'],
        mode='lines+markers',
        name=arm,
        line=dict(width=2),
        marker=dict(size=6)
    ))

fig_alt.add_hline(y=40, line_dash="dash", line_color="red", 
                  annotation_text="Upper Limit of Normal")

fig_alt.update_layout(
    title='ALT (Liver Enzyme) Levels Over Time',
    xaxis_title='Week',
    yaxis_title='ALT (U/L)',
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_alt, use_container_width=True)

st.markdown("---")

# ========== SITE PERFORMANCE ==========
st.subheader("Site Performance Metrics")

site_metrics = patients_filtered.groupby('site_id').agg({
    'patient_id': 'count',
    'completed_trial': 'sum'
}).reset_index()

site_metrics.columns = ['site_id', 'enrolled', 'completed']
site_metrics['dropout_rate'] = ((site_metrics['enrolled'] - site_metrics['completed']) / 
                                 site_metrics['enrolled'] * 100)

site_metrics = site_metrics.merge(sites_df[['site_id', 'site_name', 'city']], on='site_id')

# Sort by enrollment for better visualization
site_metrics = site_metrics.sort_values('enrolled', ascending=True)

fig_sites = go.Figure()

fig_sites.add_trace(go.Bar(
    name='Completed',
    y=site_metrics['site_name'],
    x=site_metrics['completed'],
    orientation='h',
    marker=dict(
        color='#5dae8b',
        line=dict(color='white', width=1)
    ),
    text=site_metrics['completed'],
    textposition='inside',
    hovertemplate='<b>%{y}</b><br>Completed: %{x}<extra></extra>'
))

fig_sites.add_trace(go.Bar(
    name='Enrolled',
    y=site_metrics['site_name'],
    x=site_metrics['enrolled'],
    orientation='h',
    marker=dict(
        color='#a8c0ff',
        line=dict(color='white', width=1)
    ),
    text=site_metrics['enrolled'],
    textposition='inside',
    hovertemplate='<b>%{y}</b><br>Total Enrolled: %{x}<extra></extra>'
))

fig_sites.update_layout(
    title='Patient Enrollment and Trial Completion by Research Site',
    xaxis_title='Number of Patients',
    yaxis_title='',
    barmode='overlay',
    height=450,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    yaxis=dict(
        tickfont=dict(size=11)
    )
)

st.plotly_chart(fig_sites, use_container_width=True)

# Add completion rate table
st.markdown("**Site Completion Rates**")

site_table = site_metrics[['site_name', 'city', 'enrolled', 'completed', 'dropout_rate']].copy()
site_table.columns = ['Site', 'Location', 'Enrolled', 'Completed', 'Dropout Rate (%)']
site_table['Dropout Rate (%)'] = site_table['Dropout Rate (%)'].apply(lambda x: f'{x:.1f}%')
site_table = site_table.sort_values('Enrolled', ascending=False)

st.dataframe(site_table, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Clinical Trial Analytics Dashboard | Phase III Multi-Center Study</p>
    <p style='font-size: 0.9em;'>For Research and Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)