import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def plot_correlation_heatmap(df, method='spearman', figsize=(12, 10)):
    """Create a correlation heatmap"""
    correlation_matrix = df.select_dtypes(include=[np.number]).corr(method='spearman')

    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title(f'{method.capitalize()} Correlation Matrix Heatmap')
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64

def load_and_prepare_data(file_path):
    """
    Load and prepare the manufacturing dataset with proper data types
    """
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.isocalendar().week
    
    # Convert boolean failure column
    df['fallo_detectado'] = df['fallo_detectado'].map({'Sí': 1, 'No': 0})
    
    # Fill missing values
    df['vibración'] = df['vibración'].fillna(df['vibración'].median())
    df['eficiencia_porcentual'] = df['eficiencia_porcentual'].fillna(df['eficiencia_porcentual'].median())
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def plot_failures_per_month(df):
    """
    Plot average failures per month with trend analysis
    """
    # Group by year-month and calculate failure statistics
    monthly_failures = df.groupby([df['timestamp'].dt.to_period('M')]).agg({
        'fallo_detectado': ['sum', 'mean', 'count'],
        'maquina_id': 'nunique'
    }).round(3)
    
    monthly_failures.columns = ['total_failures', 'failure_rate', 'total_records', 'unique_machines']
    monthly_failures['avg_failures_per_machine'] = monthly_failures['total_failures'] / monthly_failures['unique_machines']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Failure Analysis by Month', fontsize=16, fontweight='bold')
    
    # Total failures per month
    axes[0,0].plot(monthly_failures.index.astype(str), monthly_failures['total_failures'], 
                   marker='o', linewidth=2, markersize=8, color='#e74c3c')
    axes[0,0].set_title('Total Failures per Month', fontweight='bold')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Number of Failures')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Failure rate per month
    axes[0,1].plot(monthly_failures.index.astype(str), monthly_failures['failure_rate'] * 100, 
                   marker='s', linewidth=2, markersize=8, color='#f39c12')
    axes[0,1].set_title('Failure Rate per Month (%)', fontweight='bold')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Failure Rate (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Average failures per machine
    axes[1,0].bar(monthly_failures.index.astype(str), monthly_failures['avg_failures_per_machine'], 
                  color='#9b59b6', alpha=0.7)
    axes[1,0].set_title('Average Failures per Machine per Month', fontweight='bold')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Avg Failures per Machine')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Failure types distribution
    failure_types = df[df['fallo_detectado'] == 1]['tipo_fallo'].value_counts()
    axes[1,1].pie(failure_types.values, labels=failure_types.index, autopct='%1.1f%%', 
                  colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    axes[1,1].set_title('Distribution of Failure Types', fontweight='bold')
    
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64
    
def plot_unexpected_stops_heatmaps(df):
    """
    Create heatmaps for unexpected stops by machine and operator
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('🔥 Unexpected Stops Analysis - Heatmaps', fontsize=18, fontweight='bold')
    
    # 1. Unexpected stops by machine and operator
    stops_matrix = df.pivot_table(values='paradas_imprevistas', 
                                  index='maquina_id', 
                                  columns='operador_id', 
                                  aggfunc='mean', 
                                  fill_value=0)
    
    sns.heatmap(stops_matrix, annot=True, fmt='.2f', cmap='Reds', 
                ax=axes[0,0], cbar_kws={'label': 'Avg Unexpected Stops'})
    axes[0,0].set_title('Unexpected Stops: Machine vs Operator', fontweight='bold', pad=20)
    axes[0,0].set_xlabel('Operator ID', fontweight='bold')
    axes[0,0].set_ylabel('Machine ID', fontweight='bold')
    
    # 2. Unexpected stops by machine and shift
    stops_shift_matrix = df.pivot_table(values='paradas_imprevistas', 
                                        index='maquina_id', 
                                        columns='turno', 
                                        aggfunc='mean', 
                                        fill_value=0)
    
    sns.heatmap(stops_shift_matrix, annot=True, fmt='.2f', cmap='Oranges', 
                ax=axes[0,1], cbar_kws={'label': 'Avg Unexpected Stops'})
    axes[0,1].set_title('Unexpected Stops: Machine vs Shift', fontweight='bold', pad=20)
    axes[0,1].set_xlabel('Shift', fontweight='bold')
    axes[0,1].set_ylabel('Machine ID', fontweight='bold')
    
    # 3. Efficiency by machine and operator
    efficiency_matrix = df.pivot_table(values='eficiencia_porcentual', 
                                       index='maquina_id', 
                                       columns='operador_id', 
                                       aggfunc='mean', 
                                       fill_value=0)
    
    sns.heatmap(efficiency_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=axes[1,0], cbar_kws={'label': 'Avg Efficiency %'})
    axes[1,0].set_title('Efficiency: Machine vs Operator', fontweight='bold', pad=20)
    axes[1,0].set_xlabel('Operator ID', fontweight='bold')
    axes[1,0].set_ylabel('Machine ID', fontweight='bold')
    
    # 4. Defective units by machine and product
    defects_matrix = df.pivot_table(values='unidades_defectuosas', 
                                    index='maquina_id', 
                                    columns='producto_id', 
                                    aggfunc='mean', 
                                    fill_value=0)
    
    sns.heatmap(defects_matrix, annot=True, fmt='.1f', cmap='Blues', 
                ax=axes[1,1], cbar_kws={'label': 'Avg Defective Units'})
    axes[1,1].set_title('Defective Units: Machine vs Product', fontweight='bold', pad=20)
    axes[1,1].set_xlabel('Product ID', fontweight='bold')
    axes[1,1].set_ylabel('Machine ID', fontweight='bold')
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64
    
def analyze_machine_performance(df):
    """
    Comprehensive machine performance analysis
    """
    machine_stats = df.groupby('maquina_id').agg({
        'eficiencia_porcentual': ['mean', 'std', 'min', 'max'],
        'paradas_imprevistas': ['sum', 'mean'],
        'paradas_programadas': ['sum', 'mean'],
        'fallo_detectado': ['sum', 'mean'],
        'unidades_defectuosas': ['sum', 'mean'],
        'cantidad_producida': ['sum', 'mean'],
        'consumo_energia': ['mean'],
        'temperatura': ['mean', 'std'],
        'vibración': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    machine_stats.columns = ['_'.join(col).strip() for col in machine_stats.columns.values]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('🏭 Machine Performance Analysis', fontsize=18, fontweight='bold')
    
    # Average efficiency by machine
    axes[0,0].bar(machine_stats.index, machine_stats['eficiencia_porcentual_mean'], 
                  color='#2ecc71', alpha=0.7)
    axes[0,0].set_title('Average Efficiency by Machine', fontweight='bold')
    axes[0,0].set_ylabel('Efficiency (%)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Total unexpected stops
    axes[0,1].bar(machine_stats.index, machine_stats['paradas_imprevistas_sum'], 
                  color='#e74c3c', alpha=0.7)
    axes[0,1].set_title('Total Unexpected Stops by Machine', fontweight='bold')
    axes[0,1].set_ylabel('Unexpected Stops')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Total production by machine
    axes[0,2].bar(machine_stats.index, machine_stats['cantidad_producida_sum'], 
                  color='#3498db', alpha=0.7)
    axes[0,2].set_title('Total Production by Machine', fontweight='bold')
    axes[0,2].set_ylabel('Units Produced')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Failure rate by machine
    axes[1,0].bar(machine_stats.index, machine_stats['fallo_detectado_mean'] * 100, 
                  color='#f39c12', alpha=0.7)
    axes[1,0].set_title('Failure Rate by Machine (%)', fontweight='bold')
    axes[1,0].set_ylabel('Failure Rate (%)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Average energy consumption
    axes[1,1].bar(machine_stats.index, machine_stats['consumo_energia_mean'], 
                  color='#9b59b6', alpha=0.7)
    axes[1,1].set_title('Average Energy Consumption by Machine', fontweight='bold')
    axes[1,1].set_ylabel('Energy Consumption')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Defect rate by machine
    defect_rate = (machine_stats['unidades_defectuosas_sum'] / machine_stats['cantidad_producida_sum']) * 100
    axes[1,2].bar(machine_stats.index, defect_rate, color='#e67e22', alpha=0.7)
    axes[1,2].set_title('Defect Rate by Machine (%)', fontweight='bold')
    axes[1,2].set_ylabel('Defect Rate (%)')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64
    
def analyze_operator_performance(df):
    """
    Operator performance analysis with rankings
    """
    operator_stats = df.groupby('operador_id').agg({
        'eficiencia_porcentual': ['mean', 'std'],
        'paradas_imprevistas': ['sum', 'mean'],
        'fallo_detectado': ['sum', 'mean'],
        'unidades_defectuosas': ['sum', 'mean'],
        'cantidad_producida': ['sum', 'mean']
    }).round(2)
    
    operator_stats.columns = ['_'.join(col).strip() for col in operator_stats.columns.values]
    
    # Calculate performance score (higher is better)
    operator_stats['performance_score'] = (
        operator_stats['eficiencia_porcentual_mean'] * 0.4 +
        (100 - operator_stats['paradas_imprevistas_mean'] * 10) * 0.3 +
        (100 - operator_stats['fallo_detectado_mean'] * 100) * 0.3
    )
    
    operator_stats = operator_stats.sort_values('performance_score', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('👨‍🔧 Operator Performance Analysis', fontsize=16, fontweight='bold')
    
    # Efficiency ranking
    axes[0,0].barh(operator_stats.index, operator_stats['eficiencia_porcentual_mean'], 
                   color=plt.cm.RdYlGn(operator_stats['eficiencia_porcentual_mean']/100))
    axes[0,0].set_title('Average Efficiency by Operator', fontweight='bold')
    axes[0,0].set_xlabel('Efficiency (%)')
    
    # Unexpected stops
    axes[0,1].barh(operator_stats.index, operator_stats['paradas_imprevistas_sum'], 
                   color=plt.cm.Reds(operator_stats['paradas_imprevistas_sum']/operator_stats['paradas_imprevistas_sum'].max()))
    axes[0,1].set_title('Total Unexpected Stops by Operator', fontweight='bold')
    axes[0,1].set_xlabel('Unexpected Stops')
    
    # Performance score ranking
    axes[1,0].barh(operator_stats.index, operator_stats['performance_score'], 
                   color=plt.cm.viridis(operator_stats['performance_score']/100))
    axes[1,0].set_title('Overall Performance Score by Operator', fontweight='bold')
    axes[1,0].set_xlabel('Performance Score')
    
    # Production vs defects scatter
    axes[1,1].scatter(operator_stats['cantidad_producida_sum'], 
                      operator_stats['unidades_defectuosas_sum'],
                      s=100, alpha=0.7, c=operator_stats['eficiencia_porcentual_mean'], 
                      cmap='RdYlGn')
    axes[1,1].set_title('Production vs Defects by Operator', fontweight='bold')
    axes[1,1].set_xlabel('Total Production')
    axes[1,1].set_ylabel('Total Defects')
    
    # Add operator labels to scatter plot
    for i, op in enumerate(operator_stats.index):
        axes[1,1].annotate(op, 
                          (operator_stats['cantidad_producida_sum'].iloc[i], 
                           operator_stats['unidades_defectuosas_sum'].iloc[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64
    
def analyze_temporal_patterns(df):
    """
    Analyze patterns by time of day, day of week, and shifts
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('⏰ Temporal Patterns Analysis', fontsize=18, fontweight='bold')
    
    # Efficiency by hour of day
    hourly_efficiency = df.groupby('hour')['eficiencia_porcentual'].mean()
    axes[0,0].plot(hourly_efficiency.index, hourly_efficiency.values, 
                   marker='o', linewidth=3, markersize=8, color='#2ecc71')
    axes[0,0].set_title('Efficiency by Hour of Day', fontweight='bold')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Average Efficiency (%)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Failures by day of week
    daily_failures = df.groupby('day_of_week')['fallo_detectado'].sum()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_failures = daily_failures.reindex(day_order)
    axes[0,1].bar(daily_failures.index, daily_failures.values, 
                  color='#e74c3c', alpha=0.7)
    axes[0,1].set_title('Total Failures by Day of Week', fontweight='bold')
    axes[0,1].set_ylabel('Number of Failures')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Shift performance comparison
    shift_stats = df.groupby('turno').agg({
        'eficiencia_porcentual': 'mean',
        'paradas_imprevistas': 'mean',
        'fallo_detectado': 'mean'
    })
    
    x = np.arange(len(shift_stats.index))
    width = 0.25
    
    axes[0,2].bar(x - width, shift_stats['eficiencia_porcentual'], width, 
                  label='Efficiency', color='#2ecc71', alpha=0.7)
    axes[0,2].bar(x, shift_stats['paradas_imprevistas'] * 20, width, 
                  label='Unexpected Stops (x20)', color='#e74c3c', alpha=0.7)
    axes[0,2].bar(x + width, shift_stats['fallo_detectado'] * 100, width, 
                  label='Failure Rate (x100)', color='#f39c12', alpha=0.7)
    
    axes[0,2].set_title('Performance by Shift', fontweight='bold')
    axes[0,2].set_xlabel('Shift')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(shift_stats.index)
    axes[0,2].legend()
    
    # Production volume by hour
    hourly_production = df.groupby('hour')['cantidad_producida'].mean()
    axes[1,0].bar(hourly_production.index, hourly_production.values, 
                  color='#3498db', alpha=0.7)
    axes[1,0].set_title('Average Production by Hour', fontweight='bold')
    axes[1,0].set_xlabel('Hour')
    axes[1,0].set_ylabel('Average Production')
    
    # Energy consumption patterns
    hourly_energy = df.groupby('hour')['consumo_energia'].mean()
    axes[1,1].plot(hourly_energy.index, hourly_energy.values, 
                   marker='s', linewidth=3, markersize=6, color='#9b59b6')
    axes[1,1].set_title('Energy Consumption by Hour', fontweight='bold')
    axes[1,1].set_xlabel('Hour')
    axes[1,1].set_ylabel('Average Energy Consumption')
    axes[1,1].grid(True, alpha=0.3)
    
    # Temperature and vibration patterns
    temp_vib_by_hour = df.groupby('hour')[['temperatura', 'vibración']].mean()
    ax_temp = axes[1,2]
    ax_vib = ax_temp.twinx()
    
    line1 = ax_temp.plot(temp_vib_by_hour.index, temp_vib_by_hour['temperatura'], 
                         'r-', linewidth=3, label='Temperature', marker='o')
    line2 = ax_vib.plot(temp_vib_by_hour.index, temp_vib_by_hour['vibración'], 
                        'b-', linewidth=3, label='Vibration', marker='s')
    
    ax_temp.set_xlabel('Hour')
    ax_temp.set_ylabel('Temperature (°C)', color='r')
    ax_vib.set_ylabel('Vibration Level', color='b')
    ax_temp.set_title('Temperature and Vibration by Hour', fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_temp.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64

def create_correlation_analysis(df):
    """
    Create correlation matrix and analysis of key variables
    """
    # Select numerical columns for correlation
    numerical_cols = ['temperatura', 'vibración', 'humedad', 'tiempo_ciclo', 
                      'eficiencia_porcentual', 'consumo_energia', 'cantidad_producida',
                      'unidades_defectuosas', 'paradas_programadas', 'paradas_imprevistas']
    
    correlation_matrix = df[numerical_cols].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('🔍 Correlation Analysis', fontsize=18, fontweight='bold')
    
    # Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=axes[0], cbar_kws={'label': 'Correlation Coefficient'})
    axes[0].set_title('Correlation Matrix of Key Variables', fontweight='bold', pad=20)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Focus on efficiency correlations
    efficiency_corr = correlation_matrix['eficiencia_porcentual'].sort_values(key=abs, ascending=False)[1:]  # Exclude self-correlation
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in efficiency_corr.values]
    axes[1].barh(range(len(efficiency_corr)), efficiency_corr.values, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(efficiency_corr)))
    axes[1].set_yticklabels(efficiency_corr.index)
    axes[1].set_title('Variables Most Correlated with Efficiency', fontweight='bold', pad=20)
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add correlation values as text
    for i, v in enumerate(efficiency_corr.values):
        axes[1].text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                     va='center', ha='left' if v > 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_base64

