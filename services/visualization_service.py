# 📊 UPDATED VISUALIZATION SERVICE - Integrated with Your Existing Service

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import os
import io
import base64

logger = logging.getLogger(__name__)

class SmartVisualizationService:
    """Enhanced visualization service with your exact implementation + improvements"""
    
    def __init__(self):
        self.logger = logger
        self._setup_styling()
        self._ensure_charts_directory()
    
    def _ensure_charts_directory(self):
        """Create charts directory"""
        try:
            os.makedirs("static/charts", exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create charts directory: {e}")
    
    def _setup_styling(self):
        """Setup matplotlib styling"""
        plt.rcParams.update({
            'figure.figsize': [12, 7],
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'font.size': 11,
            'axes.titlesize': 14,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    async def create_smart_visualization(
        self, 
        data: List[Dict], 
        query_context: str = None,
        chart_title: str = None
    ) -> Dict[str, Any]:
        """
        Create BAR or PIE chart based on data - ENHANCED VERSION
        Now gets called after SQL query execution as per your requirement
        """
        try:
            if not data or len(data) == 0:
                return {
                    'success': False,
                    'error': 'No data provided for visualization'
                }
            
            logger.info(f"📊 Creating visualization for {len(data)} data points")
            
            df = pd.DataFrame(data)
            
            # Decide: BAR or PIE?
            chart_type = self._decide_chart_type(df, query_context)
            
            # Generate smart title if not provided
            if not chart_title:
                chart_title = self._generate_chart_title(df, query_context, chart_type)
            
            # Create the chart
            if chart_type == 'pie':
                chart_result = self._create_pie_chart(df, chart_title)
            else:
                chart_result = self._create_bar_chart(df, chart_title)
            
            if not chart_result:
                raise Exception("Chart creation failed")
            
            # Save chart
            chart_url, chart_base64 = self._save_chart()
            
            # Generate enhanced insights
            insights = self._generate_enhanced_insights(df, chart_type, query_context)
            
            return {
                'success': True,
                'chart_type': chart_type,
                'chart_url': chart_url,
                'chart_base64': chart_base64,
                'title': chart_title,
                'data_points': len(data),
                'columns': list(df.columns),
                'insights': insights,
                'visualization_context': f"Generated from SQL query results"
            }
        
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_chart_title(self, df: pd.DataFrame, query_context: str, chart_type: str) -> str:
        """Generate smart chart title based on data and context"""
        try:
            if query_context:
                # Extract key words from query context
                query_lower = query_context.lower()
                
                # Common patterns
                if 'count' in query_lower:
                    return f"Count Analysis ({chart_type.title()} Chart)"
                elif 'revenue' in query_lower or 'sales' in query_lower:
                    return f"Revenue Analysis ({chart_type.title()} Chart)"
                elif 'customer' in query_lower:
                    return f"Customer Analysis ({chart_type.title()} Chart)"
                elif 'order' in query_lower:
                    return f"Order Analysis ({chart_type.title()} Chart)"
                elif 'product' in query_lower:
                    return f"Product Analysis ({chart_type.title()} Chart)"
            
            # Fallback based on data
            if len(df.columns) > 1:
                return f"Data Analysis - {df.columns[0]} vs {df.columns[1]} ({chart_type.title()} Chart)"
            else:
                return f"Data Distribution ({chart_type.title()} Chart)"
                
        except:
            return f"Query Results Visualization ({chart_type.title()} Chart)"
    
    def _decide_chart_type(self, df: pd.DataFrame, query_context: str = None) -> str:
        """Enhanced chart type decision with better context awareness"""
        if query_context:
            query_lower = query_context.lower()
            
            # PIE chart indicators
            pie_keywords = [
                'distribution', 'proportion', 'percentage', 'breakdown', 
                'split', 'share', 'composition', 'ratio'
            ]
            if any(word in query_lower for word in pie_keywords):
                return 'pie'
            
            # BAR chart indicators  
            bar_keywords = [
                'compare', 'comparison', 'top', 'highest', 'lowest', 
                'count', 'total', 'sum', 'revenue', 'sales'
            ]
            if any(word in query_lower for word in bar_keywords):
                return 'bar'
        
        # Data-based decision
        if len(df) <= 8 and len(df) >= 2:
            return 'pie'  # Good for small categorical data
        
        # Check if we have good categorical + numeric data for pie
        has_categorical = False
        has_numeric = False
        
        for col in df.columns:
            sample = df[col].dropna().head(5)
            if not sample.empty:
                if pd.api.types.is_numeric_dtype(sample):
                    has_numeric = True
                else:
                    has_categorical = True
        
        if has_categorical and has_numeric and len(df) <= 10:
            return 'pie'
        
        # Default to bar chart
        return 'bar'
    
    def _create_pie_chart(self, df: pd.DataFrame, title: str) -> bool:
        """Enhanced pie chart creation"""
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Find categorical and numerical columns
            cat_col = None
            num_col = None
            
            for col in df.columns:
                sample = df[col].dropna().head(5)
                if sample.empty:
                    continue
                    
                if pd.api.types.is_numeric_dtype(sample) and not all(isinstance(x, bool) for x in sample):
                    if num_col is None:  # Take first numeric column
                        num_col = col
                else:
                    if cat_col is None:  # Take first categorical column
                        cat_col = col
            
            # Prepare data for pie chart
            if cat_col and num_col:
                # Group by category, sum numeric values
                grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                value_label = num_col.replace('_', ' ').title()
            elif cat_col:
                # Just count categories
                grouped = df[cat_col].value_counts()
                value_label = 'Count'
            else:
                # Use first column
                grouped = df.iloc[:, 0].value_counts()
                value_label = 'Count'
            
            # Limit to top 10 for readability
            if len(grouped) > 10:
                other_sum = grouped.iloc[10:].sum()
                grouped = grouped.iloc[:10]
                if other_sum > 0:
                    grouped['Others'] = other_sum
            
            # Create pie chart with enhanced styling
            colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))
            wedges, texts, autotexts = ax.pie(
                grouped.values,
                labels=grouped.index,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*grouped.values.sum())})',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 9},
                explode=[0.05 if i == 0 else 0 for i in range(len(grouped))]  # Explode largest slice
            )
            
            # Enhance text styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add legend with values
            legend_labels = [f"{label}: {int(value)}" for label, value in zip(grouped.index, grouped.values)]
            ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            return True
            
        except Exception as e:
            logger.error(f"Pie chart creation failed: {e}")
            return False
    
    def _create_bar_chart(self, df: pd.DataFrame, title: str) -> bool:
        """Enhanced bar chart creation"""
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Find categorical and numerical columns
            cat_col = None
            num_col = None
            
            for col in df.columns:
                sample = df[col].dropna().head(5)
                if sample.empty:
                    continue
                    
                if pd.api.types.is_numeric_dtype(sample) and not all(isinstance(x, bool) for x in sample):
                    if num_col is None:  # Take first numeric column
                        num_col = col
                else:
                    if cat_col is None:  # Take first categorical column
                        cat_col = col
            
            # Prepare data
            if cat_col and num_col:
                # Group by category, sum values
                grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(20)
                y_label = num_col.replace('_', ' ').title()
            elif cat_col:
                # Count categories
                grouped = df[cat_col].value_counts().head(20)
                y_label = 'Count'
            else:
                # Use first column
                grouped = df.iloc[:, 0].value_counts().head(20)
                y_label = 'Count'
            
            # Create bar chart with enhanced styling
            colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
            bars = ax.bar(range(len(grouped)), grouped.values, color=colors, 
                         edgecolor='black', linewidth=0.5, alpha=0.8)
            
            # Set labels with rotation for better readability
            ax.set_xticks(range(len(grouped)))
            labels = [str(label)[:15] + '...' if len(str(label)) > 15 else str(label) 
                     for label in grouped.index]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            ax.set_xlabel(cat_col.replace('_', ' ').title() if cat_col else 'Category', 
                         fontweight='bold')
            ax.set_ylabel(y_label, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Enhanced grid and styling
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            return True
            
        except Exception as e:
            logger.error(f"Bar chart creation failed: {e}")
            return False
    
    def _save_chart(self):
        """Save chart and return URL + base64"""
        try:
            timestamp = int(datetime.now().timestamp() * 1000)
            filename = f"chart_{timestamp}.png"
            filepath = os.path.join("static", "charts", filename)
            
            chart_url = None
            chart_base64 = None
            
            # Try to save to file
            try:
                plt.savefig(filepath, bbox_inches='tight', dpi=150, facecolor='white')
                chart_url = f"/static/charts/{filename}"
                logger.info(f"📊 Chart saved: {filepath}")
            except Exception as e:
                logger.warning(f"File save failed: {e}")
            
            # Always create base64 as fallback
            try:
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
                buffer.seek(0)
                chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
            except Exception as e:
                logger.error(f"Base64 encoding failed: {e}")
            
            plt.close()
            
            return chart_url, chart_base64
            
        except Exception as e:
            logger.error(f"Chart saving failed: {e}")
            plt.close()
            return None, None
    
    def _generate_enhanced_insights(self, df: pd.DataFrame, chart_type: str, query_context: str) -> List[str]:
        """Generate enhanced insights about the visualization"""
        insights = []
        
        # Basic insights
        insights.append(f"📊 Created {chart_type} chart from {len(df)} data points")
        insights.append(f"📈 Data contains {len(df.columns)} attributes: {', '.join(df.columns)}")
        
        # Chart type specific insights
        if chart_type == 'pie':
            insights.append("🥧 Pie chart shows proportional distribution - perfect for understanding parts of a whole")
        else:
            insights.append("📊 Bar chart compares values across categories - excellent for spotting differences and trends")
        
        # Data insights
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                max_val = df[col].max()
                min_val = df[col].min()
                insights.append(f"📊 Range: {min_val:.2f} to {max_val:.2f} in {col}")
        except:
            pass
        
        # Context insights
        if query_context:
            if 'customer' in query_context.lower():
                insights.append("👥 This customer analysis can help identify key segments and opportunities")
            elif 'revenue' in query_context.lower() or 'sales' in query_context.lower():
                insights.append("💰 This revenue analysis provides insights into financial performance")
            elif 'product' in query_context.lower():
                insights.append("🛍️ This product analysis helps understand inventory and demand patterns")
        
        return insights