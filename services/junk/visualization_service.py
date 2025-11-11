# FIXED VISUALIZATION SERVICE - Only BAR and PIE charts

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

from utils.logger_config import get_logger

logger = get_logger(__name__)

class SmartVisualizationService:
    """Simple visualization service - Only BAR and PIE charts"""
    
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
            'figure.figsize': [10, 6],
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'font.size': 11,
            'axes.titlesize': 14,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    async def create_smart_visualization(self, data: List[Dict], query_context: str = None, 
                                        chart_title: str = None) -> Dict[str, Any]:
        """Create BAR or PIE chart based on data"""
        try:
            if not data or len(data) == 0:
                return {
                    'success': False,
                    'error': 'No data provided'
                }
            
            df = pd.DataFrame(data)
            
            # Decide: BAR or PIE?
            chart_type = self._decide_chart_type(df, query_context)
            
            # Create the chart
            if chart_type == 'pie':
                chart_result = self._create_pie_chart(df, chart_title or 'Data Distribution')
            else:
                chart_result = self._create_bar_chart(df, chart_title or 'Data Comparison')
            
            if not chart_result:
                raise Exception("Chart creation failed")
            
            # Save chart
            chart_url, chart_base64 = self._save_chart()
            
            return {
                'success': True,
                'chart_type': chart_type,
                'chart_url': chart_url,
                'chart_base64': chart_base64,
                'insights': self._generate_insights(df, chart_type)
            }
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _decide_chart_type(self, df: pd.DataFrame, query_context: str = None) -> str:
        """Decide between BAR and PIE - ONLY THESE TWO"""
        
        if query_context:
            query_lower = query_context.lower()
            
            # PIE chart indicators
            if any(word in query_lower for word in ['distribution', 'proportion', 'percentage', 'breakdown', 'split', 'share']):
                return 'pie'
        
        # PIE chart: Few categories (<=8) with good distribution
        if len(df) <= 8:
            return 'pie'
        
        # BAR chart: Default for everything else
        return 'bar'
    
    def _create_pie_chart(self, df: pd.DataFrame, title: str) -> bool:
        """Create pie chart"""
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
                    num_col = col
                else:
                    cat_col = col
            
            # Prepare data for pie chart
            if cat_col and num_col:
                # Group by category, sum numeric values
                grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
            elif cat_col:
                # Just count categories
                grouped = df[cat_col].value_counts()
            else:
                # Use first column
                grouped = df.iloc[:, 0].value_counts()
            
            # Limit to top 10
            grouped = grouped.head(10)
            
            # Create pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))
            wedges, texts, autotexts = ax.pie(
                grouped.values, 
                labels=grouped.index, 
                autopct='%1.1f%%',
                startangle=90, 
                colors=colors,
                textprops={'fontsize': 10}
            )
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return True
            
        except Exception as e:
            logger.error(f"Pie chart creation failed: {e}")
            return False
    
    def _create_bar_chart(self, df: pd.DataFrame, title: str) -> bool:
        """Create bar chart"""
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 6))
            
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
            elif cat_col:
                # Count categories
                grouped = df[cat_col].value_counts().head(20)
            else:
                # Use first column
                grouped = df.iloc[:, 0].value_counts().head(20)
            
            # Create bar chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
            bars = ax.bar(range(len(grouped)), grouped.values, color=colors, edgecolor='black', linewidth=0.5)
            
            # Set labels
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_xlabel(cat_col.replace('_', ' ').title() if cat_col else 'Category', fontweight='bold')
            ax.set_ylabel(num_col.replace('_', ' ').title() if num_col else 'Count', fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.grid(True, alpha=0.3, axis='y')
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
                logger.info(f"Chart saved: {filepath}")
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
    
    def _generate_insights(self, df: pd.DataFrame, chart_type: str) -> List[str]:
        """Generate simple insights about the visualization"""
        insights = []
        
        insights.append(f"Created {chart_type} chart from {len(df)} data points")
        
        if chart_type == 'pie':
            insights.append("Pie chart shows proportional distribution - great for understanding parts of a whole")
        else:
            insights.append("Bar chart compares values across categories - perfect for spotting differences")
        
        # Add a data insight
        if len(df.columns) > 0:
            insights.append(f"Data contains {len(df.columns)} attributes")
        
        return insights