from typing import Dict, Any, List
import openai
import pandas as pd
from app.config import settings
from app.repositories.csv_repository import CSVRepository
from app.core.exceptions import AIServiceError
import json

class AIService:
    """Service for AI-powered analysis"""
    
    def __init__(self, csv_repository: CSVRepository):
        self.csv_repository = csv_repository
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
        else:
            raise AIServiceError("OpenAI API key not configured")
    
    async def analyze_data(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform AI analysis on data"""
        try:
            # Get data sample
            df = self._get_data_sample(request.data_columns, request.max_records)
            
            # Generate data summary
            data_summary = self._generate_data_summary(df)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(df, request, data_summary)
            
            # Call OpenAI API
            response = await self._call_openai_api(prompt)
            
            # Parse response
            analysis_result = self._parse_ai_response(response)
            
            return AnalysisResponse(
                analysis_type=request.analysis_type,
                insights=analysis_result.get("insights", ""),
                recommendations=analysis_result.get("recommendations"),
                confidence_score=analysis_result.get("confidence_score"),
                data_summary=data_summary
            )
        
        except Exception as e:
            raise AIServiceError(f"AI analysis failed: {str(e)}")
    
    def _get_data_sample(self, columns: List[str], max_records: int) -> pd.DataFrame:
        """Get a sample of data for analysis"""
        df = self.csv_repository.get_all_data()
        
        # Filter columns if specified
        if columns:
            available_columns = [col for col in columns if col in df.columns]
            df = df[available_columns]
        
        # Sample data if too large
        if len(df) > max_records:
            df = df.sample(n=max_records)
        
        return df
    
    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the data"""
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_stats": {},
            "categorical_stats": {}
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns statistics  
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary["categorical_stats"][col] = {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        
        return summary
    
    def _create_analysis_prompt(
        self, 
        df: pd.DataFrame, 
        request: AnalysisRequest, 
        data_summary: Dict[str, Any]
    ) -> str:
        """Create manufacturing-specific prompt for OpenAI analysis"""
        
        # Sample of actual data
        data_sample = df.head(10).to_dict('records')
        
        # Manufacturing context
        manufacturing_context = """
        This is manufacturing/production data from an industrial facility with the following key metrics:
        - timestamp: Date and time of data collection
        - turno: Work shift (Mañana/Morning, Tarde/Afternoon, Noche/Night)
        - operador_id: Operator identifier
        - maquina_id: Machine identifier
        - producto_id: Product identifier
        - temperatura: Operating temperature
        - vibración: Machine vibration levels
        - humedad: Humidity levels
        - tiempo_ciclo: Cycle time for production
        - fallo_detectado: Whether a failure was detected (Sí/No)
        - tipo_fallo: Type of failure (Mecánico/Mechanical, Eléctrico/Electrical)
        - cantidad_producida: Units produced
        - unidades_defectuosas: Defective units
        - eficiencia_porcentual: Production efficiency percentage
        - consumo_energia: Energy consumption
        - paradas_programadas: Scheduled stops
        - paradas_imprevistas: Unscheduled stops
        - observaciones: Operator observations
        """
        
        prompt = f"""
        You are an industrial data analyst specializing in manufacturing operations. Analyze the following production dataset.
        
        {manufacturing_context}
        
        Analysis Type: {request.analysis_type}
        Additional Context: {request.additional_context or 'None provided'}
        
        Dataset Summary:
        - Total Records: {data_summary['shape'][0]}
        - Date Range: {data_summary.get('date_range', 'Not available')}
        - Machines: {data_summary.get('unique_machines', 'Unknown')} unique machines
        - Operators: {data_summary.get('unique_operators', 'Unknown')} unique operators
        - Products: {data_summary.get('unique_products', 'Unknown')} unique products
        - Shifts Distribution: {data_summary.get('shifts', {})}
        
        Key Manufacturing Metrics:
        {self._format_manufacturing_metrics(data_summary.get('numeric_stats', {}))}
        
        Sample Data (first 10 records):
        {self._format_sample_data_for_ai(data_sample)}
        
        Focus your analysis on:
        1. Production efficiency patterns across shifts, machines, and operators
        2. Failure analysis and predictive insights
        3. Energy consumption optimization opportunities
        4. Quality control (defect rates and patterns)
        5. Machine performance and maintenance insights
        6. Operational recommendations for improvement
        
        Provide actionable insights that can help:
        - Improve production efficiency
        - Reduce failures and downtime
        - Optimize energy consumption
        - Enhance quality control
        - Better resource allocation
        
        Format your response as JSON with the following structure:
        {{
            "insights": "detailed manufacturing insights focusing on efficiency, quality, and operational patterns",
            "recommendations": "specific actionable recommendations for production optimization",
            "confidence_score": 0.85
        }}
        """
        
        return prompt
    
    def _format_manufacturing_metrics(self, numeric_stats: Dict) -> str:
        """Format key manufacturing metrics for AI prompt"""
        if not numeric_stats:
            return "No numeric statistics available"
        
        key_metrics = ['temperatura', 'vibración', 'eficiencia_porcentual', 
                      'cantidad_producida', 'unidades_defectuosas', 'consumo_energia']
        
        formatted_metrics = []
        for metric in key_metrics:
            if metric in numeric_stats:
                stats = numeric_stats[metric]
                formatted_metrics.append(
                    f"  {metric}: avg={stats.get('mean', 0):.2f}, "
                    f"min={stats.get('min', 0):.2f}, max={stats.get('max', 0):.2f}"
                )
        
        return "\n".join(formatted_metrics) if formatted_metrics else "Key metrics not available"
    
    def _format_sample_data_for_ai(self, sample_data: List[Dict]) -> str:
        """Format sample data for AI analysis, focusing on key fields"""
        if not sample_data:
            return "No sample data available"
        
        key_fields = ['timestamp', 'turno', 'maquina_id', 'temperatura', 'eficiencia_porcentual', 
                     'cantidad_producida', 'unidades_defectuosas', 'fallo_detectado', 'tipo_fallo']
        
        formatted_samples = []
        for i, record in enumerate(sample_data[:5]):  # Limit to 5 records for prompt efficiency
            filtered_record = {k: v for k, v in record.items() if k in key_fields}
            formatted_samples.append(f"Record {i+1}: {filtered_record}")
        
        return "\n".join(formatted_samples)
