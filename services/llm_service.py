# 🤖 LLM SERVICE - Claude 3.5 Sonnet + OpenAI + Gemini

import logging
from typing import Dict, Any, Optional, List
import asyncio
from anthropic import Anthropic
import openai
import google.generativeai as genai
from utils.config import config, LLM_CONFIGS

logger = logging.getLogger(__name__)

class LLMService:
    """
    Multi-LLM service following your architecture:
    1. Primary: Claude 3.5 Sonnet (Anthropic)
    2. Fallback 1: GPT-4 (OpenAI)
    3. Fallback 2: Gemini Pro (Google)
    """
    
    def __init__(self):
        self.primary_model = config.PRIMARY_LLM
        self.fallback_models = [config.FALLBACK_LLM_1, config.FALLBACK_LLM_2]
        
        # Initialize clients
        self.anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        openai.api_key = config.OPENAI_API_KEY
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        logger.info(f"✅ LLM Service initialized - Primary: {self.primary_model}")
    
    async def get_completion(
        self, 
        prompt: str, 
        max_tokens: int = 4000,
        temperature: float = 0.1,
        model: Optional[str] = None
    ) -> str:
        """
        Get completion with automatic fallback
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness  
            model: Specific model to use (optional)
            
        Returns:
            str: Generated completion
        """
        target_model = model or self.primary_model
        models_to_try = [target_model] + [m for m in self.fallback_models if m != target_model]
        
        for model_name in models_to_try:
            try:
                logger.info(f"🎯 Attempting completion with {model_name}")
                
                completion = await self._call_specific_model(
                    model_name, prompt, max_tokens, temperature
                )
                
                if completion:
                    logger.info(f"✅ Success with {model_name}")
                    return completion
                    
            except Exception as e:
                logger.warning(f"⚠️ {model_name} failed: {e}")
                continue
        
        raise Exception("All LLM models failed to generate completion")
    
    async def _call_specific_model(
        self, 
        model_name: str, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """Call specific LLM model"""
        model_config = LLM_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        provider = model_config["provider"]
        
        if provider == "anthropic":
            return await self._call_anthropic(model_name, prompt, max_tokens, temperature)
        elif provider == "openai":
            return await self._call_openai(model_name, prompt, max_tokens, temperature)
        elif provider == "gemini":
            return await self._call_gemini(model_name, prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _call_anthropic(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Anthropic Claude API"""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _call_openai(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call OpenAI GPT API"""
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_gemini(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Google Gemini API"""
        try:
            gemini_model = genai.GenerativeModel(model)
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def generate_sql_query(
        self, 
        user_prompt: str,
        table_metadata: List[Dict[str, Any]],
        column_metadata: List[Dict[str, Any]], 
        common_prompts: List[Dict[str, Any]]
    ) -> str:
        """
        Generate SQL query using exact prompt template from your architecture
        
        Args:
            user_prompt: User's question
            table_metadata: Relevant table information
            column_metadata: Relevant column information  
            common_prompts: Similar prompt-SQL examples
            
        Returns:
            str: Generated SQL query
        """
        # Format table metadata
        tables_text = "\n".join([
            f"- {table['table_name']}: {table.get('table_description', 'No description')}"
            for table in table_metadata
        ])
        
        # Format column metadata
        columns_text = "\n".join([
            f"- {col['table_name']}.{col['column_name']}: {col.get('column_description', 'No description')}"
            for col in column_metadata
        ])
        
        # Format common prompt examples
        examples_text = "\n".join([
            f"User: {example['prompt']}\nSQL: {example['sql_query']}\n"
            for example in common_prompts[:5]  # Limit to top 5 examples
        ])
        
        # Use exact prompt template from your PDF architecture
        from utils.config import PROMPT_TEMPLATES
        sql_prompt = PROMPT_TEMPLATES["data_question"]
        sql_prompt = sql_prompt.replace("<User Prompt>", user_prompt)
        sql_prompt = sql_prompt.replace("<Table Metadata>", tables_text)
        sql_prompt = sql_prompt.replace("<Column Metadata>", columns_text)
        sql_prompt = sql_prompt.replace("<Common Prompt SQLs>", examples_text)
        
        logger.info(f"🎯 Generating SQL for: {user_prompt}")
        
        completion = await self.get_completion(
            prompt=sql_prompt,
            max_tokens=1000,
            temperature=0.1
        )
        
        # Extract SQL from completion (remove any explanatory text)
        sql_query = self._extract_sql(completion)
        
        logger.info(f"✅ Generated SQL: {sql_query}")
        return sql_query
    
    async def answer_business_question(
        self, 
        user_prompt: str,
        business_context: List[Dict[str, Any]]
    ) -> str:
        """
        Answer non-data question using business context
        
        Args:
            user_prompt: User's question
            business_context: Relevant business context pieces
            
        Returns:
            str: Generated answer
        """
        # Format business context
        context_text = "\n".join([
            context.get('document_piece', '')
            for context in business_context
        ])
        
        # Use exact prompt template from your PDF architecture
        from utils.config import PROMPT_TEMPLATES
        business_prompt = PROMPT_TEMPLATES["non_data_question"]
        business_prompt = business_prompt.replace("<User Prompt>", user_prompt)
        business_prompt = business_prompt.replace("<Business Context>", context_text)
        
        logger.info(f"🎯 Answering business question: {user_prompt}")
        
        completion = await self.get_completion(
            prompt=business_prompt,
            max_tokens=2000,
            temperature=0.3
        )
        
        logger.info(f"✅ Generated business answer")
        return completion
    
    def _extract_sql(self, completion: str) -> str:
        """Extract clean SQL from LLM completion"""
        # Remove common prefixes/suffixes
        lines = completion.strip().split('\n')
        sql_lines = []
        
        in_sql_block = False
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and common explanatory text
            if not line or line.startswith("Here") or line.startswith("The SQL"):
                continue
            
            # Look for SQL keywords
            if any(line.upper().startswith(keyword) for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
                in_sql_block = True
                sql_lines.append(line)
            elif in_sql_block and line.endswith(';'):
                sql_lines.append(line)
                break
            elif in_sql_block:
                sql_lines.append(line)
        
        if not sql_lines:
            # Fallback: return the completion as-is
            return completion.strip()
        
        return '\n'.join(sql_lines)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all configured models"""
        status = {}
        
        for model_name in [self.primary_model] + self.fallback_models:
            try:
                # Test with a simple prompt
                test_prompt = "Say 'OK' if you can respond."
                result = await self._call_specific_model(model_name, test_prompt, 10, 0.1)
                status[model_name] = {"available": True, "test_response": result[:20]}
            except Exception as e:
                status[model_name] = {"available": False, "error": str(e)}
        
        return status