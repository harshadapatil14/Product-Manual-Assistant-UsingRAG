"""
Prompt Optimizer Service for improving RAG results
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class PromptOptimizer:
    """Service for optimizing prompts to improve RAG performance"""
    
    def __init__(self):
        self.prompt_templates = {
            'basic': {
                'system': "You are a helpful assistant that answers questions based on the provided context. Answer accurately and concisely.",
                'user': "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            },
            'detailed': {
                'system': """You are an expert assistant specializing in product manuals and technical documentation. 
                Your task is to provide accurate, detailed, and helpful answers based on the given context.
                
                Guidelines:
                - Always base your answer on the provided context
                - If the context doesn't contain enough information, clearly state what's missing
                - Provide step-by-step instructions when applicable
                - Use clear, professional language
                - Include relevant details and specifications when available""",
                'user': """Context Information:
{context}

User Question: {question}

Please provide a comprehensive answer based on the context above. If any information is missing from the context, please indicate this clearly."""
            },
            'step_by_step': {
                'system': """You are a technical support specialist. When answering questions, always:
                1. Break down complex procedures into clear steps
                2. Highlight important safety warnings or precautions
                3. Mention any required tools or materials
                4. Provide troubleshooting tips when relevant
                5. Use numbered lists for procedures""",
                'user': "Based on this context:\n{context}\n\nQuestion: {question}\n\nProvide a step-by-step answer:"
            },
            'troubleshooting': {
                'system': """You are a troubleshooting expert. When users have problems:
                1. First, identify the most likely cause based on symptoms
                2. Provide step-by-step diagnostic steps
                3. Offer multiple solutions, starting with the simplest
                4. Include safety warnings where applicable
                5. Suggest when professional help might be needed""",
                'user': "Context: {context}\n\nProblem: {question}\n\nPlease help troubleshoot this issue:"
            }
        }
    
    def get_optimized_prompt(self, query: str, context: str, style: str = 'detailed') -> Dict:
        """
        Get an optimized prompt based on query analysis
        
        Args:
            query: User's question
            context: Retrieved context
            style: Prompt style to use
            
        Returns:
            Dictionary with optimized prompt components
        """
        # Analyze query type
        query_type = self._analyze_query_type(query)
        
        # Select appropriate prompt template
        template = self.prompt_templates.get(style, self.prompt_templates['detailed'])
        
        # Customize based on query type
        if query_type == 'how_to':
            template = self.prompt_templates['step_by_step']
        elif query_type == 'problem':
            template = self.prompt_templates['troubleshooting']
        
        # Build the prompt
        system_prompt = template['system']
        user_prompt = template['user'].format(
            context=context,
            question=query
        )
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'query_type': query_type,
            'style': style
        }
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze query to determine the best prompt style"""
        query_lower = query.lower()
        
        # How-to questions
        how_to_indicators = ['how to', 'how do i', 'steps', 'procedure', 'install', 'setup', 'configure']
        if any(indicator in query_lower for indicator in how_to_indicators):
            return 'how_to'
        
        # Problem/troubleshooting questions
        problem_indicators = ['error', 'problem', 'issue', 'not working', 'broken', 'fix', 'troubleshoot']
        if any(indicator in query_lower for indicator in problem_indicators):
            return 'problem'
        
        # Definition/what questions
        definition_indicators = ['what is', 'what are', 'define', 'meaning', 'explain']
        if any(indicator in query_lower for indicator in definition_indicators):
            return 'definition'
        
        return 'general'
    
    def generate_context_enhanced_prompt(self, query: str, context_chunks: List[str]) -> Dict:
        """
        Generate a prompt that better utilizes context chunks
        
        Args:
            query: User's question
            context_chunks: List of retrieved context chunks
            
        Returns:
            Dictionary with enhanced prompt
        """
        # Organize context by relevance
        organized_context = self._organize_context_chunks(context_chunks)
        
        # Create enhanced system prompt
        system_prompt = """You are an expert assistant with access to detailed product documentation. 
        Your task is to provide accurate, helpful answers based on the provided context sections.
        
        Important guidelines:
        - Use information from ALL relevant context sections
        - If information conflicts between sections, mention this and provide the most recent/correct information
        - If the context doesn't fully answer the question, clearly state what additional information would be helpful
        - Provide specific details, part numbers, page references when available
        - Structure your answer logically with clear sections if needed"""
        
        # Create enhanced user prompt
        context_sections = []
        for i, (section_type, chunks) in enumerate(organized_context.items(), 1):
            context_sections.append(f"Section {i} ({section_type}):\n" + "\n".join(chunks))
        
        user_prompt = f"""Context Sections:
{chr(10).join(context_sections)}

Question: {query}

Please provide a comprehensive answer using all relevant information from the context sections above."""
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'context_sections': len(context_sections),
            'organized_context': organized_context
        }
    
    def _organize_context_chunks(self, chunks: List[str]) -> Dict:
        """Organize context chunks by type/relevance"""
        organized = {
            'general_info': [],
            'procedures': [],
            'specifications': [],
            'warnings': []
        }
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            
            # Categorize based on content
            if any(word in chunk_lower for word in ['warning', 'caution', 'danger', 'safety']):
                organized['warnings'].append(chunk)
            elif any(word in chunk_lower for word in ['step', 'procedure', 'install', 'setup']):
                organized['procedures'].append(chunk)
            elif any(word in chunk_lower for word in ['specification', 'dimension', 'weight', 'voltage']):
                organized['specifications'].append(chunk)
            else:
                organized['general_info'].append(chunk)
        
        # Remove empty categories
        return {k: v for k, v in organized.items() if v}
    
    def create_few_shot_prompt(self, query: str, context: str, examples: List[Dict]) -> Dict:
        """
        Create a few-shot prompt with examples
        
        Args:
            query: User's question
            context: Retrieved context
            examples: List of example Q&A pairs
            
        Returns:
            Dictionary with few-shot prompt
        """
        system_prompt = """You are a helpful assistant that answers questions based on product documentation. 
        Follow the pattern shown in the examples below."""
        
        # Build examples section
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"""
Example {i}:
Context: {example['context']}
Question: {example['question']}
Answer: {example['answer']}
"""
        
        user_prompt = f"""{examples_text}
Context: {context}
Question: {query}
Answer:"""
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'examples_count': len(examples)
        }
    
    def get_chain_of_thought_prompt(self, query: str, context: str) -> Dict:
        """
        Create a chain-of-thought prompt for complex reasoning
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Dictionary with chain-of-thought prompt
        """
        system_prompt = """You are a logical assistant that thinks through problems step by step. 
        When answering questions, first analyze the context, then reason through the answer, and finally provide a clear response."""
        
        user_prompt = f"""Context: {context}

Question: {query}

Let me think through this step by step:

1. First, I need to understand what information is available in the context...
2. Then, I should identify the key points relevant to the question...
3. Finally, I'll provide a clear answer based on this analysis...

Answer:"""
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'reasoning_style': 'chain_of_thought'
        } 