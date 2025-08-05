"""
Enhanced QA Engine with RAG optimizations
"""
import streamlit as st
from typing import List, Dict, Optional
from services.ai_service import AIService
from services.prompt_optimizer import PromptOptimizer
from services.advanced_retrieval import AdvancedRetrieval
from text_chunker.chunker import TextChunker
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SUPPORTED_LANGUAGES
from deep_translator import GoogleTranslator

class QAEngine:
    """Enhanced QA Engine with RAG optimizations"""
    
    def __init__(self, model_name: str = "gemini", language: str = "english"):
        self.ai_service = AIService(model_name)
        self.language = language
        self.prompt_optimizer = PromptOptimizer()
        self.advanced_retrieval = AdvancedRetrieval()
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="product_manual",
            embedding_function=self.embedding_function
        )
        
        # RAG optimization settings
        self.retrieval_strategy = "hybrid"  # hybrid, rerank, multi_query, semantic_filter
        self.prompt_style = "detailed"  # basic, detailed, step_by_step, troubleshooting
        self.use_enhanced_context = True
        self.use_chain_of_thought = False
    
    def process_document(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> bool:
        """
        Process and store document chunks in vector database
        
        Args:
            text: Document text to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing data
            self.collection.delete(where={})
            
            # Chunk the text
            chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.chunk_text(text)
            
            # Store chunks in vector database
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "source": "product_manual"
                })
                ids.append(f"chunk_{i}")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Processed {len(chunks)} chunks successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return False
    
    # def ask(self, question: str, output_language: str = None) -> str:
    #     """
    #     Ask a question and get an enhanced answer using RAG optimizations
        
    #     Args:
    #         question: User's question
    #         output_language: Language for the answer (defaults to self.language)
            
    #     Returns:
    #         Generated answer
    #     """
    #     try:
    #         language = output_language or self.language

    #         # 1. Generate answer in English
    #         english_answer = self.rag_chain.invoke({"question": question})

    #         # 2. Translate if needed
    #         if language.lower() != "english":
    #             translator = Translator()
    #             translated_answer = translator.translate(english_answer, dest=self.language_code(language)).text
    #             return translated_answer

    #         return english_answer

    #     except Exception as e:
    #         return f"Error: {str(e)}"
    #     """try:
    #         # Use specified language or default
    #         language = output_language or self.language
            
    #         # Retrieve relevant chunks
    #         chunks = self.get_similar_chunks(question, n_results=8)
            
        
    #         # Apply advanced retrieval strategies
    #         if self.use_enhanced_context:
    #             enhanced_result = self.advanced_retrieval.get_enhanced_context(
    #                 question, chunks, self.retrieval_strategy
    #             )
    #             context = enhanced_result['enhanced_context']
    #             retrieval_metadata = enhanced_result
    #         else:
    #             context = "\n\n".join(chunks)
    #             retrieval_metadata = {}
            
    #         # Generate optimized prompt
    #         if self.use_chain_of_thought:
    #             prompt_data = self.prompt_optimizer.get_chain_of_thought_prompt(question, context)
    #         else:
    #             prompt_data = self.prompt_optimizer.get_optimized_prompt(
    #                 question, context, self.prompt_style
    #             )
            
    #         lang_config = SUPPORTED_LANGUAGES.get(output_language, SUPPORTED_LANGUAGES["english"])
    #         system_prompt = lang_config["system"]
    #         instructions = lang_config["instructions"]

    #         # Create final prompt
    #         user_prompt = f"{instructions}\n\nContext:\n{context}\n\nQuestion:\n{question}"

    #         # Generate answer
    #         answer = self.ai_service.generate_response(
    #             user_prompt,      
    #             system_prompt,
    #             language
    #         )

            
    #         # Store metadata for analysis
    #         self._store_query_metadata(question, answer, retrieval_metadata, prompt_data)
            
    #         return answer
            
    #     except Exception as e:
    #         print(f"❌ Error generating answer: {str(e)}")
    #         return f"Sorry, I encountered an error while processing your question: {str(e)}" 
    #         """
    def ask(self, question: str, output_language: str = None) -> str:
        """
        Ask a question and get an enhanced answer using RAG optimizations, with optional translation
        
        Args:
            question: User's question
            output_language: Language for the answer (defaults to self.language)
            
        Returns:
            Generated answer (translated if needed)
        """
        try:
            # Use specified language or fallback
            language = output_language or self.language
            
            # Step 1: Retrieve relevant chunks
            chunks = self.get_similar_chunks(question, n_results=8)

            # Step 2: Enhanced context from retrieval strategy
            if self.use_enhanced_context:
                enhanced_result = self.advanced_retrieval.get_enhanced_context(
                    question, chunks, self.retrieval_strategy
                )
                context = enhanced_result['enhanced_context']
                retrieval_metadata = enhanced_result
            else:
                context = "\n\n".join(chunks)
                retrieval_metadata = {}

            # Step 3: Optimize the prompt
            if self.use_chain_of_thought:
                prompt_data = self.prompt_optimizer.get_chain_of_thought_prompt(question, context)
            else:
                prompt_data = self.prompt_optimizer.get_optimized_prompt(
                    question, context, self.prompt_style
                )

            lang_config = SUPPORTED_LANGUAGES.get(language.lower(), SUPPORTED_LANGUAGES["english"])
            system_prompt = lang_config["system"]
            instructions = lang_config["instructions"]

            # Final prompt
            user_prompt = f"{instructions}\n\nContext:\n{context}\n\nQuestion:\n{question}"

            # Step 4: Generate response using the selected LLM
            answer = self.ai_service.generate_response(
                user_prompt, system_prompt, language
            )

            # Step 5: Optional Translation to non-English
            if language.lower() != "english":
                answer = GoogleTranslator(source='auto', target=self.language_code(language)).translate(answer)

            # Step 6: Store metadata
            self._store_query_metadata(question, answer, retrieval_metadata, prompt_data)

            return answer

        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

    def language_code(self, lang):
        """Convert full language name to ISO code for translation"""
        codes = {
            "english": "en",
            "hindi": "hi",
            "marathi": "mr",
            "german": "de"
        }
        return codes.get(lang.lower(), "en")

        

    def get_similar_chunks(self, query: str, n_results: int = 5) -> List[str]:
        """
        Get similar chunks from vector database
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar chunks
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results['documents']:
                return results['documents'][0]
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving chunks: {str(e)}")
            return []
    
    def _store_query_metadata(self, question: str, answer: str, retrieval_metadata: Dict, prompt_data: Dict):
        """Store metadata about the query for analysis"""
        # This could be used for improving the system over time
        metadata = {
            'question': question,
            'answer': answer,
            'retrieval_strategy': retrieval_metadata.get('strategy', 'unknown'),
            'prompt_style': prompt_data.get('style', 'unknown'),
            'query_type': prompt_data.get('query_type', 'unknown'),
            'chunks_analyzed': retrieval_metadata.get('chunks_analyzed', 0),
            'top_chunks': retrieval_metadata.get('top_chunks', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in session state for analysis
        if 'query_metadata' not in st.session_state:
            st.session_state.query_metadata = []
        
        st.session_state.query_metadata.append(metadata)
    
    def set_retrieval_strategy(self, strategy: str):
        """Set the retrieval strategy"""
        if strategy in ['hybrid', 'rerank', 'multi_query', 'semantic_filter']:
            self.retrieval_strategy = strategy
            print(f"✅ Retrieval strategy set to: {strategy}")
        else:
            print(f"❌ Invalid retrieval strategy: {strategy}")
    
    def set_prompt_style(self, style: str):
        """Set the prompt style"""
        if style in ['basic', 'detailed', 'step_by_step', 'troubleshooting']:
            self.prompt_style = style
            print(f"✅ Prompt style set to: {style}")
        else:
            print(f"❌ Invalid prompt style: {style}")
    
    def toggle_enhanced_context(self, enabled: bool):
        """Toggle enhanced context processing"""
        self.use_enhanced_context = enabled
        print(f"✅ Enhanced context: {'enabled' if enabled else 'disabled'}")
    
    def toggle_chain_of_thought(self, enabled: bool):
        """Toggle chain-of-thought reasoning"""
        self.use_chain_of_thought = enabled
        print(f"✅ Chain-of-thought: {'enabled' if enabled else 'disabled'}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the QA engine"""
        try:
            # Get collection info
            collection_count = self.collection.count()
            
            # Get recent query metadata
            recent_queries = st.session_state.get('query_metadata', [])
            
            # Calculate metrics
            total_queries = len(recent_queries)
            avg_chunks_analyzed = 0
            strategy_usage = {}
            
            if recent_queries:
                avg_chunks_analyzed = sum(q.get('chunks_analyzed', 0) for q in recent_queries) / total_queries
                
                # Count strategy usage
                for query in recent_queries:
                    strategy = query.get('retrieval_strategy', 'unknown')
                    strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            return {
                'total_chunks': collection_count,
                'total_queries': total_queries,
                'avg_chunks_analyzed': round(avg_chunks_analyzed, 2),
                'strategy_usage': strategy_usage,
                'current_strategy': self.retrieval_strategy,
                'current_prompt_style': self.prompt_style,
                'enhanced_context_enabled': self.use_enhanced_context,
                'chain_of_thought_enabled': self.use_chain_of_thought
            }
            
        except Exception as e:
            print(f"❌ Error getting performance metrics: {str(e)}")
            return {}
    
    def test_retrieval_strategies(self, query: str) -> Dict:
        """Test different retrieval strategies on a query"""
        try:
            # Get basic chunks
            basic_chunks = self.get_similar_chunks(query, n_results=8)
            
            if not basic_chunks:
                return {'error': 'No chunks found for query'}
            
            # Test each strategy
            results = {}
            for strategy in ['hybrid', 'rerank', 'multi_query', 'semantic_filter']:
                enhanced_result = self.advanced_retrieval.get_enhanced_context(
                    query, basic_chunks, strategy
                )
                results[strategy] = {
                    'chunks_analyzed': enhanced_result.get('chunks_analyzed', 0),
                    'top_chunks': enhanced_result.get('top_chunks', 0),
                    'context_length': len(enhanced_result.get('enhanced_context', '')),
                    'metadata': {k: v for k, v in enhanced_result.items() 
                               if k not in ['enhanced_context']}
                }
            
            return results
            
        except Exception as e:
            print(f"❌ Error testing retrieval strategies: {str(e)}")
            return {'error': str(e)}
    
    def compare_prompt_styles(self, query: str, context: str) -> Dict:
        """Compare different prompt styles"""
        try:
            results = {}
            for style in ['basic', 'detailed', 'step_by_step', 'troubleshooting']:
                prompt_data = self.prompt_optimizer.get_optimized_prompt(
                    query, context, style
                )
                results[style] = {
                    'system_prompt_length': len(prompt_data.get('system_prompt', '')),
                    'user_prompt_length': len(prompt_data.get('user_prompt', '')),
                    'query_type': prompt_data.get('query_type', 'unknown'),
                    'style': prompt_data.get('style', 'unknown')
                }
            
            return results
            
        except Exception as e:
            print(f"❌ Error comparing prompt styles: {str(e)}")
            return {'error': str(e)} 