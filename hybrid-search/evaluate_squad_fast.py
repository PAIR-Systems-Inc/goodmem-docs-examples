#!/usr/bin/env python3
"""
Fast SQuAD Retrieval Evaluation - Optimized for Large Datasets with Dynamic Embedder Configuration

Optimizations:
1. Dynamic embedder discovery from space configuration
2. Batch retrieval requests
3. Concurrent API calls  
4. Reduced API overhead
5. Efficient memory availability check
6. Progress checkpointing

Usage Examples:
    # Basic usage (automatically finds ground truth file based on space ID):
    python evaluate_squad_fast.py --space-id SPACE-ID
    
    # With custom performance settings:
    python evaluate_squad_fast.py --space-id SPACE-ID --batch-size 200 --threads 8
    
    # With custom embedder weights (JSON format):
    python evaluate_squad_fast.py --space-id SPACE-ID --custom-weights '{"embedder-id-1": 1.5, "embedder-id-2": 0.8}'
    
    # Limited evaluation for testing:
    python evaluate_squad_fast.py --space-id SPACE-ID --limit 1000 --top-k 5
    
    # Override automatic ground truth detection:
    python evaluate_squad_fast.py --space-id SPACE-ID --ground-truth custom_ground_truth.json

Parameters:
    --space-id        GoodMem space ID containing the memories
    --ground-truth    Optional: Path to ground truth JSON file (auto-detected if not specified)
    --custom-weights  Optional JSON string of embedder weights (overrides space defaults)
    --batch-size      Questions per batch (default: 100, recommended: 100-500)
    --threads         Concurrent threads (default: 4, recommended: 4-8)
    --top-k           Number of results to retrieve per question (default: 10)
    --limit           Limit evaluation to N questions (for testing)
"""

import json
import logging
import os
import sys
import time
from typing import List, Dict, Optional
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add path for updated streaming client
# sys.path.insert(0, '/path/to/goodmem/clients/python')  # Uncomment and update path if needed

# Import GoodMem client
try:
    from goodmem_client import (
        MemoriesApi, Configuration, ApiClient, SpacesApi
    )
    from goodmem_client.streaming import MemoryStreamClient
    from goodmem_client.models.space_key import SpaceKey
    from goodmem_client.models.embedder_weight import EmbedderWeight
except ImportError as e:
    logging.error(f"Failed to import GoodMem client: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# GoodMem configuration
GOODMEM_API_KEY = "your_api_key_here"  # Replace with your actual GoodMem API key
GOODMEM_SERVER_URL = "http://localhost:8080"

# Dynamic embedder configuration will be fetched from space


class FastSquadEvaluator:
    def __init__(self, space_id: str, num_threads: int = 4):
        self.space_id = space_id
        self.num_threads = num_threads
        
        # Create multiple clients for concurrent requests
        self.clients = [self._create_stream_client() for _ in range(num_threads)]
        self.client_lock = threading.Lock()
        self.available_clients = list(range(num_threads))
        
        # Fetch embedders and weights from space configuration
        self.embedders_config = self._fetch_space_embedders()
        
    def _create_stream_client(self) -> MemoryStreamClient:
        """Create streaming client for retrieval"""
        configuration = Configuration()
        configuration.host = GOODMEM_SERVER_URL
        
        api_client = ApiClient(configuration)
        api_client.configuration.api_key = {"ApiKeyAuth": GOODMEM_API_KEY}
        
        return MemoryStreamClient(api_client)
    
    def _fetch_space_embedders(self) -> List[Dict]:
        """Fetch embedders and their weights from space configuration"""
        try:
            configuration = Configuration()
            configuration.host = GOODMEM_SERVER_URL
            
            api_client = ApiClient(configuration)
            api_client.configuration.api_key = {"ApiKeyAuth": GOODMEM_API_KEY}
            
            spaces_api = SpacesApi(api_client)
            
            # Get space details
            space_response = spaces_api.get_space(id=self.space_id)
            
            embedders_config = []
            
            # Extract embedders and weights from space configuration
            # Handle both possible API response formats
            space_embedders = None
            if hasattr(space_response, 'space_embedders') and space_response.space_embedders:
                space_embedders = space_response.space_embedders
            elif hasattr(space_response, 'embedder_weights') and space_response.embedder_weights:
                space_embedders = space_response.embedder_weights
                
            if space_embedders:
                for embedder_item in space_embedders:
                    # Handle different attribute names
                    embedder_id = getattr(embedder_item, 'embedder_id', 
                                        getattr(embedder_item, 'embedderID', None))
                    weight = getattr(embedder_item, 'weight', 
                                   getattr(embedder_item, 'default_retrieval_weight', 
                                          getattr(embedder_item, 'defaultRetrievalWeight', 1.0)))
                    
                    if embedder_id:
                        embedders_config.append({
                            'embedder_id': embedder_id,
                            'weight': weight
                        })
            
            if not embedders_config:
                logger.warning(f"No embedders found for space {self.space_id}")
                return []
            
            logger.info(f"📋 Found {len(embedders_config)} embedders for space {self.space_id}:")
            for config in embedders_config:
                logger.info(f"   • Embedder {config['embedder_id']}: weight={config['weight']}")
            
            # Output embedder info in machine-readable format for optimization script
            embedder_info = {
                'space_id': self.space_id,
                'embedders': embedders_config
            }
            results_dir = f"results_{self.space_id}"
            os.makedirs(results_dir, exist_ok=True)
            embedder_info_file = os.path.join(results_dir, "embedder_info.json")
            try:
                with open(embedder_info_file, 'w') as f:
                    json.dump(embedder_info, f, indent=2)
                logger.info(f"🔧 Embedder info saved to: {embedder_info_file}")
            except Exception as e:
                logger.warning(f"Failed to save embedder info: {e}")
            
            return embedders_config
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch space embedders: {e}")
            return []
    
    def _apply_custom_weights(self, custom_weights: Dict):
        """Apply custom weights to embedders by embedder ID"""
        for config in self.embedders_config:
            embedder_id = config['embedder_id']
            if embedder_id in custom_weights:
                old_weight = config['weight']
                config['weight'] = custom_weights[embedder_id]
                logger.info(f"🔄 Updated embedder {embedder_id[:8]}... weight: {old_weight} -> {config['weight']}")
    
    def _get_client(self) -> tuple[MemoryStreamClient, int]:
        """Get an available client (thread-safe)"""
        while True:
            with self.client_lock:
                if self.available_clients:
                    client_idx = self.available_clients.pop()
                    return self.clients[client_idx], client_idx
            time.sleep(0.01)  # Brief wait if no clients available
    
    def _return_client(self, client_idx: int):
        """Return client to pool (thread-safe)"""
        with self.client_lock:
            self.available_clients.append(client_idx)
    
    def _refresh_client_pool(self):
        """Refresh the entire client pool to reset connections"""
        logger.info(f"🔄 Refreshing client pool to reset connections...")
        self.clients = [self._create_stream_client() for _ in range(self.num_threads)]
        self.available_clients = list(range(self.num_threads))
        logger.info(f"✅ Client pool refreshed with {self.num_threads} new clients")
    
    def retrieve_batch(self, questions_batch: List[Dict], top_k: int = 10) -> List[Dict]:
        """Process a batch of questions concurrently using client pool"""
        def process_single_question(question_data):
            client, client_idx = self._get_client()
            try:
                result = self._retrieve_single_fast(question_data, client, top_k)
                return result
            except Exception as e:
                return {
                    'question_id': question_data.get('question_id', 'unknown'),
                    'question_text': question_data.get('question_text', ''),
                    'correct_rank': float('inf'),
                    'found_in_results': False,
                    'retrieved_count': 0,
                    'error': f'Retrieval failed: {str(e)}'
                }
            finally:
                self._return_client(client_idx)
        
        # Process batch concurrently
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_question = {
                executor.submit(process_single_question, q): q 
                for q in questions_batch
            }
            
            for future in as_completed(future_to_question):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per question
                    results.append(result)
                except Exception as e:
                    question_data = future_to_question[future]
                    results.append({
                        'question_id': question_data['question_id'],
                        'question_text': question_data['question_text'],
                        'correct_rank': float('inf'),
                        'found_in_results': False,
                        'error': str(e)
                    })
        
        return results
    
    def _retrieve_single_fast(self, question_data: Dict, client: MemoryStreamClient, top_k: int) -> Dict:
        """Fast retrieval for a single question"""
        question_text = question_data['question_text']
        correct_memory_id = question_data['correct_memory_id']
        question_id = question_data['question_id']
        
        try:
            # Create space key with weights from space configuration
            if not self.embedders_config:
                logger.error(f"❌ No embedders configured for space {self.space_id}")
                return {
                    'question_id': question_id,
                    'question_text': question_text,
                    'correct_rank': float('inf'),
                    'found_in_results': False,
                    'retrieved_count': 0,
                    'error': 'No embedders configured for space'
                }
                
            embedder_weights = [
                EmbedderWeight(embedder_id=config['embedder_id'], weight=config['weight'])
                for config in self.embedders_config
            ]
            
            space_key = SpaceKey(
                space_id=self.space_id,
                embedder_weights=embedder_weights
            )
            
            # Fast retrieval - collect only what we need
            retrieved_memory_ids = []
            memory_definitions = {}
            
            for event in client.retrieve_memory_stream(
                message=question_text,
                space_keys=[space_key],
                requested_size=top_k
            ):
                # Quick memory definition collection
                if hasattr(event, 'memory_definition') and event.memory_definition:
                    memory = event.memory_definition
                    # Try different ways to get memory ID
                    if isinstance(memory, dict):
                        memory_id = memory.get('memoryId', memory.get('memory_id', memory.get('id', 'unknown')))
                    else:
                        memory_id = getattr(memory, 'memory_id', 
                                   getattr(memory, 'memoryId', 
                                   getattr(memory, 'id', 'unknown')))
                    memory_definitions[len(memory_definitions)] = str(memory_id)
                
                # Quick chunk collection
                if hasattr(event, 'retrieved_item') and event.retrieved_item:
                    retrieved_item = event.retrieved_item
                    
                    memory_index = getattr(retrieved_item, 'memory_index', None)
                    if memory_index is None and hasattr(retrieved_item, 'chunk'):
                        chunk_ref = retrieved_item.chunk
                        memory_index = getattr(chunk_ref, 'memory_index', None)
                    
                    if memory_index is not None and memory_index in memory_definitions:
                        memory_id = memory_definitions[memory_index]
                        if memory_id not in retrieved_memory_ids:
                            retrieved_memory_ids.append(memory_id)
                    elif hasattr(retrieved_item, 'memory_id'):
                        memory_id = retrieved_item.memory_id
                        if memory_id not in retrieved_memory_ids:
                            retrieved_memory_ids.append(memory_id)
            
            # Find rank quickly
            correct_rank = float('inf')
            found_in_results = False
            
            for rank, memory_id in enumerate(retrieved_memory_ids[:top_k], 1):
                if memory_id == correct_memory_id:
                    correct_rank = rank
                    found_in_results = True
                    break
            
            # Debug logging for investigation
            if len(retrieved_memory_ids) == 0:
                logger.debug(f"⚠️ No memories retrieved for question {question_id[:8]}...: {question_text[:50]}...")
            
            return {
                'question_id': question_id,
                'question_text': question_text,
                'correct_rank': correct_rank,
                'found_in_results': found_in_results,
                'retrieved_count': len(retrieved_memory_ids)
            }
            
        except Exception as e:
            return {
                'question_id': question_id,
                'question_text': question_text,
                'correct_rank': float('inf'),
                'found_in_results': False,
                'error': str(e)
            }
    
    def fast_evaluate_with_checkpointing(self, ground_truth_file: str, top_k: int = 10, batch_size: int = 100, 
                                       limit: Optional[int] = None, custom_weights: Optional[Dict] = None) -> Dict:
        """Fast evaluation with checkpointing and batching"""
        logger.info("🚀 FAST SQuAD Retrieval Evaluation")
        logger.info("=" * 60)
        logger.info(f"Batch size: {batch_size}, Threads: {self.num_threads}")
        
        # Create results directory for this space
        results_dir = f"results_{self.space_id}"
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"📁 Results directory: {results_dir}")
        
        # Load and filter ground truth
        logger.info("📥 Loading ground truth...")
        try:
            with open(ground_truth_file, 'r') as f:
                all_questions = json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load ground truth: {e}")
            return {}
        
        # Filter to questions with memory IDs
        questions_data = [q for q in all_questions if q.get('correct_memory_id')]
        
        # DETERMINISM: Sort questions by question_id to ensure consistent ordering
        questions_data.sort(key=lambda x: x.get('question_id', ''))
        logger.info(f"🔄 Sorted questions by question_id for deterministic results")
        
        # Apply limit if specified
        if limit and limit < len(questions_data):
            questions_data = questions_data[:limit]
            logger.info(f"⚠️  Limited to {limit:,} questions for testing")
        
        logger.info(f"✅ Evaluating {len(questions_data):,} questions with available memories")
        
        # DIAGNOSTIC: Log first few question IDs for reproducibility verification
        first_questions = [q['question_id'] for q in questions_data[:5]]
        logger.info(f"🔍 First 5 question IDs: {first_questions}")
        
        # Apply custom weights if provided
        if custom_weights:
            self._apply_custom_weights(custom_weights)
        
        # Create checkpoint file name based on current embedder configuration
        weights_str = '_'.join([f"{config['embedder_id'][:8]}w{config['weight']}" for config in self.embedders_config])
        checkpoint_file = os.path.join(results_dir, f"eval_checkpoint_{weights_str}.json")
        completed_results = []
        start_batch = 0
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed_results = checkpoint['results']
                start_batch = checkpoint['next_batch']
            logger.info(f"📂 Resuming from batch {start_batch}, {len(completed_results)} completed")
        except FileNotFoundError:
            logger.info("🆕 Starting fresh evaluation")
        
        # Process in batches
        total_batches = (len(questions_data) + batch_size - 1) // batch_size
        
        logger.info(f"🎯 Processing {total_batches} batches from batch {start_batch}")
        
        all_results = completed_results
        
        for batch_idx in tqdm(range(start_batch, total_batches), desc="Processing batches"):
            # PERIODIC CLIENT REFRESH: Every 50 batches or ~2 minutes to prevent session timeout
            if batch_idx > 0 and batch_idx % 50 == 0:
                logger.info(f"🔄 Batch {batch_idx}: Refreshing client pool (every 50 batches)")
                self._refresh_client_pool()
                time.sleep(1)  # Brief pause after refresh
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(questions_data))
            batch_questions = questions_data[start_idx:end_idx]
            
            # Process batch concurrently
            batch_start_time = time.time()
            batch_results = self.retrieve_batch(batch_questions, top_k)
            batch_time = time.time() - batch_start_time
            
            # DIAGNOSTIC: Analyze batch results for anomalies
            found_count = sum(1 for r in batch_results if r.get('found_in_results', False))
            error_count = sum(1 for r in batch_results if 'error' in r)
            zero_retrieved = sum(1 for r in batch_results if r.get('retrieved_count', 0) == 0)
            
            all_results.extend(batch_results)
            
            # Save checkpoint every 10 batches
            if (batch_idx + 1) % 10 == 0:
                checkpoint = {
                    'next_batch': batch_idx + 1,
                    'results': all_results,
                    'timestamp': time.time()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
            
            # Update progress with diagnostic info
            rate = len(batch_results) / batch_time if batch_time > 0 else 0
            
            # Flag suspicious ultra-fast batches
            is_suspicious = rate > 1000  # More than 1000 q/s is suspicious
            
            if is_suspicious:
                # Log details of suspicious batches for analysis
                suspicious_questions = [q['question_id'] for q in batch_questions[:3]]  # First 3 question IDs
                zero_retrieved_questions = [r['question_id'] for r in batch_results if r.get('retrieved_count', 0) == 0][:3]
                tqdm.write(f"⚠️  SUSPICIOUS Batch {batch_idx+1}: {len(batch_results)} questions in {batch_time:.3f}s ({rate:.1f} q/s) - Found: {found_count}, Errors: {error_count}, Zero retrieved: {zero_retrieved}")
                tqdm.write(f"      Sample questions: {suspicious_questions}")
                if zero_retrieved_questions:
                    tqdm.write(f"      Zero retrieved questions: {zero_retrieved_questions}")
            else:
                tqdm.write(f"Batch {batch_idx+1}: {len(batch_results)} questions in {batch_time:.1f}s ({rate:.1f} q/s) - Found: {found_count}")
        
        # Compute final metrics and identify missing terms
        metrics = self._compute_metrics_fast(all_results)
        
        # Extract questions where retrieval failed to find the correct answer
        missing_terms = []
        for result in all_results:
            if not result.get('found_in_results', False) and 'error' not in result:
                # This question has a memory but retrieval missed it
                missing_entry = {
                    'question_id': result['question_id'],
                    'question_text': result['question_text'],
                    'correct_memory_id': questions_data[0]['correct_memory_id'] if questions_data else None,  # Will fix this
                    'retrieved_count': result.get('retrieved_count', 0)
                }
                missing_terms.append(missing_entry)
        
        # Find the original question data for missing terms to get correct info
        question_lookup = {q['question_id']: q for q in questions_data}
        for missing in missing_terms:
            original_q = question_lookup.get(missing['question_id'])
            if original_q:
                missing['correct_memory_id'] = original_q['correct_memory_id']
                missing['correct_answer_text'] = original_q['correct_answer_text']
                missing['correct_sentence_text'] = original_q['correct_sentence_text']
                missing['article_title'] = original_q['article_title']
        
        # Save missing terms
        weights_str = '_'.join([f"{config['embedder_id'][:8]}w{config['weight']}" for config in self.embedders_config])
        missing_file = os.path.join(results_dir, f"missingTerms_{weights_str}.json")
        try:
            with open(missing_file, 'w') as f:
                json.dump(missing_terms, f, indent=2)
            logger.info(f"📝 Missing terms saved to: {missing_file}")
            logger.info(f"📊 {len(missing_terms):,} questions missed by retrieval")
        except Exception as e:
            logger.warning(f"Failed to save missing terms: {e}")
        
        # DIAGNOSTIC: Analyze overall results for anomalies
        total_zero_retrieved = sum(1 for r in all_results if r.get('retrieved_count', 0) == 0)
        total_errors = sum(1 for r in all_results if 'error' in r)
        
        # Display results
        logger.info(f"\n{'='*60}")
        logger.info("📊 FAST EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        # Display embedder weights dynamically
        weights_info = ', '.join([f"{config['embedder_id'][:8]}...={config['weight']}" for config in self.embedders_config])
        logger.info(f"⚖️  Embedder Weights: {weights_info}")
        logger.info(f"📋 Questions evaluated: {metrics['total_questions']:,}")
        logger.info(f"📋 Questions found: {metrics['questions_with_answers_found']:,}")
        logger.info(f"📋 Coverage: {metrics['questions_with_answers_found']/metrics['total_questions']*100:.1f}%")
        logger.info(f"📋 Zero retrieved: {total_zero_retrieved:,} ({total_zero_retrieved/metrics['total_questions']*100:.1f}%)")
        logger.info(f"📋 Errors: {total_errors:,} ({total_errors/metrics['total_questions']*100:.1f}%)")
        logger.info(f"\n🎯 Performance Metrics:")
        logger.info(f"   • MRR: {metrics['mrr']:.6f}")
        logger.info(f"   • Recall@1: {metrics['recall_at_1']:.6f}")
        logger.info(f"   • Recall@5: {metrics['recall_at_5']:.6f}")
        logger.info(f"   • Recall@10: {metrics['recall_at_10']:.6f}")
        
        # Save final results
        results_file = os.path.join(results_dir, f"squad_fast_eval_results_{weights_str}.json")
        # Create weights dict for results
        weights_dict = {config['embedder_id']: config['weight'] for config in self.embedders_config}
        
        final_results = {
            'evaluation_timestamp': time.time(),
            'space_id': self.space_id,
            'embedders_config': self.embedders_config,
            'weights': weights_dict,
            'metrics': metrics,
            'individual_results': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
        # Clean up checkpoint
        try:
            os.remove(checkpoint_file)
        except:
            pass
        
        return final_results
    
    def _compute_metrics_fast(self, results: List[Dict]) -> Dict:
        """Fast metrics computation"""
        ranks = {}
        
        for result in results:
            if result.get('found_in_results', False):
                ranks[result['question_id']] = result['correct_rank']
        
        if not ranks:
            return {
                'total_questions': len(results),
                'questions_with_answers_found': 0,
                'mrr': 0.0,
                'recall_at_1': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0
            }
        
        # Fast calculations
        total_questions = len(results)
        mrr = sum(1/rank for rank in ranks.values()) / total_questions
        
        def recall_at_n(n):
            return sum(1 for rank in ranks.values() if rank <= n) / total_questions
        
        return {
            'total_questions': total_questions,
            'questions_with_answers_found': len(ranks),
            'mrr': mrr,
            'recall_at_1': recall_at_n(1),
            'recall_at_5': recall_at_n(5),
            'recall_at_10': recall_at_n(10)
        }


def main():
    """Main execution with optimizations"""
    parser = argparse.ArgumentParser(description="Fast SQuAD retrieval evaluation with dynamic embedder configuration")
    parser.add_argument("--ground-truth", type=str, default=None,
                       help="Path to ground truth JSON file (auto-detected if not specified)")
    parser.add_argument("--space-id", type=str, required=True)
    parser.add_argument("--custom-weights", type=str, default=None,
                       help="JSON string of custom weights, e.g. '{\"embedder-id-1\": 1.5, \"embedder-id-2\": 0.8}'")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100, 
                       help="Questions per batch (default: 100)")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of concurrent threads (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of questions to evaluate (default: all)")
    
    args = parser.parse_args()
    
    # Auto-detect ground truth file if not provided
    if args.ground_truth is None:
        ground_truth_file = f"squad_1.1_ground_truth_{args.space_id}.json"
        logger.info(f"🔍 Auto-detecting ground truth file: {ground_truth_file}")
        
        # Check if the file exists
        if not os.path.exists(ground_truth_file):
            logger.error(f"❌ Auto-detected ground truth file not found: {ground_truth_file}")
            logger.error("💡 Either provide --ground-truth explicitly or ensure the file follows the naming pattern:")
            logger.error(f"   squad_1.1_ground_truth_{{SPACE-ID}}.json")
            sys.exit(1)
        
        args.ground_truth = ground_truth_file
        logger.info(f"✅ Using auto-detected ground truth file: {ground_truth_file}")
    else:
        logger.info(f"📂 Using provided ground truth file: {args.ground_truth}")
    
    logger.info(f"🔬 FAST SQUAD EVALUATION (Dynamic Embedders)")
    logger.info(f"Threads: {args.threads}, Batch size: {args.batch_size}")
    
    evaluator = FastSquadEvaluator(args.space_id, args.threads)
    
    # Parse custom weights if provided
    custom_weights = None
    if args.custom_weights:
        try:
            custom_weights = json.loads(args.custom_weights)
            logger.info(f"🎯 Custom weights provided: {custom_weights}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in custom weights: {e}")
            sys.exit(1)
    
    try:
        results = evaluator.fast_evaluate_with_checkpointing(
            ground_truth_file=args.ground_truth,
            top_k=args.top_k,
            batch_size=args.batch_size,
            limit=args.limit,
            custom_weights=custom_weights
        )
        
        if results:
            logger.info("🎉 Fast evaluation completed!")
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\n⏸️ Evaluation interrupted - progress saved in checkpoint")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()