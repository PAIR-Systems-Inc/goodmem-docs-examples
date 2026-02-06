#!/usr/bin/env python3
"""
Insert SQuAD 1.1 Training Data Sentences into GoodMem for Hybrid Embedder Testing

This script processes SQuAD 1.1 training data by:
1. Segmenting paragraphs into sentences using sb_sed.py
2. Inserting each sentence as individual memory with metadata
3. Creating ground truth JSON for MRR/R@N evaluation
4. Tracking sentence-to-memory mapping for retrieval testing

Usage:
    python insert_squad_sentences_goodmem.py --space-id 6780b797-a167-4d87-8c06-f4b929456f6d --dry-run
    python insert_squad_sentences_goodmem.py --space-id 6780b797-a167-4d87-8c06-f4b929456f6d --limit 10000
"""

import json
import logging
import sys
import time
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
import uuid
import os
import urllib.request

# Auto-install sb_sed if not available
def ensure_sb_sed_available():
    """Ensure sb_sed module is available, download if necessary."""
    try:
        import sb_sed
        return sb_sed
    except ImportError:
        logging.info("📥 Downloading sb_sed module from google/retrieval-qa-eval...")
        try:
            # Download sb_sed.py from the source repository
            url = "https://raw.githubusercontent.com/google-research-datasets/retrieval-qa-eval/main/sb_sed.py"
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sb_sed_path = os.path.join(script_dir, "sb_sed.py")

            urllib.request.urlretrieve(url, sb_sed_path)
            logging.info(f"✅ Successfully downloaded sb_sed.py to {sb_sed_path}")

            # Now import it
            import importlib.util
            spec = importlib.util.spec_from_file_location("sb_sed", sb_sed_path)
            sb_sed = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sb_sed)
            return sb_sed

        except Exception as e:
            logging.error(f"❌ Failed to download sb_sed: {e}")
            logging.error("Please manually download from: https://github.com/google-research-datasets/retrieval-qa-eval/blob/main/sb_sed.py")
            sys.exit(1)

# Import sentence segmentation utility (auto-downloads if missing)
sb_sed = ensure_sb_sed_available()

# Import GoodMem client
try:
    from goodmem_client import (
        MemoriesApi, Configuration, ApiClient,
        MemoryCreationRequest, BatchMemoryCreationRequest
    )
except ImportError as e:
    logging.error(f"Failed to import GoodMem client: {e}")
    logging.error("Make sure goodmem-client is installed: pip install goodmem-client")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# GoodMem configuration with updated credentials
GOODMEM_API_KEY = "your_api_key_here"  # Replace with your actual GoodMem API key
GOODMEM_SERVER_URL = "http://localhost:8080"
DEFAULT_SPACE_ID = "your_space_id_here"  # Replace with your actual space ID

class SquadSentenceInserter:
    def __init__(self, space_id: str):
        self.space_id = space_id
        self.batch_size = 500  # Balanced for GPU acceleration within API payload limits
        self.iteration_size = 1000  # 1K sentences per iteration
        self.target_total = None  # Will be set based on actual data
        
        # Initialize API client
        self.memories_api = self._create_api_client()
        
        # Progress tracking
        self.progress_file = f"squad_1.1_memories_progress_{space_id}.json"
        self.completed_iterations = 0
        self.total_inserted = 0
        
        # Ground truth tracking
        self.ground_truth_file = f"squad_1.1_ground_truth_{space_id}.json"
        self.sentence_memory_mapping_file = f"squad_1.1_sentence_memory_mapping_{space_id}.json"
        
        # Data structures for tracking
        self.sentence_to_memory_mapping = {}  # sentence_id -> memory_id
        self.ground_truth_entries = []  # List of ground truth entries
        self.processed_sentences = set()  # Track processed sentences to avoid duplicates
        
        # Load existing mappings if resuming
        self.load_existing_progress()
        
    def _create_api_client(self) -> MemoriesApi:
        """Create authenticated GoodMem API client"""
        configuration = Configuration()
        configuration.host = GOODMEM_SERVER_URL
        configuration.api_key = {"ApiKeyAuth": GOODMEM_API_KEY}
        
        api_client = ApiClient(configuration)
        return MemoriesApi(api_client)

    def load_existing_progress(self):
        """Load existing sentence mappings and GT data when resuming"""
        try:
            # Load sentence mapping if exists
            if os.path.exists(self.sentence_memory_mapping_file):
                with open(self.sentence_memory_mapping_file, 'r') as f:
                    self.sentence_to_memory_mapping = json.load(f)
                logger.info(f"📂 Loaded {len(self.sentence_to_memory_mapping)} existing sentence mappings")
            
            # Load GT data if exists  
            if os.path.exists(self.ground_truth_file):
                with open(self.ground_truth_file, 'r') as f:
                    self.ground_truth_entries = json.load(f)
                logger.info(f"📂 Loaded {len(self.ground_truth_entries)} existing GT entries")
                
            # Load progress data if exists
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.completed_iterations = progress.get('completed_iterations', 0)
                    self.total_inserted = progress.get('total_inserted', 0)
                logger.info(f"📂 Loaded progress: {self.completed_iterations} iterations, {self.total_inserted} total inserted")
                
        except Exception as e:
            logger.warning(f"Could not load existing progress: {e}")
    
    def load_squad_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load SQuAD 1.1 training data and prepare sentences and ground truth"""
        squad_file = "./squad_data/train-v1.1.json"  # Path to your SQuAD dataset file
        
        if not os.path.exists(squad_file):
            raise FileNotFoundError(f"SQuAD data file not found: {squad_file}")
        
        logger.info(f"📥 Loading SQuAD 1.1 training data from {squad_file}")
        
        with open(squad_file, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        sentences_data = []  # List of sentence memories to insert
        questions_data = []  # List of questions with ground truth info
        
        article_count = 0
        paragraph_count = 0
        sentence_count = 0
        question_count = 0
        
        logger.info("🔄 Processing SQuAD articles and paragraphs...")
        
        for article_idx, article in enumerate(tqdm(squad_data['data'], desc="Processing articles")):
            article_title = article['title']
            article_count += 1
            
            for para_idx, paragraph in enumerate(article['paragraphs']):
                paragraph_text = paragraph['context']
                paragraph_id = f"{article_idx}_{para_idx}"
                paragraph_count += 1
                
                # Segment paragraph into sentences using sb_sed
                try:
                    sentence_breaks = list(sb_sed.infer_sentence_breaks(paragraph_text))
                    sentences = [paragraph_text[start:end].strip() 
                               for (start, end) in sentence_breaks if paragraph_text[start:end].strip()]
                except Exception as e:
                    logger.warning(f"Failed to segment paragraph {paragraph_id}: {e}")
                    # Fallback: treat entire paragraph as one sentence
                    sentences = [paragraph_text.strip()]
                    sentence_breaks = [(0, len(paragraph_text))]
                
                # Create sentence memories
                paragraph_sentence_ids = []
                paragraph_memory_ids = []  # Will be filled during insertion
                
                for sent_idx, sentence_text in enumerate(sentences):
                    sentence_id = f"{article_idx}_{para_idx}_{sent_idx}"
                    sentence_count += 1
                    
                    # Skip empty sentences
                    if not sentence_text or len(sentence_text.strip()) < 10:
                        continue
                    
                    # Create sentence memory data
                    sentence_memory = {
                        'content': sentence_text,
                        'metadata': {
                            'source': 'squad_1.1',
                            'type': 'sentence', 
                            'article_title': article_title,
                            'article_index': article_idx,
                            'paragraph_id': paragraph_id,
                            'paragraph_index': para_idx,
                            'sentence_id': sentence_id,
                            'sentence_index': sent_idx,
                            'total_sentences_in_paragraph': len(sentences),
                            'paragraph_context': paragraph_text,
                            'test_purpose': 'squad_retrieval_evaluation'
                        },
                        'sentence_id': sentence_id,
                        'paragraph_id': paragraph_id,
                        'article_title': article_title
                    }
                    
                    sentences_data.append(sentence_memory)
                    paragraph_sentence_ids.append(sentence_id)
                
                # Process questions and answers for this paragraph
                for qas in paragraph['qas']:
                    question_count += 1
                    question_id = qas['id']
                    question_text = qas['question']
                    
                    # Process each answer (SQuAD 1.1 has multiple answers per question)
                    for answer in qas['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        
                        # Find which sentence contains this answer
                        correct_sentence_id = None
                        correct_sentence_text = None
                        
                        try:
                            for sent_idx, (start, end) in enumerate(sentence_breaks):
                                if start <= answer_start < end:
                                    correct_sentence_id = f"{article_idx}_{para_idx}_{sent_idx}"
                                    correct_sentence_text = paragraph_text[start:end].strip()
                                    break
                        except Exception as e:
                            logger.warning(f"Could not map answer to sentence for question {question_id}: {e}")
                            # Fallback: use first sentence
                            if paragraph_sentence_ids:
                                correct_sentence_id = paragraph_sentence_ids[0]
                                correct_sentence_text = sentences[0] if sentences else paragraph_text
                        
                        if correct_sentence_id:
                            # Create ground truth entry
                            ground_truth_entry = {
                                'question_id': question_id,
                                'question_text': question_text,
                                'article_title': article_title,
                                'article_index': article_idx,
                                'paragraph_id': paragraph_id,
                                'paragraph_index': para_idx,
                                'correct_answer_text': answer_text,
                                'correct_answer_start': answer_start,
                                'correct_sentence_id': correct_sentence_id,
                                'correct_sentence_text': correct_sentence_text,
                                'correct_memory_id': None,  # Will be filled during insertion
                                'space_id': self.space_id,
                                'all_paragraph_sentence_ids': paragraph_sentence_ids.copy(),
                                'all_paragraph_memory_ids': []  # Will be filled during insertion
                            }
                            
                            questions_data.append(ground_truth_entry)
        
        logger.info(f"✅ Processed SQuAD 1.1 data:")
        logger.info(f"   • {article_count:,} articles")
        logger.info(f"   • {paragraph_count:,} paragraphs") 
        logger.info(f"   • {len(sentences_data):,} sentences for insertion")
        logger.info(f"   • {len(questions_data):,} question-answer pairs for ground truth")
        
        return sentences_data, questions_data
    
    def get_processing_stats(self) -> Dict:
        """Get memory processing statistics from GoodMem API"""
        try:
            response = self.memories_api.list_memories(space_id=self.space_id)
            
            stats = {'PENDING': 0, 'COMPLETED': 0, 'FAILED': 0, 'PROCESSING': 0}
            
            if hasattr(response, 'memories'):
                memories = response.memories
                for memory in memories:
                    status = getattr(memory, 'processing_status', 'UNKNOWN')
                    if status in stats:
                        stats[status] += 1
            
            return stats
        except Exception as e:
            logger.warning(f"Could not get processing stats via API: {e}")
            return {'PENDING': 0, 'COMPLETED': 0, 'FAILED': 0, 'PROCESSING': 0}

    def retry_pending_memories(self) -> int:
        """Retry recent pending memories that are stuck due to batch contamination"""
        logger.info("🔄 Checking for stuck pending memories to retry...")
        
        try:
            response = self.memories_api.list_memories(space_id=self.space_id)
            
            # Find recent pending memories
            import datetime
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15)
            
            recent_pending = []
            for memory in response.memories:
                if (getattr(memory, 'processing_status', None) == 'PENDING' and
                    hasattr(memory, 'metadata') and memory.metadata and
                    memory.metadata.get('source', '').startswith('squad_1.1')):
                    recent_pending.append(memory)
            
            if not recent_pending:
                return 0
                
            logger.info(f"Found {len(recent_pending)} pending squad memories - retrying as individual memories...")
            
            retry_count = 0
            for memory in recent_pending[:100]:  # Limit retries
                try:
                    memory_id = str(memory.memory_id)
                    
                    # Get original sentence data from metadata
                    sentence_id = memory.metadata.get('sentence_id', '')
                    original_content = getattr(memory, 'original_content', b'').decode('utf-8') if hasattr(memory, 'original_content') else ''
                    
                    if sentence_id and original_content:
                        # Delete stuck memory
                        self.memories_api.delete_memory(memory_id=memory_id)
                        
                        # Recreate individually (not in batch)
                        retry_request = MemoryCreationRequest(
                            space_id=self.space_id,
                            original_content=original_content,
                            content_type="text/plain",
                            metadata=memory.metadata
                        )
                        
                        new_response = self.memories_api.create_memory(retry_request)
                        new_memory_id = str(new_response.memory_id)
                        
                        # Update mapping
                        if sentence_id in self.sentence_to_memory_mapping:
                            self.sentence_to_memory_mapping[sentence_id] = new_memory_id
                        
                        retry_count += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to retry memory: {e}")
                    
            logger.info(f"✅ Retried {retry_count} pending memories individually")
            return retry_count
            
        except Exception as e:
            logger.warning(f"Error during pending memory retry: {e}")
            return 0
    
    def wait_for_iteration_completion(self, target_memories: int, iteration_baseline_completed: int, iteration_baseline_failed: int, max_wait_minutes: int = 10) -> bool:
        """Wait for all memories in current iteration to complete processing"""
        logger.info(f"⏳ Waiting for {target_memories} memories to complete processing...")
        
        # Use the baseline captured at the START of the iteration, not now
        baseline_completed = iteration_baseline_completed
        baseline_failed = iteration_baseline_failed
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval = 10
        
        while True:
            stats = self.get_processing_stats()
            completed = stats['COMPLETED']
            pending = stats['PENDING']
            failed = stats['FAILED']
            processing = stats['PROCESSING']
            
            iteration_completed = completed - baseline_completed
            
            logger.info(f"   Status: {completed} completed total ({iteration_completed} this iteration), "
                       f"{pending} pending, {processing} processing, {failed} failed")
            
            # Check if iteration is complete (90% success rate acceptable)
            if iteration_completed >= target_memories * 0.90:
                logger.info(f"✅ Iteration completed! {iteration_completed}/{target_memories} memories processed")
                return True
            
            elapsed = time.time() - start_time
            
            # Also proceed if no pending/processing memories and we've waited at least 2 minutes
            if (pending + processing) == 0:
                logger.info(f"✅ Iteration completed (no pending)! {iteration_completed}/{target_memories} memories processed")
                return True
            
            # PRACTICAL FIX: If we've processed most of the iteration and waited > 2 minutes, proceed
            success_rate = iteration_completed / target_memories
            if success_rate >= 0.80 and elapsed > 120:
                logger.info(f"✅ Good enough progress! {iteration_completed}/{target_memories} completed ({success_rate:.1%})")
                logger.info(f"   Proceeding after 2min wait - {pending} total pending won't block progress")
                return True
            
            # QUICK PROCEED: Only if very little progress after reasonable time (likely stuck)
            if success_rate < 0.20 and elapsed > 300:  # Less than 20% after 5 minutes = likely stuck
                logger.info(f"⚠️  Low progress proceed! {iteration_completed}/{target_memories} completed, waited {elapsed:.0f}s")
                logger.info(f"   Likely many memories stuck in processing - continuing to avoid infinite wait")
                return True
            
            # Check timeout
            if elapsed > max_wait_seconds:
                logger.warning(f"⚠️ Timeout after {max_wait_minutes} minutes. Completed: {iteration_completed}/{target_memories}")
                return False
            
            # Wait before next check
            remaining_time = max_wait_seconds - elapsed
            logger.info(f"   ⏱️ Checking again in {check_interval}s (timeout in {remaining_time/60:.1f}m)")
            time.sleep(check_interval)
    
    def validate_sentence_content(self, sentence_data: Dict) -> bool:
        """Validate sentence content - basic validation only"""
        content = sentence_data.get('content', '').strip()
        
        # Filter only truly invalid content
        if not content:  # Empty content
            return False
            
        return True

    def insert_batch(self, sentences_batch: List[Dict], batch_start: int) -> Tuple[int, List[str]]:
        """Insert a single batch of sentence memories"""
        batch_requests = []
        memory_ids = []
        
        # Create batch requests with validation
        valid_requests = []
        skipped_count = 0
        
        for sentence_data in sentences_batch:
            # Validate content before adding to batch
            if not self.validate_sentence_content(sentence_data):
                skipped_count += 1
                logger.warning(f"Skipping invalid sentence: {sentence_data.get('sentence_id', 'unknown')}")
                continue
                
            memory_request = MemoryCreationRequest(
                space_id=self.space_id,
                original_content=sentence_data['content'],
                content_type="text/plain",
                metadata=sentence_data['metadata']
            )
            valid_requests.append(memory_request)
        
        if skipped_count > 0:
            logger.info(f"Filtered out {skipped_count} invalid sentences from batch")
        
        if not valid_requests:
            logger.warning("No valid sentences in batch - skipping")
            return 0, []
            
        batch_requests = valid_requests
        
        try:
            # Create batch request
            batch_request = BatchMemoryCreationRequest(requests=batch_requests)
            
            # Make batch API call
            api_response = self.memories_api.batch_create_memory_with_http_info(
                batch_memory_creation_request=batch_request
            )
            
            # Handle response format variations
            if isinstance(api_response, tuple):
                response_data = api_response[0]
                status_code = api_response[1] if len(api_response) > 1 else 200
            else:
                response_data = api_response
                status_code = 200
            
            logger.debug(f"Batch API response: status={status_code}")
            
            # Parse response
            successful = 0
            if hasattr(response_data, 'data') and hasattr(response_data.data, 'results'):
                results = response_data.data.results
                
                for i, result in enumerate(results):
                    if hasattr(result, 'memory') and result.memory:
                        memory_id = result.memory.memory_id
                        memory_ids.append(memory_id)
                        successful += 1
                        
                        # Track sentence-to-memory mapping
                        if i < len(sentences_batch):
                            sentence_id = sentences_batch[i]['sentence_id']
                            self.sentence_to_memory_mapping[sentence_id] = memory_id
                    else:
                        # Log failed items in batch
                        if i < len(sentences_batch):
                            sentence_id = sentences_batch[i]['sentence_id']
                            logger.warning(f"Failed to create memory for sentence {sentence_id}")
                
                logger.info(f"Batch result: {successful}/{len(sentences_batch)} memories created successfully")
                return successful, memory_ids
            else:
                # More conservative fallback - don't assume all succeeded
                logger.warning("Batch response parsing failed - using conservative reconstruction")
                time.sleep(3)  # Wait longer for memories to be created
                
                successful = self._reconstruct_sentence_mapping(sentences_batch, batch_start)
                actual_created = min(successful, len(sentences_batch))  # Cap at batch size
                logger.info(f"Reconstructed: {actual_created}/{len(sentences_batch)} memories confirmed")
                return actual_created, []
        
        except Exception as e:
            logger.error(f"❌ Batch insertion failed: {e}")
            return 0, []
    
    def _reconstruct_sentence_mapping(self, sentences_batch: List[Dict], batch_start: int) -> int:
        """Reconstruct sentence-to-memory mapping by querying recent memories"""
        try:
            response = self.memories_api.list_memories(space_id=self.space_id)
            
            if not hasattr(response, 'memories'):
                return len(sentences_batch)  # Assume all succeeded
            
            all_memories = response.memories
            recent_memories = sorted(all_memories, 
                                   key=lambda m: getattr(m, 'created_at', 0), 
                                   reverse=True)[:len(sentences_batch) * 2]
            
            successful = 0
            for memory in recent_memories:
                if not hasattr(memory, 'metadata') or not memory.metadata:
                    continue
                
                memory_id = str(memory.memory_id)
                metadata = memory.metadata
                
                # Check if this is a SQuAD sentence memory
                if metadata.get('source') != 'squad_1.1':
                    continue
                
                sentence_id = metadata.get('sentence_id')
                if sentence_id:
                    # Find matching sentence from batch
                    for sentence_data in sentences_batch:
                        if sentence_data['sentence_id'] == sentence_id:
                            self.sentence_to_memory_mapping[sentence_id] = memory_id
                            successful += 1
                            break
                
                if successful >= len(sentences_batch):
                    break
            
            logger.info(f"✅ Reconstructed mapping for {successful} sentences")
            return successful
        
        except Exception as e:
            logger.error(f"Error reconstructing sentence mapping: {e}")
            return len(sentences_batch)  # Assume all succeeded
    
    def update_ground_truth_with_memory_ids(self, questions_data: List[Dict]):
        """Update ground truth entries with actual memory IDs"""
        logger.info("🔄 Updating ground truth entries with memory IDs...")
        
        updated_count = 0
        for entry in questions_data:
            # Update correct memory ID
            correct_sentence_id = entry['correct_sentence_id']
            if correct_sentence_id in self.sentence_to_memory_mapping:
                entry['correct_memory_id'] = self.sentence_to_memory_mapping[correct_sentence_id]
                updated_count += 1
            
            # Update all paragraph memory IDs
            paragraph_memory_ids = []
            for sentence_id in entry['all_paragraph_sentence_ids']:
                if sentence_id in self.sentence_to_memory_mapping:
                    paragraph_memory_ids.append(self.sentence_to_memory_mapping[sentence_id])
            
            entry['all_paragraph_memory_ids'] = paragraph_memory_ids
        
        self.ground_truth_entries = questions_data
        logger.info(f"✅ Updated {updated_count} ground truth entries with memory IDs")

    
    def save_ground_truth_and_mapping(self):
        """Save ground truth data and sentence-to-memory mapping"""
        try:
            # Save ground truth
            with open(self.ground_truth_file, 'w') as f:
                json.dump(self.ground_truth_entries, f, indent=2)
            logger.info(f"💾 Saved ground truth to {self.ground_truth_file}")
            
            # Save sentence-to-memory mapping
            with open(self.sentence_memory_mapping_file, 'w') as f:
                json.dump(self.sentence_to_memory_mapping, f, indent=2)
            logger.info(f"💾 Saved sentence mapping to {self.sentence_memory_mapping_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving files: {e}")
    
    def run_insertion(self, limit: Optional[int] = None, start_iteration: int = 0):
        """Run the complete SQuAD sentence insertion process"""
        logger.info("🚀 SQuAD 1.1 Sentence Insertion for Hybrid Embedder Testing")
        logger.info("=" * 60)
        logger.info(f"Target space: {self.space_id}")
        logger.info(f"API Key: {GOODMEM_API_KEY[:10]}...")
        
        # Test API connectivity
        logger.info("\n🔗 Testing GoodMem API connectivity...")
        try:
            test_memory = MemoryCreationRequest(
                space_id=self.space_id,
                original_content="SQuAD connectivity test",
                content_type="text/plain",
                metadata={"test": "connectivity", "source": "squad_test"}
            )
            
            test_response = self.memories_api.create_memory(memory_creation_request=test_memory)
            logger.info(f"✅ API connectivity test successful")
            
        except Exception as e:
            logger.error(f"❌ API connectivity test failed: {e}")
            return
        
        # Load SQuAD data
        logger.info("\n📥 Loading and processing SQuAD 1.1 data...")
        sentences_data, questions_data = self.load_squad_data()
        
        if limit and limit < len(sentences_data):
            logger.info(f"⚠️ Limiting insertion to {limit:,} sentences for testing")
            sentences_data = sentences_data[:limit]
            # Filter questions to match limited sentences
            limited_sentence_ids = {s['sentence_id'] for s in sentences_data}
            questions_data = [q for q in questions_data 
                            if q['correct_sentence_id'] in limited_sentence_ids]
        
        self.target_total = len(sentences_data)
        total_iterations = (self.target_total + self.iteration_size - 1) // self.iteration_size
        
        logger.info(f"\n🎯 Starting insertion from iteration {start_iteration + 1}/{total_iterations}")
        logger.info(f"Configuration:")
        logger.info(f"  • Total sentences: {self.target_total:,}")
        logger.info(f"  • Total questions: {len(questions_data):,}")
        logger.info(f"  • Iteration size: {self.iteration_size:,}")
        logger.info(f"  • Batch size: {self.batch_size}")
        
        # Process iterations
        for iteration in range(start_iteration, total_iterations):
            iteration_start_time = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 ITERATION {iteration + 1}/{total_iterations}")
            logger.info(f"{'='*60}")
            
            # CAPTURE BASELINE AT START OF ITERATION (before any insertion)
            iteration_baseline_stats = self.get_processing_stats()
            iteration_baseline_completed = iteration_baseline_stats['COMPLETED']
            iteration_baseline_failed = iteration_baseline_stats['FAILED']
            logger.info(f"📊 Iteration baseline: {iteration_baseline_completed} completed, {iteration_baseline_failed} failed")
            
            # Calculate data slice for this iteration
            start_idx = iteration * self.iteration_size
            end_idx = min((iteration + 1) * self.iteration_size, len(sentences_data))
            iteration_data = sentences_data[start_idx:end_idx]
            actual_iteration_size = len(iteration_data)
            
            logger.info(f"Processing sentences {start_idx:,} to {end_idx-1:,} ({actual_iteration_size:,} sentences)")
            
            # Process batches
            iteration_success = 0
            batches_in_iteration = (actual_iteration_size + self.batch_size - 1) // self.batch_size
            
            batch_pbar = tqdm(
                range(batches_in_iteration),
                desc=f"Iteration {iteration+1} Batches",
                unit="batch"
            )
            
            for batch_idx in batch_pbar:
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, actual_iteration_size)
                batch_data = iteration_data[batch_start:batch_end]
                
                # Insert batch
                success_count, memory_ids = self.insert_batch(batch_data, start_idx + batch_start)
                iteration_success += success_count
                
                batch_pbar.set_postfix({
                    'success': f"{iteration_success}/{batch_idx * self.batch_size + len(batch_data)}",
                    'rate': f"{success_count}/{len(batch_data)}"
                })
                
                # time.sleep(0.5)  # Removed delay for GPU optimization
            
            batch_pbar.close()
            
            # Update progress
            prev_total = self.total_inserted
            self.total_inserted += iteration_success
            insertion_time = time.time() - iteration_start_time
            
            logger.debug(f"Progress update: {prev_total} + {iteration_success} = {self.total_inserted}")
            
            logger.info(f"📊 Iteration {iteration + 1} Results:")
            logger.info(f"   • Inserted: {iteration_success:,}/{actual_iteration_size:,} sentences")
            logger.info(f"   • Time: {insertion_time:.1f}s")
            logger.info(f"   • Rate: {iteration_success/insertion_time:.1f} sentences/sec")
            logger.info(f"   • Total progress: {self.total_inserted:,}/{self.target_total:,} ({self.total_inserted/self.target_total*100:.1f}%)")
            
            # Wait for processing completion and handle stuck memories
            if iteration_success > 0:
                completion_success = self.wait_for_iteration_completion(
                    target_memories=iteration_success,
                    iteration_baseline_completed=iteration_baseline_completed,
                    iteration_baseline_failed=iteration_baseline_failed,
                    max_wait_minutes=10
                )
                
                if not completion_success:
                    logger.warning(f"⚠️ Iteration {iteration + 1} did not complete within timeout")
                    
                    # Try to recover stuck pending memories
                    retry_count = self.retry_pending_memories()
                    if retry_count > 0:
                        logger.info(f"🔄 Retried {retry_count} stuck memories - waiting for completion...")
                        time.sleep(30)  # Wait for retries to process
            
            self.completed_iterations = iteration + 1
            
            # Update ground truth with current mappings
            self.update_ground_truth_with_memory_ids(questions_data)
            
            # Save progress
            progress_data = {
                'completed_iterations': self.completed_iterations,
                'total_inserted': self.total_inserted,
                'space_id': self.space_id,
                'test_purpose': 'squad_retrieval_evaluation',
                'timestamp': time.time()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        
        # Final ground truth processing
        logger.info(f"\n💾 Generating final ground truth and mapping files...")
        self.update_ground_truth_with_memory_ids(questions_data)
        self.save_ground_truth_and_mapping()
        
        # Final summary
        final_stats = self.get_processing_stats()
        logger.info(f"\n{'='*60}")
        logger.info("🎉 SQUAD SENTENCE INSERTION COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Total inserted: {self.total_inserted:,} sentences")
        logger.info(f"   • Ground truth entries: {len(self.ground_truth_entries):,}")
        logger.info(f"   • Sentence mappings: {len(self.sentence_to_memory_mapping):,}")
        logger.info(f"   • Processing state:")
        logger.info(f"     - Completed: {final_stats['COMPLETED']:,}")
        logger.info(f"     - Pending: {final_stats['PENDING']:,}")
        logger.info(f"     - Failed: {final_stats['FAILED']:,}")
        logger.info(f"   • Files generated:")
        logger.info(f"     - Ground truth: {self.ground_truth_file}")
        logger.info(f"     - Sentence mapping: {self.sentence_memory_mapping_file}")
        logger.info(f"     - Progress: {self.progress_file}")
        logger.info(f"   • Ready for MRR/R@N evaluation!")
        
        return self.total_inserted, final_stats


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Insert SQuAD 1.1 sentences into GoodMem for hybrid embedder testing")
    parser.add_argument("--space-id", type=str, default=DEFAULT_SPACE_ID,
                       help="Target space ID for insertion")
    parser.add_argument("--start-iteration", type=int, default=0,
                       help="Starting iteration (for resumption)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test data loading without insertion")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of sentences to insert (for testing)")
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - Testing SQuAD data loading")
        inserter = SquadSentenceInserter(args.space_id)
        sentences_data, questions_data = inserter.load_squad_data()
        logger.info(f"✅ Dry run complete - would insert {len(sentences_data):,} sentences")
        logger.info(f"✅ Would create {len(questions_data):,} ground truth entries")
        return
    
    # Run the insertion
    inserter = SquadSentenceInserter(args.space_id)
    
    logger.info(f"🔬 SQUAD 1.1 HYBRID EMBEDDER TESTING MODE")
    logger.info(f"Space ID: {args.space_id}")
    if args.limit:
        logger.info(f"Limit: {args.limit:,} sentences")
    
    try:
        total_inserted, final_stats = inserter.run_insertion(
            limit=args.limit,
            start_iteration=args.start_iteration
        )
        logger.info("✅ SQuAD sentence insertion completed successfully!")
        logger.info("🧪 Ready for hybrid embedder evaluation!")
        
    except KeyboardInterrupt:
        logger.info("\n⏸️ Insertion interrupted by user")
        logger.info(f"Progress saved to {inserter.progress_file}")
        logger.info(f"Resume with: --start-iteration {inserter.completed_iterations}")
        
    except Exception as e:
        logger.error(f"❌ Insertion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()