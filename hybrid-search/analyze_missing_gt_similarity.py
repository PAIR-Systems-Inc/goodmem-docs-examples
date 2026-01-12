#!/usr/bin/env python3
"""
Analyze Missing Ground Truth Similarity Scores

This script analyzes why ground truth memories are missing from the top-10 results
by computing similarity scores between queries and both retrieved results and ground truth.

It compares:
1. Query vs Top-10 retrieved memories (with similarity scores)
2. Query vs Ground truth memory (similarity score)
3. Analyzes the similarity score difference

Usage:
    python analyze_missing_gt_similarity.py --space-id SPACE-ID --missing-file path/to/missingTerms.json --limit 10
"""

import json
import logging
import sys
import time
import argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Optional imports for database access
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not available - SQL similarity computation will not work")

# Add path for local GoodMem client from official repo
# sys.path.insert(0, '/path/to/goodmem/clients/python')  # Uncomment and update path if needed

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


class SimilarityAnalyzer:
    def __init__(self, space_id: str):
        self.space_id = space_id

        # Create API clients
        self.configuration = Configuration()
        self.configuration.host = GOODMEM_SERVER_URL

        self.api_client = ApiClient(self.configuration)
        self.api_client.configuration.api_key = {"ApiKeyAuth": GOODMEM_API_KEY}

        self.memories_api = MemoriesApi(self.api_client)
        self.spaces_api = SpacesApi(self.api_client)
        self.stream_client = MemoryStreamClient(self.api_client)

        # Fetch embedders and weights from space configuration
        self.embedders_config = self._fetch_space_embedders()

    def _fetch_space_embedders(self) -> List[Dict]:
        """Fetch embedders and their weights from space configuration"""
        try:
            space_response = self.spaces_api.get_space(id=self.space_id)
            embedders_config = []

            # Extract embedders and weights from space configuration
            space_embedders = None
            if hasattr(space_response, 'space_embedders') and space_response.space_embedders:
                space_embedders = space_response.space_embedders
            elif hasattr(space_response, 'embedder_weights') and space_response.embedder_weights:
                space_embedders = space_response.embedder_weights

            if space_embedders:
                for embedder_item in space_embedders:
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

            logger.info(f"📋 Found {len(embedders_config)} embedders for space {self.space_id}")
            for config in embedders_config:
                logger.info(f"   • Embedder {config['embedder_id']}: weight={config['weight']}")

            return embedders_config

        except Exception as e:
            logger.error(f"❌ Failed to fetch space embedders: {e}")
            return []

    def get_query_embedding(self, query_text: str) -> Optional[np.ndarray]:
        """Get embedding vector for a query using the embedder endpoint"""
        try:
            import requests

            # Call the Qwen embedder container directly
            embedder_url = "http://localhost:9006/embed"  # Qwen embedder on port 9006

            response = requests.post(
                embedder_url,
                json={"inputs": [query_text]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # Response is a list of embeddings directly
                if isinstance(result, list) and result:
                    # Convert to numpy array
                    embedding = np.array(result[0])
                    return embedding
                else:
                    logger.error("No embeddings returned from embedder")
                    return None
            else:
                logger.error(f"Embedder request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return None

    def get_top_k_with_scores(self, query_text: str, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Retrieve top-k memories with similarity scores"""
        try:
            # Create space key with weights
            embedder_weights = [
                EmbedderWeight(embedder_id=config['embedder_id'], weight=config['weight'])
                for config in self.embedders_config
            ]

            space_key = SpaceKey(
                space_id=self.space_id,
                embedder_weights=embedder_weights
            )

            results = []
            memory_definitions = {}

            for event in self.stream_client.retrieve_memory_stream(
                message=query_text,
                space_keys=[space_key],
                requested_size=k
            ):
                # Collect memory definitions
                if hasattr(event, 'memory_definition') and event.memory_definition:
                    memory = event.memory_definition
                    if isinstance(memory, dict):
                        memory_id = memory.get('memoryId', memory.get('memory_id', memory.get('id', 'unknown')))
                    else:
                        memory_id = getattr(memory, 'memory_id',
                                   getattr(memory, 'memoryId',
                                   getattr(memory, 'id', 'unknown')))
                    memory_definitions[len(memory_definitions)] = {
                        'memory_id': str(memory_id),
                        'memory_data': memory
                    }

                # Collect retrieved items with scores
                if hasattr(event, 'retrieved_item') and event.retrieved_item:
                    retrieved_item = event.retrieved_item

                    # Debug: Check what attributes are available
                    if len(results) == 0:  # Only log for first item to avoid spam
                        logger.debug(f"Retrieved item attributes: {dir(retrieved_item)}")
                        if hasattr(retrieved_item, '__dict__'):
                            logger.debug(f"Retrieved item dict: {retrieved_item.__dict__}")

                    # Get similarity score from chunk.relevance_score
                    similarity_score = None
                    if hasattr(retrieved_item, 'chunk') and retrieved_item.chunk:
                        chunk = retrieved_item.chunk
                        if hasattr(chunk, 'relevance_score') and chunk.relevance_score is not None:
                            # Negate the score since GoodMem returns negative inner product
                            similarity_score = -chunk.relevance_score
                            logger.debug(f"Found relevance_score: {chunk.relevance_score}, converted to: {similarity_score}")

                    if similarity_score is None:
                        similarity_score = 0.0  # Fallback

                    # Get memory index
                    memory_index = getattr(retrieved_item, 'memory_index', None)
                    if memory_index is None and hasattr(retrieved_item, 'chunk'):
                        chunk_ref = retrieved_item.chunk
                        memory_index = getattr(chunk_ref, 'memory_index', None)

                    if memory_index is not None and memory_index in memory_definitions:
                        memory_info = memory_definitions[memory_index]
                        memory_id = memory_info['memory_id']
                        memory_data = memory_info['memory_data']

                        # Only add if not already present
                        if not any(result[0] == memory_id for result in results):
                            results.append((memory_id, similarity_score or 0.0, memory_data))
                    elif hasattr(retrieved_item, 'memory_id'):
                        memory_id = retrieved_item.memory_id
                        if not any(result[0] == memory_id for result in results):
                            results.append((memory_id, similarity_score or 0.0, {}))

            return results[:k]

        except Exception as e:
            logger.error(f"Failed to get top-k results: {e}")
            return []

    def get_memory_similarity_via_sql(self, query_embedding: np.ndarray, memory_id: str, embedder_id: str) -> Optional[float]:
        """Get similarity between query and memory via direct SQL computation"""
        if not PSYCOPG2_AVAILABLE:
            logger.error("psycopg2 not available for SQL operations")
            return None

        try:
            # Database connection - GoodMem database settings from Docker container
            conn = psycopg2.connect(
                host="localhost",
                port="5432",
                database="goodmem_db",
                user="goodmem_admin",
                password="ayk317gk"
            )

            # SQL query to get embedding and compute similarity
            # Join memory_chunk and dense_chunk_pointer to get embeddings for a memory
            query = """
            SELECT dcp.embedding_vector <#> %s::vector as similarity_score
            FROM goodmem.memory_chunk mc
            JOIN goodmem.dense_chunk_pointer dcp ON mc.chunk_id = dcp.chunk_id
            WHERE mc.memory_id = %s AND dcp.embedder_id = %s
            LIMIT 1
            """

            # Convert numpy array to list and pad to 1536 dimensions (same as database default)
            query_embedding_list = query_embedding.tolist()

            # Pad with zeros if needed (Qwen is 1024-dim, database expects 1536-dim)
            target_dim = 1536
            if len(query_embedding_list) < target_dim:
                query_embedding_list.extend([0.0] * (target_dim - len(query_embedding_list)))
                logger.debug(f"Padded query embedding from {len(query_embedding)} to {target_dim} dimensions")

            with conn.cursor() as cursor:
                cursor.execute(query, (json.dumps(query_embedding_list), memory_id, embedder_id))
                result = cursor.fetchone()

                if result:
                    # PostgreSQL vector extension returns negative inner product for <#> operator
                    # so we negate it to get the actual inner product
                    similarity_score = -result[0]
                    return float(similarity_score)
                else:
                    logger.error(f"No embedding found for memory {memory_id} with embedder {embedder_id}")
                    return None

        except Exception as e:
            logger.error(f"Failed to get memory similarity via SQL: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()

    def get_memory_content(self, memory_id: str) -> Optional[str]:
        """Get the text content of a memory"""
        try:
            memory_response = self.memories_api.get_memory(id=memory_id, include_content=True)
            if not memory_response:
                return None

            # Get memory text - should now have original_content due to include_content=True
            memory_text = None
            possible_attrs = ['original_content', 'text', 'content', 'data', 'body', 'message']

            for attr in possible_attrs:
                if hasattr(memory_response, attr):
                    memory_text = getattr(memory_response, attr)
                    if memory_text:
                        # Check if content is base64 encoded and decode it
                        if attr == 'original_content':
                            try:
                                import base64
                                decoded_content = base64.b64decode(memory_text).decode('utf-8')
                                memory_text = decoded_content
                            except Exception:
                                # If decoding fails, use as-is
                                pass
                        break

            if not memory_text:
                if hasattr(memory_response, '__dict__'):
                    memory_dict = memory_response.__dict__
                    logger.debug(f"Memory {memory_id} attributes: {list(memory_dict.keys())}")
                    if 'original_content' in memory_dict and memory_dict['original_content']:
                        memory_text = memory_dict['original_content']
                        logger.debug(f"Found content in original_content for {memory_id}")
                    else:
                        # Try other content keys
                        for key in memory_dict.keys():
                            if 'content' in key.lower() and key != 'content_type' and memory_dict[key]:
                                memory_text = memory_dict[key]
                                logger.debug(f"Found content in {key} for {memory_id}")
                                break

            return memory_text

        except Exception as e:
            logger.error(f"Failed to get memory content for {memory_id}: {e}")
            return None

    def compute_similarity_score(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute inner product similarity between two vectors"""
        try:
            return float(np.dot(vec1, vec2))
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    def analyze_missing_question(self, missing_item: Dict, verbose: bool = False) -> Dict:
        """Analyze a single missing question"""
        question_text = missing_item['question_text']
        gt_memory_id = missing_item['correct_memory_id']
        question_id = missing_item['question_id']

        logger.info(f"🔍 Analyzing question: {question_id}")
        logger.info(f"   Query: {question_text[:100]}...")

        analysis_result = {
            'question_id': question_id,
            'question_text': question_text,
            'gt_memory_id': gt_memory_id,
            'analysis_success': False,
            'error': None
        }

        try:
            # 1. Get query embedding
            logger.info("   📊 Getting query embedding...")
            query_embedding = self.get_query_embedding(question_text)
            if query_embedding is None:
                analysis_result['error'] = 'Failed to get query embedding'
                return analysis_result

            # 2. Get top-10 results with scores
            logger.info("   🔍 Getting top-10 retrieval results...")
            top_k_results = self.get_top_k_with_scores(question_text, k=10)
            if not top_k_results:
                analysis_result['error'] = 'Failed to get top-k results'
                return analysis_result

            # 3. Get ground truth similarity via SQL
            logger.info("   🎯 Computing ground truth similarity via SQL...")
            if not self.embedders_config:
                analysis_result['error'] = 'No embedders configured'
                return analysis_result

            embedder_id = self.embedders_config[0]['embedder_id']
            gt_similarity = self.get_memory_similarity_via_sql(query_embedding, gt_memory_id, embedder_id)
            if gt_similarity is None:
                analysis_result['error'] = 'Failed to compute ground truth similarity'
                return analysis_result

            # 4. Process top-k results - use retrieved scores directly and compute exact similarities via SQL
            top_k_analysis = []
            for rank, (memory_id, retrieved_score, memory_data) in enumerate(top_k_results, 1):
                # Compute actual similarity via SQL for exact comparison
                actual_similarity = self.get_memory_similarity_via_sql(query_embedding, memory_id, embedder_id)

                top_k_analysis.append({
                    'rank': rank,
                    'memory_id': memory_id,
                    'retrieved_score': retrieved_score,  # Score from retrieval API
                    'actual_similarity': actual_similarity,  # Exact similarity via SQL
                    'similarity_diff_vs_gt': (actual_similarity - gt_similarity) if actual_similarity is not None else None
                })

            # 5. Analysis results
            analysis_result.update({
                'analysis_success': True,
                'gt_similarity_score': gt_similarity,
                'top_k_results': top_k_analysis,
                'gt_rank_if_inserted': self._compute_gt_rank(gt_similarity, top_k_analysis),
                'min_top_k_similarity': min([r['actual_similarity'] for r in top_k_analysis if r['actual_similarity'] is not None], default=None),
                'max_top_k_similarity': max([r['actual_similarity'] for r in top_k_analysis if r['actual_similarity'] is not None], default=None)
            })

            logger.info(f"   ✅ GT similarity: {gt_similarity:.6f}")
            top1_sim = top_k_analysis[0]['actual_similarity'] if top_k_analysis[0]['actual_similarity'] is not None else None
            top1_sim_str = f"{top1_sim:.6f}" if top1_sim is not None else "N/A"
            logger.info(f"   📊 Top-1 similarity: {top1_sim_str}")

            # Verbose output
            if verbose:
                self._display_verbose_analysis(question_text, top_k_analysis, gt_memory_id, gt_similarity)

        except Exception as e:
            analysis_result['error'] = str(e)
            logger.error(f"   ❌ Analysis failed: {e}")

        return analysis_result

    def _display_verbose_analysis(self, query_text: str, top_k_analysis: List[Dict], gt_memory_id: str, gt_similarity: float):
        """Display detailed verbose analysis"""
        print(f"\n{'='*80}")
        print(f"🔍 QUERY:")
        print(f"{'='*80}")
        print(f"{query_text}")

        print(f"\n{'='*80}")
        print(f"🏆 TOP-10 RETRIEVAL RESULTS:")
        print(f"{'='*80}")

        for result in top_k_analysis:
            rank = result['rank']
            memory_id = result['memory_id']
            retrieved_score = result['retrieved_score']
            actual_similarity = result['actual_similarity']

            # Get memory content
            memory_content = self.get_memory_content(memory_id)
            content_preview = (memory_content[:200] + "...") if memory_content and len(memory_content) > 200 else (memory_content or "Content not available")

            actual_sim_str = f"{actual_similarity:.6f}" if actual_similarity is not None else "N/A"
            retrieved_score_str = f"{retrieved_score:.6f}" if retrieved_score is not None else "N/A"

            print(f"\n📍 RANK {rank}:")
            print(f"   Memory ID: {memory_id}")
            print(f"   Retrieved Score: {retrieved_score_str}")
            print(f"   Actual Similarity: {actual_sim_str}")
            print(f"   Content: {content_preview}")

        print(f"\n{'='*80}")
        print(f"🎯 GROUND TRUTH:")
        print(f"{'='*80}")

        gt_content = self.get_memory_content(gt_memory_id)
        gt_content_preview = (gt_content[:200] + "...") if gt_content and len(gt_content) > 200 else (gt_content or "Content not available")

        print(f"   Memory ID: {gt_memory_id}")
        print(f"   Similarity Score: {gt_similarity:.6f}")
        print(f"   Content: {gt_content_preview}")

        # Compute rank if GT were inserted
        gt_rank = self._compute_gt_rank(gt_similarity, top_k_analysis)
        print(f"   Rank if inserted: {gt_rank}")

        print(f"\n{'='*80}")

    def _compute_gt_rank(self, gt_similarity: float, top_k_analysis: List[Dict]) -> int:
        """Compute what rank the GT would have if inserted into top-k results"""
        valid_similarities = [r['actual_similarity'] for r in top_k_analysis if r['actual_similarity'] is not None]
        valid_similarities.append(gt_similarity)
        valid_similarities.sort(reverse=True)
        return valid_similarities.index(gt_similarity) + 1

    def analyze_missing_terms(self, missing_file: str, limit: Optional[int] = None, verbose: bool = False) -> Dict:
        """Analyze all missing terms from the file"""
        logger.info(f"🚀 MISSING GROUND TRUTH SIMILARITY ANALYSIS")
        logger.info("=" * 60)

        # Load missing terms
        try:
            with open(missing_file, 'r') as f:
                missing_terms = json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load missing terms file: {e}")
            return {}

        if limit and limit < len(missing_terms):
            missing_terms = missing_terms[:limit]
            logger.info(f"⚠️  Limited to {limit} questions for analysis")

        logger.info(f"📊 Analyzing {len(missing_terms)} missing questions")

        # Analyze each missing question
        analysis_results = []
        successful_analyses = []

        for i, missing_item in enumerate(tqdm(missing_terms, desc="Analyzing questions")):
            logger.info(f"\n--- Question {i+1}/{len(missing_terms)} ---")

            result = self.analyze_missing_question(missing_item, verbose)
            analysis_results.append(result)

            if result['analysis_success']:
                successful_analyses.append(result)

            # Brief pause to avoid overwhelming the API
            time.sleep(0.1)

        # Compute summary statistics
        if successful_analyses:
            gt_similarities = [r['gt_similarity_score'] for r in successful_analyses]
            top1_similarities = [r['top_k_results'][0]['actual_similarity']
                               for r in successful_analyses
                               if r['top_k_results'] and r['top_k_results'][0]['actual_similarity'] is not None]

            similarity_diffs = [r['top_k_results'][0]['actual_similarity'] - r['gt_similarity_score']
                              for r in successful_analyses
                              if r['top_k_results'] and r['top_k_results'][0]['actual_similarity'] is not None]

            summary_stats = {
                'total_analyzed': len(analysis_results),
                'successful_analyses': len(successful_analyses),
                'gt_similarity_stats': {
                    'mean': np.mean(gt_similarities),
                    'std': np.std(gt_similarities),
                    'min': np.min(gt_similarities),
                    'max': np.max(gt_similarities)
                },
                'top1_similarity_stats': {
                    'mean': np.mean(top1_similarities) if top1_similarities else None,
                    'std': np.std(top1_similarities) if top1_similarities else None,
                    'min': np.min(top1_similarities) if top1_similarities else None,
                    'max': np.max(top1_similarities) if top1_similarities else None
                },
                'similarity_diff_stats': {
                    'mean': np.mean(similarity_diffs) if similarity_diffs else None,
                    'std': np.std(similarity_diffs) if similarity_diffs else None,
                    'min': np.min(similarity_diffs) if similarity_diffs else None,
                    'max': np.max(similarity_diffs) if similarity_diffs else None
                }
            }
        else:
            summary_stats = {
                'total_analyzed': len(analysis_results),
                'successful_analyses': 0
            }

        final_results = {
            'analysis_timestamp': time.time(),
            'space_id': self.space_id,
            'embedders_config': self.embedders_config,
            'summary_statistics': summary_stats,
            'individual_analyses': analysis_results
        }

        # Display results
        logger.info(f"\n{'='*60}")
        logger.info("📊 SIMILARITY ANALYSIS RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"📋 Questions analyzed: {len(analysis_results)}")
        logger.info(f"✅ Successful analyses: {len(successful_analyses)}")

        if successful_analyses:
            logger.info(f"\n🎯 Ground Truth Similarity Statistics:")
            logger.info(f"   • Mean: {summary_stats['gt_similarity_stats']['mean']:.6f}")
            logger.info(f"   • Std:  {summary_stats['gt_similarity_stats']['std']:.6f}")
            logger.info(f"   • Range: [{summary_stats['gt_similarity_stats']['min']:.6f}, {summary_stats['gt_similarity_stats']['max']:.6f}]")

            if summary_stats['top1_similarity_stats']['mean'] is not None:
                logger.info(f"\n🥇 Top-1 Similarity Statistics:")
                logger.info(f"   • Mean: {summary_stats['top1_similarity_stats']['mean']:.6f}")
                logger.info(f"   • Std:  {summary_stats['top1_similarity_stats']['std']:.6f}")
                logger.info(f"   • Range: [{summary_stats['top1_similarity_stats']['min']:.6f}, {summary_stats['top1_similarity_stats']['max']:.6f}]")

                logger.info(f"\n📊 Similarity Difference (Top-1 - GT) Statistics:")
                logger.info(f"   • Mean: {summary_stats['similarity_diff_stats']['mean']:.6f}")
                logger.info(f"   • Std:  {summary_stats['similarity_diff_stats']['std']:.6f}")
                logger.info(f"   • Range: [{summary_stats['similarity_diff_stats']['min']:.6f}, {summary_stats['similarity_diff_stats']['max']:.6f}]")

        return final_results


def main():
    parser = argparse.ArgumentParser(description="Analyze missing ground truth similarity scores")
    parser.add_argument("--space-id", type=str, required=True, help="GoodMem space ID")
    parser.add_argument("--missing-file", type=str, required=True, help="Path to missing terms JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to analyze")
    parser.add_argument("--output", type=str, default=None, help="Output file path (optional)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed query, top-10 results content, and ground truth content")

    args = parser.parse_args()

    logger.info(f"🔬 SIMILARITY ANALYSIS")
    logger.info(f"Space ID: {args.space_id}")
    logger.info(f"Missing file: {args.missing_file}")

    analyzer = SimilarityAnalyzer(args.space_id)

    try:
        results = analyzer.analyze_missing_terms(
            missing_file=args.missing_file,
            limit=args.limit,
            verbose=args.verbose
        )

        if results:
            # Save results
            if args.output:
                output_file = args.output
            else:
                output_file = f"similarity_analysis_{args.space_id}_{int(time.time())}.json"

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"💾 Results saved to: {output_file}")
            logger.info("🎉 Analysis completed!")
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏸️ Analysis interrupted")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()