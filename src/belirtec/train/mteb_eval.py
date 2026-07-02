#!/usr/bin/env python3
"""
Legal Evaluation Script

This single-file script combines the evaluation core functionality:
- Legal (Turkish Legal Embedding Benchmark) task registration
- Model2Vec adapter for static embedding models
- Model loader with automatic max_seq_length optimization
- MTEB evaluator with support for Turkish and legal domain tasks

Usage:
    python legal_evaluation.py --model "model_name" [options]

Examples:
    # Evaluate a sentence-transformers model
    python legal_evaluation.py --model "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    
    # Evaluate a model2vec model
    python legal_evaluation.py --model "minishlab/potion-base-8M"
    
    # Specify output directory and batch size
    python legal_evaluation.py --model "model_name" --output-dir "./results" --batch-size 64
"""

import os
import sys
import time
import json
import math
import logging
import argparse
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import mteb
from mteb.abstasks import TaskMetadata
from mteb.overview import TASKS_REGISTRY

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

TASK_CATEGORIES = {
    "Classification": [
        "MassiveIntentClassification", 
        "MassiveScenarioClassification", 
        "MultilingualSentimentClassification", 
        "SIB200Classification", 
        "TurkishMovieSentimentClassification", 
        "TurkishProductSentimentClassification"
    ],
    "Clustering": ["SIB200ClusteringS2S"],
    "PairClassification": ["XNLI", "XNLIV2"],
    "Retrieval": [
        "TurHistQuadRetrieval", 
        "WebFAQRetrieval", 
        "XQuADRetrieval",
        "MKQARetrieval"
    ],
    "STS": ["STS22.v2"],
    # Turkish Legal Task Categories
    "Contracts": ["TurkishLegalQA"],
    "Regulation": ["TurkishTaxRulings"],
    "Caselaw": ["TurkishCourtOfCassation"],
}

LEGAL_CATEGORIES = ["Contracts", "Regulation", "Caselaw"]

DEFAULT_SETTINGS = {
    "batch_size": 32,
    "output_dir": "results",
}

MAX_SEQUENCE_LENGTH_LIMIT = 2048


# =============================================================================
# Utility Functions
# =============================================================================

def convert_model_name_to_safe_filename(model_name: str) -> str:
    """Convert a model name to a safe filename format."""
    return model_name.replace("/", "__", 1)


def convert_safe_filename_to_model_name(safe_name: str) -> str:
    """Convert a safe filename back to original model name format."""
    if "__" in safe_name:
        return safe_name.replace("__", "/", 1)
    return safe_name


def format_parameter_count(n_parameters: Optional[Union[int, float]]) -> str:
    """Format parameter count in human-readable format (K, M, B)."""
    if n_parameters is None or not n_parameters:
        return "Unknown"
    
    n_thousand = int(n_parameters // 1e3)
    if n_thousand < 1:
        return str(int(n_parameters))
    
    n_zeros = math.log10(n_thousand) if n_thousand > 0 else 0
    if n_zeros >= 6:
        return str(n_thousand // (10**6)) + "B"
    if n_zeros >= 3:
        return str(n_thousand // (10**3)) + "M"
    return str(n_thousand) + "K"


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler()]
    )


# =============================================================================
# Legal Task Registration
# =============================================================================

class LegalTaskConfig:
    """Configuration for a single Legal task."""
    
    def __init__(self, name: str, dataset_id: str, revision: str, category: str, 
                 description: str, main_score: str = "ndcg_at_10"):
        self.name = name
        self.dataset_id = dataset_id
        self.revision = revision
        self.category = category
        self.description = description
        self.main_score = main_score


# Turkish Legal Task Definitions
LEGAL_TASKS = [
    LegalTaskConfig(
        name="TurkishLegalQA",
        dataset_id="newmindai/contract-retrieval",
        revision="main",
        category="Contracts",
        description="Turkish legal question answering retrieval task"
    ),
    LegalTaskConfig(
        name="TurkishTaxRulings",
        dataset_id="newmindai/regulation-retrieval",
        revision="main",
        category="Regulation",
        description="Turkish legal tax rulings retrieval task"
    ),
    LegalTaskConfig(
        name="TurkishCourtOfCassation",
        dataset_id="newmindai/caselaw-retrieval",
        revision="main",
        category="Caselaw",
        description="Turkish Court of Cassation case law retrieval task"
    ),
]


def register_legal_task(task_config: LegalTaskConfig) -> None:
    """Register a single Legal task with MTEB's task registry."""
    if task_config.name in TASKS_REGISTRY:
        logger.debug(f"Legal task '{task_config.name}' already registered")
        return
    
    from datasets import load_dataset
    from mteb.abstasks.AbsTaskRetrieval import HFDataLoader
    
    class OfflineHFDataLoader(HFDataLoader):
        """HFDataLoader with offline mode fix for multi-config datasets."""
        
        def _load_qrels(self, split):
            """Override to handle both 'default' and 'qrels' config names."""
            if self.hf_repo:
                try:
                    qrels_ds = load_dataset(
                        self.hf_repo_qrels,
                        name="default",
                        keep_in_memory=self.keep_in_memory,
                        streaming=self.streaming,
                        trust_remote_code=self.trust_remote_code,
                    )[split]
                except Exception as e:
                    error_msg = str(e)
                    if "'default'" in error_msg and ("not found" in error_msg or "Couldn't find" in error_msg):
                        qrels_ds = load_dataset(
                            self.hf_repo_qrels,
                            name="qrels",
                            keep_in_memory=self.keep_in_memory,
                            streaming=self.streaming,
                            trust_remote_code=self.trust_remote_code,
                        )[split]
                    else:
                        raise
            else:
                qrels_ds = load_dataset(
                    "csv",
                    data_files=self.qrels_file,
                    delimiter="\t",
                    keep_in_memory=self.keep_in_memory,
                )
            
            from datasets import Features, Value
            features = Features({
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            })
            qrels_ds = qrels_ds.cast(features)
            self.qrels = qrels_ds
    
    class LegalRetrievalTask(mteb.AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=task_config.name,
            dataset={
                "path": task_config.dataset_id,
                "revision": task_config.revision,
            },
            main_score=task_config.main_score,
            description=task_config.description,
            type="Retrieval",
            category="s2p",
            modalities=["text"],
            eval_splits=["train"],
            eval_langs=["tur-Latn"],
            date=("2025-11-01", "2026-01-15"),
            domains=["Legal"],
            task_subtypes=["Question answering"],
            license="cc-by-4.0",
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="found",
        )
        
        def load_data(self, **kwargs):
            """Use custom HFDataLoader with offline mode fix."""
            if self.data_loaded:
                return
            
            self.corpus, self.queries, self.relevant_docs = {}, {}, {}
            dataset_path = self.metadata_dict["dataset"]["path"]
            hf_repo_qrels = (
                dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
            )
            
            for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
                corpus, queries, qrels = OfflineHFDataLoader(
                    hf_repo=dataset_path,
                    hf_repo_qrels=hf_repo_qrels,
                    streaming=False,
                    keep_in_memory=False,
                    trust_remote_code=self.metadata_dict["dataset"].get("trust_remote_code", False),
                ).load(split=split)
                
                queries = {query["id"]: query["text"] for query in queries}
                corpus = {doc["id"]: doc.get("title", "") + " " + doc["text"] for doc in corpus}
                
                self.corpus[split] = corpus
                self.queries[split] = queries
                self.relevant_docs[split] = qrels
            
            self.data_loaded = True
    
    TASKS_REGISTRY[task_config.name] = LegalRetrievalTask
    logger.info(f"Registered Legal task: {task_config.name} ({task_config.category})")


def register_all_legal_tasks() -> None:
    """Register all Legal tasks with MTEB's task registry."""
    logger.info("Registering Legal tasks...")
    for task_config in LEGAL_TASKS:
        register_legal_task(task_config)
    logger.info(f"Successfully registered {len(LEGAL_TASKS)} Legal tasks")


def get_legal_tasks() -> List[mteb.AbsTask]:
    """Get all registered Legal tasks as MTEB task objects."""
    register_all_legal_tasks()
    tasks = []
    for task_config in LEGAL_TASKS:
        try:
            task = mteb.get_task(task_config.name)
            tasks.append(task)
        except Exception as e:
            logger.error(f"Failed to retrieve Legal task '{task_config.name}': {e}")
    logger.info(f"Retrieved {len(tasks)} Legal tasks")
    return tasks


def get_legal_task_category(task_name: str) -> Optional[str]:
    """Get the category for a specific Legal task."""
    for task_config in LEGAL_TASKS:
        if task_config.name == task_name:
            return task_config.category
    return None


# =============================================================================
# Model2Vec Adapter
# =============================================================================

class Model2VecAdapter:
    """
    Adapter to make model2vec StaticModel compatible with sentence-transformers interface.
    """
    
    def __init__(self, model, model_name: str = None):
        self.model = model
        self.model_name = model_name or "model2vec-model"
        self._device = "cpu"
        
        try:
            test_embedding = self.model.encode(["test"])
            self._embedding_dimension = test_embedding.shape[1] if len(test_embedding.shape) > 1 else len(test_embedding[0])
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            self._embedding_dimension = None
    
    def encode(self, sentences: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress_bar: bool = True,
               convert_to_tensor: bool = True,
               **kwargs) -> Union[List[List[float]], np.ndarray, torch.Tensor]:
        """Encode sentences using model2vec."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        try:
            embeddings = self.model.encode(sentences)
            
            if convert_to_tensor:
                if isinstance(embeddings, np.ndarray):
                    return torch.from_numpy(embeddings)
                else:
                    return torch.tensor(embeddings)
            else:
                if isinstance(embeddings, torch.Tensor):
                    return embeddings.numpy()
                elif isinstance(embeddings, list):
                    return np.array(embeddings)
                else:
                    return embeddings
        except Exception as e:
            logger.error(f"Error encoding sentences: {e}")
            raise
    
    def get_sentence_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of sentence embeddings."""
        return self._embedding_dimension
    
    @property
    def device(self):
        return self._device
    
    def parameters(self):
        return iter([])
    
    def to(self, device):
        if device != "cpu":
            logger.warning("model2vec models are CPU-based and cannot be moved to GPU")
        return self
    
    def eval(self):
        return self
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def load_model2vec_model(model_name_or_path: str) -> Model2VecAdapter:
    """Load a model2vec model and wrap it with the adapter."""
    try:
        from model2vec import StaticModel
        logger.info(f"Loading model2vec model: {model_name_or_path}")
        model = StaticModel.from_pretrained(model_name_or_path)
        adapter = Model2VecAdapter(model, model_name_or_path)
        logger.info(f"Successfully loaded model2vec model with dimension {adapter.get_sentence_embedding_dimension()}")
        return adapter
    except Exception as e:
        logger.error(f"Failed to load model2vec model {model_name_or_path}: {e}")
        raise


# =============================================================================
# Turkish Uncased Model Handling
# =============================================================================

def _turkish_lower(text: str) -> str:
    """
    Apply Turkish-specific lowercase conversion.
    
    Turkish has special casing rules:
    - I (dotless capital I) -> ı (dotless lowercase i)
    - İ (dotted capital I) -> i (dotted lowercase i)
    
    Standard Python .lower() incorrectly converts I->i, which breaks
    Turkish uncased model vocabularies that expect ı.
    """
    text = text.replace('I', 'ı')  # Dotless I -> dotless ı
    text = text.replace('İ', 'i')  # Dotted İ -> dotted i
    return text.lower()


def _is_turkish_uncased_model(model_name: str) -> bool:
    """
    Detect if a model is a Turkish uncased model requiring special lowercase handling.
    """
    name_lower = model_name.lower()
    
    # Must contain "uncased" 
    if "uncased" not in name_lower:
        return False
    
    # Must be Turkish-related
    turkish_indicators = ["turkish", "turk", "/tr-", "-tr-", "_tr_", "/tr_"]
    return any(indicator in name_lower for indicator in turkish_indicators)


class TurkishUncasedModelWrapper:
    """
    Wrapper for Turkish uncased models that applies Turkish-specific lowercase
    conversion before encoding.
    """
    
    def __init__(self, model, model_name: str):
        self._model = model
        self._model_name = model_name
        logger.info(f"Wrapped {model_name} with Turkish lowercase preprocessing")
    
    def __getattr__(self, name: str):
        """Proxy all attribute access to the underlying model."""
        return getattr(self._model, name)
    
    def encode(self, sentences, **kwargs):
        """Encode sentences with Turkish lowercase preprocessing."""
        if isinstance(sentences, str):
            sentences = _turkish_lower(sentences)
        else:
            sentences = [_turkish_lower(s) for s in sentences]
        
        return self._model.encode(sentences, **kwargs)
    
    def __repr__(self) -> str:
        return f"TurkishUncasedModelWrapper({self._model_name})"


# =============================================================================
# Model Loader
# =============================================================================

def get_model_max_position_embeddings(model) -> Optional[int]:
    """Extract max_position_embeddings from the loaded model's transformer config."""
    try:
        if hasattr(model, '_first_module') and model._first_module is not None:
            transformer = model._first_module()
            if hasattr(transformer, 'auto_model'):
                config = transformer.auto_model.config
                if hasattr(config, 'max_position_embeddings'):
                    return config.max_position_embeddings
        
        if hasattr(model, '_modules') and len(model._modules) > 0:
            first_module = list(model._modules.values())[0]
            if hasattr(first_module, 'auto_model'):
                config = first_module.auto_model.config
                if hasattr(config, 'max_position_embeddings'):
                    return config.max_position_embeddings
    except Exception as e:
        logger.debug(f"Could not extract max_position_embeddings: {e}")
    
    return None


def load_sentence_transformer(
    model_name: str,
    trust_remote_code: bool = True,
    device: Optional[str] = None,
    max_seq_length_limit: int = MAX_SEQUENCE_LENGTH_LIMIT,
    apply_turkish_preprocessing: bool = True,
    **kwargs
):
    """
    Load a SentenceTransformer model with optimized max_seq_length.
    
    Automatically detects and overrides suboptimal max_seq_length settings.
    For Turkish uncased models, automatically wraps with Turkish-specific 
    lowercase preprocessing (I->ı, İ->i).
    """
    from sentence_transformers import SentenceTransformer
    
    load_kwargs = {
        'trust_remote_code': trust_remote_code,
        **kwargs
    }
    
    if device is not None:
        load_kwargs['device'] = device
    
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, **load_kwargs)
    
    original_max_seq_length = model.max_seq_length
    max_position_embeddings = get_model_max_position_embeddings(model)
    
    if max_position_embeddings is not None:
        optimal_max_seq_length = min(
            max(original_max_seq_length, max_position_embeddings),
            max_seq_length_limit
        )
        
        if original_max_seq_length != optimal_max_seq_length:
            model.max_seq_length = optimal_max_seq_length
            logger.info(
                f"Adjusted max_seq_length for {model_name}: "
                f"{original_max_seq_length} -> {optimal_max_seq_length} "
                f"(capacity: {max_position_embeddings}, limit: {max_seq_length_limit})"
            )
    else:
        if original_max_seq_length > max_seq_length_limit:
            model.max_seq_length = max_seq_length_limit
            logger.info(
                f"Limited max_seq_length for {model_name}: "
                f"{original_max_seq_length} -> {max_seq_length_limit}"
            )
    
    # Wrap Turkish uncased models with preprocessing
    if apply_turkish_preprocessing and _is_turkish_uncased_model(model_name):
        return TurkishUncasedModelWrapper(model, model_name)
    
    return model


# =============================================================================
# MTEB Evaluator
# =============================================================================

class MTEBEvaluator:
    """
    MTEB Evaluator for Turkish language and legal domain tasks.
    
    Handles the complete MTEB evaluation pipeline including:
    - Model loading and validation (sentence-transformers, model2vec)
    - Turkish task filtering and selection
    - Legal task registration and evaluation
    - Result processing and summary generation
    """
    
    def __init__(self, 
                 model_name: str,
                 output_dir: str = None,
                 batch_size: int = None,
                 trust_remote_code: bool = True,
                 device: str = None,
                 cancel_flag: threading.Event = None,
                 overwrite_results: bool = False):
        """
        Initialize the MTEB evaluator.
        
        Args:
            model_name: HuggingFace model identifier or local path
            output_dir: Directory to save evaluation results
            batch_size: Batch size for evaluation
            trust_remote_code: Whether to trust remote code for model loading
            device: Device to load the model on ('cuda', 'cpu', 'cuda:0', 'cuda:1', etc.)
            cancel_flag: Threading event for cancellation support
            overwrite_results: Force re-evaluation, overwrite existing results
        """
        self.model_name = model_name
        self.output_dir = output_dir or DEFAULT_SETTINGS["output_dir"]
        self.batch_size = batch_size or DEFAULT_SETTINGS["batch_size"]
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.cancel_flag = cancel_flag
        self.overwrite_results = overwrite_results
        
        self.sentence_model = None
        self.mteb_model = None
        self.turkish_tasks = None
        self.legal_tasks = None
        self.model_metadata = {}
        self.tokenizer = None
        self.detected_model_type = None
        
        self.safe_model_name = convert_model_name_to_safe_filename(model_name)
        self.model_output_dir = os.path.join(self.output_dir, self.safe_model_name)
        
        logger.info(f"Initialized MTEBEvaluator for {model_name}")
    
    def load_model(self) -> bool:
        """Automatically detect and load the model."""
        logger.info(f"Auto-detecting model type for: {self.model_name}")
        
        # Try sentence-transformers first
        logger.info("Attempting to load as sentence-transformers model...")
        if self._try_load_sentence_transformer():
            self.detected_model_type = "sentence-transformers"
            logger.info(f"Successfully loaded {self.model_name} as sentence-transformers model")
            return True
        
        # Try model2vec
        logger.info("Sentence-transformers loading failed, attempting model2vec...")
        if self._try_load_model2vec():
            self.detected_model_type = "model2vec"
            logger.info(f"Successfully loaded {self.model_name} as model2vec model")
            return True
        
        logger.error(f"Failed to load {self.model_name} with any available adapter")
        return False
    
    def _try_load_sentence_transformer(self) -> bool:
        """Try to load as sentence transformer model."""
        try:
            from transformers import AutoConfig
            
            # Check if this is a model2vec model
            try:
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
                if getattr(config, 'model_type', None) == 'model2vec':
                    logger.info("Detected model_type='model2vec', skipping sentence-transformers loader")
                    return False
            except Exception as e:
                if 'model2vec' in str(e).lower():
                    return False
            
            # Determine device
            if self.device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            logger.info(f"Using device: {device}")
            
            self.sentence_model = load_sentence_transformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                device=device,
                max_seq_length_limit=MAX_SEQUENCE_LENGTH_LIMIT
            )
            
            # Load tokenizer
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
            
            self.mteb_model = mteb.SentenceTransformerWrapper(self.sentence_model)
            self.model_metadata = self._extract_model_metadata()
            
            return True
            
        except Exception as e:
            logger.info(f"Could not load as sentence-transformers model: {e}")
            return False
    
    def _try_load_model2vec(self) -> bool:
        """Try to load as model2vec model."""
        try:
            self.sentence_model = load_model2vec_model(self.model_name)
            
            class Model2VecMTEBWrapper:
                def __init__(self, model):
                    self.model = model
                
                def encode(self, *args, **kwargs):
                    return self.model.encode(*args, **kwargs)
            
            self.mteb_model = Model2VecMTEBWrapper(self.sentence_model)
            self.model_metadata = self._extract_model_metadata()
            
            return True
            
        except Exception as e:
            logger.info(f"Could not load as model2vec model: {e}")
            return False
    
    def _extract_model_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the loaded model."""
        metadata = {}
        is_model2vec = isinstance(self.sentence_model, Model2VecAdapter)
        
        try:
            if hasattr(self.sentence_model, 'get_sentence_embedding_dimension'):
                metadata['embed_dim'] = self.sentence_model.get_sentence_embedding_dimension()
            
            try:
                if is_model2vec:
                    model = self.sentence_model.model
                    if hasattr(model, 'tokens') and hasattr(model, 'dim'):
                        vocab_size = len(model.tokens)
                        embed_dim = model.dim
                        total_params = vocab_size * embed_dim
                        metadata['n_parameters'] = total_params
                else:
                    total_params = sum(
                        p.numel() for p in self.sentence_model.parameters() 
                        if hasattr(p, 'numel')
                    )
                    if total_params > 0:
                        metadata['n_parameters'] = total_params
            except Exception as e:
                logger.debug(f"Could not count parameters: {e}")
            
            try:
                if is_model2vec:
                    metadata['model_architecture'] = "StaticModel"
                else:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(self.model_name)
                    if hasattr(config, 'architectures') and config.architectures:
                        metadata['model_architecture'] = config.architectures[0]
                    else:
                        metadata['model_architecture'] = "Unknown"
                    
                    for attr in ['max_position_embeddings', 'n_positions', 'seq_length']:
                        if hasattr(config, attr):
                            value = getattr(config, attr)
                            if value and isinstance(value, int) and value < 1e30:
                                metadata['max_tokens'] = int(value)
                                break
            except Exception as e:
                logger.debug(f"Could not get config information: {e}")
                metadata['model_architecture'] = "Unknown"
            
        except Exception as e:
            logger.warning(f"Error extracting model metadata: {e}")
        
        return metadata
    
    def get_turkish_tasks(self):
        """Get and filter Turkish language tasks from MTEB."""
        if self.turkish_tasks is not None:
            return self.turkish_tasks
        
        try:
            all_tasks = mteb.get_tasks(
                languages=["tur"], 
                modalities=["text"], 
                exclusive_modality_filter=True,
                exclusive_language_filter=True
            )
            
            LEGAL_TASK_NAMES = {tc.name for tc in LEGAL_TASKS}
            
            self.turkish_tasks = [
                task for task in all_tasks 
                if task.languages == ["tur"] 
                and task.metadata.name not in LEGAL_TASK_NAMES
            ]
            
            logger.info(f"Found {len(self.turkish_tasks)} Turkish tasks:")
            for i, task in enumerate(self.turkish_tasks):
                logger.info(f"  {i+1}. {task.metadata.name} - {task.metadata.type}")
            
            return self.turkish_tasks
            
        except Exception as e:
            logger.error(f"Failed to get Turkish tasks: {e}")
            return []
    
    def get_legal_tasks(self):
        """Get legal domain tasks."""
        if self.legal_tasks is not None:
            return self.legal_tasks
        
        try:
            self.legal_tasks = get_legal_tasks()
            
            logger.info(f"Found {len(self.legal_tasks)} Legal tasks:")
            for i, task in enumerate(self.legal_tasks):
                category = get_legal_task_category(task.metadata.name)
                logger.info(f"  {i+1}. {task.metadata.name} - {category}")
            
            return self.legal_tasks
            
        except Exception as e:
            logger.error(f"Failed to get Legal tasks: {e}")
            return []
    
    def get_all_tasks(self):
        """Get all tasks (Turkish + Legal)."""
        turkish_tasks = self.get_turkish_tasks()
        legal_tasks = self.get_legal_tasks()
        
        all_tasks = turkish_tasks + legal_tasks
        
        logger.info(f"Total tasks to evaluate: {len(all_tasks)} " +
                   f"(Turkish: {len(turkish_tasks)}, Legal: {len(legal_tasks)})")
        
        return all_tasks
    
    def _check_cancelled(self, operation: str = "") -> bool:
        """Check if evaluation was cancelled."""
        if self.cancel_flag and self.cancel_flag.is_set():
            logger.info(f"Evaluation cancelled{': ' + operation if operation else ''}")
            return True
        return False
    
    def run_evaluation(self) -> Tuple[bool, Optional[List[Any]], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Run the complete MTEB evaluation."""
        if not self.sentence_model or not self.mteb_model:
            logger.error("Model not loaded. Call load_model() first.")
            return False, None, None, None
        
        if self._check_cancelled("before evaluation start"):
            return False, None, None, None
        
        all_tasks = self.get_all_tasks()
        if not all_tasks:
            logger.error("No tasks found to evaluate.")
            return False, None, None, None
        
        try:
            logger.info(f"Starting evaluation for {self.model_name}")
            logger.info(f"Tasks to evaluate: {[task.metadata.name for task in all_tasks]}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            os.makedirs(self.model_output_dir, exist_ok=True)
            
            wrapped_model = self._create_cancellation_aware_model(self.mteb_model)
            
            results = []
            for task in all_tasks:
                task_name = task.metadata.name
                logger.info(f"Starting task: {task_name}")
                
                if self._check_cancelled(f"before task {task_name}"):
                    break
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                task_start = time.time()
                
                try:
                    evaluation = mteb.MTEB(tasks=[task])
                    task_results = evaluation.run(
                        wrapped_model,
                        output_folder=self.model_output_dir,
                        encode_kwargs={"batch_size": self.batch_size},
                        raise_error=False,
                        overwrite_results=self.overwrite_results
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    task_duration = time.time() - task_start
                    
                    logger.info(f"Task {task_name} completed in {task_duration:.2f}s")
                    
                    if task_results:
                        results.extend(task_results)
                    
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    task_duration = time.time() - task_start
                    
                    logger.error(f"Task {task_name} failed: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            duration = time.time() - start_time
            
            successful_tasks = len(results) if results else 0
            total_tasks = len(all_tasks)
            
            logger.info(f"Evaluation completed in {duration:.2f}s ({duration/60:.2f}m)")
            logger.info(f"Task results: {successful_tasks}/{total_tasks} successful")
            
            summary_df, summary_table_df = self._process_results(results)
            self._save_model_metadata()
            
            return successful_tasks > 0, results, summary_df, summary_table_df
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, None, None, None
    
    def _process_results(self, results: List[Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Process raw MTEB results into a summary DataFrame."""
        try:
            summary_data = []
            
            for task_result in results:
                task_name = task_result.task_name
                main_score = None
                
                for split_name in ["test", "validation", "dev", "train"]:
                    if split_name in task_result.scores:
                        split_scores = task_result.scores[split_name]
                        if isinstance(split_scores, list) and len(split_scores) > 0:
                            for item in split_scores:
                                if "main_score" in item:
                                    main_score = item["main_score"] * 100
                                    break
                        if main_score is not None:
                            break
                
                if main_score is not None:
                    summary_data.append({
                        "Task": task_name,
                        "Score": main_score
                    })
                    logger.info(f"{task_name}: {main_score:.2f}")
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                
                summary_path = os.path.join(self.model_output_dir, "task_summary.csv")
                df.to_csv(summary_path, index=False)
                logger.info(f"Task summary saved to {summary_path}")
                
                # Separate tasks
                legal_task_names = {tc.name for tc in LEGAL_TASKS}
                turkish_tasks_df = df[~df["Task"].isin(legal_task_names)]
                
                # Calculate category scores
                category_scores = {}
                for category, task_names in TASK_CATEGORIES.items():
                    category_tasks = df[df["Task"].isin(task_names)]
                    if not category_tasks.empty:
                        category_score = category_tasks["Score"].mean()
                        category_scores[category] = category_score
                        logger.info(f"{category} Score: {category_score:.2f}")
                
                # 1. Mean (Task) Score - average of all individual Turkish MTEB task scores
                mean_task_score = None
                if not turkish_tasks_df.empty:
                    mean_task_score = turkish_tasks_df["Score"].mean()
                    logger.info(f"Mean (Task) Score: {mean_task_score:.2f}")
                
                # 2. Mean (TaskType) Score - average of category averages for Turkish MTEB
                turkish_categories = ["Classification", "Clustering", "PairClassification", "Retrieval", "STS"]
                turkish_category_scores = [score for cat, score in category_scores.items() if cat in turkish_categories]
                mean_tasktype_score = None
                if turkish_category_scores:
                    mean_tasktype_score = sum(turkish_category_scores) / len(turkish_category_scores)
                    logger.info(f"Mean (TaskType) Score: {mean_tasktype_score:.2f}")
                
                # 3. Legal Score - average of legal category scores
                legal_categories = ["Contracts", "Regulation", "Caselaw"]
                legal_category_scores = [score for cat, score in category_scores.items() if cat in legal_categories]
                legal_score = None
                if legal_category_scores:
                    legal_score = sum(legal_category_scores) / len(legal_category_scores)
                    logger.info(f"Legal Score: {legal_score:.2f}")
                
                # Create summary table with parameter count
                n_params = self.model_metadata.get('n_parameters')
                param_str = format_parameter_count(n_params) if n_params else None
                
                summary_table = {
                    "Model": self.model_name,
                    "Contracts": category_scores.get("Contracts"),
                    "Regulation": category_scores.get("Regulation"),
                    "Caselaw": category_scores.get("Caselaw"),
                    "Score(Legal)": legal_score,
                    "Mean (Task)": mean_task_score,
                    "Mean (TaskType)": mean_tasktype_score,
                    "Classification": category_scores.get("Classification"),
                    "Clustering": category_scores.get("Clustering"),
                    "Pair Classification": category_scores.get("PairClassification"),
                    "Retrieval": category_scores.get("Retrieval"),
                    "STS": category_scores.get("STS"),
                    "Parameters": param_str
                }
                
                summary_table_df = pd.DataFrame([summary_table])
                
                return df, summary_table_df
            
            return None, None
                
        except Exception as e:
            logger.error(f"Failed to process results: {e}")
            return None, None
    
    def _save_model_metadata(self) -> bool:
        """Save model metadata to JSON file."""
        try:
            meta_path = os.path.join(self.model_output_dir, "model_meta.json")
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            logger.info(f"Model metadata saved to {meta_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
            return False
    
    def _create_cancellation_aware_model(self, model):
        """Create a wrapper that checks for cancellation during encoding."""
        
        class CancellationAwareModel:
            def __init__(self, wrapped_model, evaluator):
                self.wrapped_model = wrapped_model
                self.evaluator = evaluator
                self.encode_call_count = 0
            
            def encode(self, *args, **kwargs):
                self.encode_call_count += 1
                if self.evaluator._check_cancelled(f"during encoding"):
                    raise RuntimeError("Evaluation cancelled by user")
                return self.wrapped_model.encode(*args, **kwargs)
            
            def __getattr__(self, name):
                return getattr(self.wrapped_model, name)
        
        return CancellationAwareModel(model, self)
    
    def cleanup(self) -> None:
        """Clean up resources used by the evaluator."""
        try:
            self.sentence_model = None
            self.mteb_model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            logger.info("Evaluator cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the Legal evaluation script."""
    parser = argparse.ArgumentParser(
        description="Legal Evaluation Script - Unified Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python legal_evaluation.py --model "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
  python legal_evaluation.py --model "model_name" --output-dir "./results" --batch-size 64
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model identifier or local path"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Device to load model on (cuda, cpu, cuda:0, cuda:1, etc.)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-evaluation, overwrite existing results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # List tasks mode
    if args.list_tasks:
        logger.info("Listing available tasks...")
        
        # Get Turkish MTEB tasks
        turkish_tasks = mteb.get_tasks(
            languages=["tur"], 
            modalities=["text"],
            exclusive_modality_filter=True,
            exclusive_language_filter=True
        )
        
        print("\n=== Turkish MTEB Tasks ===")
        for i, task in enumerate(turkish_tasks):
            print(f"  {i+1}. {task.metadata.name} - {task.metadata.type}")
        
        print("\n=== Legal Domain Tasks ===")
        for i, tc in enumerate(LEGAL_TASKS):
            print(f"  {i+1}. {tc.name} - {tc.category}")
        
        return
    
    # Initialize evaluator
    evaluator = MTEBEvaluator(
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        overwrite_results=args.overwrite
    )
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    if not evaluator.load_model():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    logger.info(f"Model loaded successfully as {evaluator.detected_model_type}")
    logger.info(f"Model metadata: {evaluator.model_metadata}")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    success, results, summary_df, summary_table_df = evaluator.run_evaluation()
    
    if success:
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        if summary_table_df is not None:
            # Append to CSV file in output directory
            summary_csv_path = os.path.join(args.output_dir, "summary_results.csv")
            
            # Check if model already exists in CSV
            model_exists = False
            should_update = False
            if os.path.exists(summary_csv_path):
                existing_df = pd.read_csv(summary_csv_path)
                if args.model in existing_df["Model"].values:
                    model_exists = True
                    # Check if existing row has missing values in important columns
                    existing_row = existing_df[existing_df["Model"] == args.model].iloc[0]
                    important_cols = ["Contracts", "Regulation", "Caselaw", "Score(Legal)"]
                    has_missing = any(pd.isna(existing_row[col]) for col in important_cols if col in existing_row.index)
                    
                    if has_missing:
                        should_update = True
                        logger.info(f"Model '{args.model}' exists but has missing values. Updating...")
                    else:
                        logger.warning(f"Model '{args.model}' already exists with complete data. Skipping.")
            
            if not model_exists:
                # Model doesn't exist, append it
                if os.path.exists(summary_csv_path):
                    # Append without header
                    summary_table_df.to_csv(summary_csv_path, mode='a', header=False, index=False)
                else:
                    # Create new file with header
                    summary_table_df.to_csv(summary_csv_path, mode='w', header=True, index=False)
                
                logger.info(f"Summary appended to: {summary_csv_path}")
            elif should_update:
                # Model exists but has missing values, update it
                existing_df = pd.read_csv(summary_csv_path)
                # Remove old row and append new one
                existing_df = existing_df[existing_df["Model"] != args.model]
                updated_df = pd.concat([existing_df, summary_table_df], ignore_index=True)
                updated_df.to_csv(summary_csv_path, index=False)
                logger.info(f"Summary updated in: {summary_csv_path}")
            
            print("\n=== Summary Table ===")
            print(summary_table_df.to_string(index=False))
        
        logger.info(f"Results saved to: {evaluator.model_output_dir}")
    else:
        logger.error("Evaluation failed or was cancelled.")
        sys.exit(1)
    
    # Cleanup
    evaluator.cleanup()


if __name__ == "__main__":
    main()
