"""
LLM-USE V2 - Universal LLM Router with Auto-Discovery
Sistema di routing intelligente con valutazione realistica dei modelli
Production Ready with Full Monitoring, Caching, and Rate Limiting
"""

import os
import json
import time
import hashlib
import re
import sqlite3
import threading
import logging
import pickle
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import requests
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import lru_cache, wraps
from enum import Enum
import traceback
import uuid

# Ollama per modelli locali
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. Install with: pip install ollama")

# ====================
# PRODUCTION LOGGING
# ====================

class LogLevel(Enum):
    """Log levels for production logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ProductionLogger:
    """Enhanced production logger with structured logging and rotation"""
    
    def __init__(self, log_dir: str = None, app_name: str = "llm-use", 
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 max_files: int = 10):
        """
        Initialize production logger
        
        Args:
            log_dir: Directory for log files
            app_name: Application name for log files
            max_file_size: Maximum size per log file in bytes
            max_files: Maximum number of rotated files
        """
        self.app_name = app_name
        self.max_file_size = max_file_size
        self.max_files = max_files
        
        # Setup log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path.home() / ".llm-use" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_log_file = self.log_dir / f"{app_name}.log"
        self.log_lock = threading.Lock()
        
        # Setup Python logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        self._setup_handlers()
        
        # Structured log buffer for analysis
        self.log_buffer = deque(maxlen=1000)
        
        # Log session start
        self.session_id = str(uuid.uuid4())[:8]
        self.info(f"Logger initialized", extra={"session_id": self.session_id})
    
    def _setup_handlers(self):
        """Setup logging handlers with rotation"""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.current_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s | %(extra_data)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add custom filter for extra data
        file_handler.addFilter(self._add_extra_data)
        self.logger.addHandler(file_handler)
    
    def _add_extra_data(self, record):
        """Add extra data to log record"""
        extra_data = {}
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'pathname', 'process', 'processName', 'relativeCreated', 
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                extra_data[key] = value
        
        record.extra_data = json.dumps(extra_data) if extra_data else ''
        return True
    
    def _rotate_logs(self):
        """Rotate log files when size limit reached"""
        with self.log_lock:
            if not self.current_log_file.exists():
                return
            
            if self.current_log_file.stat().st_size < self.max_file_size:
                return
            
            # Rotate files
            for i in range(self.max_files - 1, 0, -1):
                old_file = self.log_dir / f"{self.app_name}.log.{i}"
                new_file = self.log_dir / f"{self.app_name}.log.{i + 1}"
                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)
            
            # Move current to .1
            first_backup = self.log_dir / f"{self.app_name}.log.1"
            if first_backup.exists():
                first_backup.unlink()
            self.current_log_file.rename(first_backup)
            
            # Recreate handlers
            self._setup_handlers()
    
    def _log(self, level: LogLevel, message: str, extra: Dict = None, 
             exc_info: bool = False):
        """Internal log method"""
        # Check rotation
        self._rotate_logs()
        
        # Prepare extra data
        log_extra = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            **(extra or {})
        }
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'session_id': self.session_id,
            'extra': extra or {}
        }
        
        # Add to buffer
        self.log_buffer.append(log_entry)
        
        # Log based on level
        if level == LogLevel.DEBUG:
            self.logger.debug(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.INFO:
            self.logger.info(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.WARNING:
            self.logger.warning(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.ERROR:
            self.logger.error(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, extra=log_extra, exc_info=exc_info)
    
    def debug(self, message: str, extra: Dict = None):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, extra)
    
    def info(self, message: str, extra: Dict = None):
        """Log info message"""
        self._log(LogLevel.INFO, message, extra)
    
    def warning(self, message: str, extra: Dict = None):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, extra)
    
    def error(self, message: str, extra: Dict = None, exc_info: bool = True):
        """Log error message"""
        self._log(LogLevel.ERROR, message, extra, exc_info)
    
    def critical(self, message: str, extra: Dict = None, exc_info: bool = True):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, extra, exc_info)
    
    def log_api_call(self, provider: str, model: str, input_tokens: int, 
                     output_tokens: int, latency: float, success: bool, 
                     error: str = None):
        """Log API call details"""
        self.info("API call", extra={
            'provider': provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'latency_ms': int(latency * 1000),
            'success': success,
            'error': error
        })
    
    def log_cache_event(self, event_type: str, key: str, hit: bool = None,
                       size_bytes: int = None):
        """Log cache events"""
        self.debug(f"Cache {event_type}", extra={
            'cache_key': key[:50],
            'cache_hit': hit,
            'size_bytes': size_bytes
        })
    
    def log_rate_limit(self, provider: str, remaining: int, reset_time: str):
        """Log rate limit events"""
        self.warning("Rate limit approaching", extra={
            'provider': provider,
            'remaining_calls': remaining,
            'reset_time': reset_time
        })
    
    def get_recent_errors(self, count: int = 10) -> List[Dict]:
        """Get recent error logs"""
        errors = [log for log in self.log_buffer 
                 if log['level'] in ['ERROR', 'CRITICAL']]
        return errors[-count:]
    
    def export_logs(self, start_time: datetime = None, 
                   end_time: datetime = None) -> List[Dict]:
        """Export logs for analysis"""
        logs = list(self.log_buffer)
        
        if start_time:
            logs = [log for log in logs 
                   if datetime.fromisoformat(log['timestamp']) >= start_time]
        
        if end_time:
            logs = [log for log in logs 
                   if datetime.fromisoformat(log['timestamp']) <= end_time]
        
        return logs

# ====================
# PRODUCTION DATABASE
# ====================

class ProductionDB:
    """SQLite database for production tracking and analytics"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize production database
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path.home() / ".llm-use" / "production.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    total_tokens_input INTEGER DEFAULT 0,
                    total_tokens_output INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0,
                    user_id TEXT,
                    metadata TEXT
                )
            """)
            
            # API calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    call_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    latency_ms INTEGER,
                    cost REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    complexity_score INTEGER,
                    cache_hit BOOLEAN DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    avg_latency_ms REAL,
                    success_rate REAL,
                    total_calls INTEGER,
                    total_tokens INTEGER,
                    total_cost REAL,
                    quality_score REAL,
                    period_start TIMESTAMP,
                    period_end TIMESTAMP
                )
            """)
            
            # Cache performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cache_hits INTEGER,
                    cache_misses INTEGER,
                    cache_size_bytes INTEGER,
                    evictions INTEGER,
                    avg_retrieval_ms REAL,
                    memory_saved_bytes INTEGER
                )
            """)
            
            # Rate limit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT,
                    limit_type TEXT,
                    remaining_calls INTEGER,
                    reset_time TIMESTAMP,
                    action_taken TEXT
                )
            """)
            
            # User feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    call_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rating INTEGER,
                    feedback_text TEXT,
                    feedback_type TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (call_id) REFERENCES api_calls(call_id)
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_session ON api_calls(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_model ON api_calls(model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_timestamp ON model_performance(timestamp)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_session(self, user_id: str = None, metadata: Dict = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, metadata)
                VALUES (?, ?, ?)
            """, (session_id, user_id, json.dumps(metadata) if metadata else None))
            conn.commit()
        
        return session_id
    
    def update_session(self, session_id: str, messages: int = None, 
                      tokens_input: int = None, tokens_output: int = None,
                      cost: float = None, end_time: bool = False):
        """Update session statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            values = []
            
            if messages is not None:
                updates.append("total_messages = total_messages + ?")
                values.append(messages)
            
            if tokens_input is not None:
                updates.append("total_tokens_input = total_tokens_input + ?")
                values.append(tokens_input)
            
            if tokens_output is not None:
                updates.append("total_tokens_output = total_tokens_output + ?")
                values.append(tokens_output)
            
            if cost is not None:
                updates.append("total_cost = total_cost + ?")
                values.append(cost)
            
            if end_time:
                updates.append("end_time = CURRENT_TIMESTAMP")
            
            if updates:
                values.append(session_id)
                cursor.execute(f"""
                    UPDATE sessions 
                    SET {', '.join(updates)}
                    WHERE session_id = ?
                """, values)
                conn.commit()
    
    def log_api_call(self, session_id: str, provider: str, model: str,
                    input_text: str, output_text: str, input_tokens: int,
                    output_tokens: int, latency_ms: int, cost: float,
                    success: bool, error_message: str = None,
                    complexity_score: int = None, cache_hit: bool = False) -> str:
        """Log API call to database"""
        call_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_calls (
                    call_id, session_id, provider, model, input_text, 
                    output_text, input_tokens, output_tokens, latency_ms,
                    cost, success, error_message, complexity_score, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id, session_id, provider, model, input_text[:1000],
                output_text[:1000] if output_text else None,
                input_tokens, output_tokens, latency_ms, cost,
                success, error_message, complexity_score, cache_hit
            ))
            conn.commit()
        
        return call_id
    
    def update_model_performance(self, model: str, provider: str,
                                period_minutes: int = 60):
        """Calculate and store model performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate metrics for the period
            period_start = datetime.now() - timedelta(minutes=period_minutes)
            
            cursor.execute("""
                SELECT 
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(*) as total_calls,
                    SUM(input_tokens + output_tokens) as total_tokens,
                    SUM(cost) as total_cost
                FROM api_calls
                WHERE model = ? AND provider = ? AND timestamp >= ?
            """, (model, provider, period_start))
            
            row = cursor.fetchone()
            
            if row and row['total_calls'] > 0:
                cursor.execute("""
                    INSERT INTO model_performance (
                        model, provider, avg_latency_ms, success_rate,
                        total_calls, total_tokens, total_cost,
                        period_start, period_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model, provider, row['avg_latency'], row['success_rate'],
                    row['total_calls'], row['total_tokens'] or 0, 
                    row['total_cost'] or 0, period_start, datetime.now()
                ))
                conn.commit()
    
    def log_cache_performance(self, hits: int, misses: int, size_bytes: int,
                            evictions: int, avg_retrieval_ms: float,
                            memory_saved: int):
        """Log cache performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cache_performance (
                    cache_hits, cache_misses, cache_size_bytes,
                    evictions, avg_retrieval_ms, memory_saved_bytes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (hits, misses, size_bytes, evictions, avg_retrieval_ms, memory_saved))
            conn.commit()
    
    def log_rate_limit_event(self, provider: str, model: str = None,
                           limit_type: str = None, remaining: int = None,
                           reset_time: datetime = None, action: str = None):
        """Log rate limit event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rate_limit_events (
                    provider, model, limit_type, remaining_calls,
                    reset_time, action_taken
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (provider, model, limit_type, remaining, reset_time, action))
            conn.commit()
    
    def add_user_feedback(self, session_id: str = None, call_id: str = None,
                        rating: int = None, feedback_text: str = None,
                        feedback_type: str = None):
        """Add user feedback"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_feedback (
                    session_id, call_id, rating, feedback_text, feedback_type
                ) VALUES (?, ?, ?, ?, ?)
            """, (session_id, call_id, rating, feedback_text, feedback_type))
            conn.commit()
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get session statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}
    
    def get_model_analytics(self, hours: int = 24) -> List[Dict]:
        """Get model performance analytics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            since = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT 
                    model,
                    provider,
                    COUNT(*) as total_calls,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    SUM(cost) as total_cost,
                    AVG(complexity_score) as avg_complexity,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as cache_hit_rate
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY model, provider
                ORDER BY total_calls DESC
            """, (since,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_error_analysis(self, hours: int = 24) -> List[Dict]:
        """Analyze recent errors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            since = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT 
                    model,
                    provider,
                    error_message,
                    COUNT(*) as error_count,
                    MAX(timestamp) as last_occurrence
                FROM api_calls
                WHERE success = 0 AND timestamp >= ?
                GROUP BY model, provider, error_message
                ORDER BY error_count DESC
            """, (since,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=days)
            
            # Delete old API calls
            cursor.execute("DELETE FROM api_calls WHERE timestamp < ?", (cutoff,))
            
            # Delete old sessions
            cursor.execute("DELETE FROM sessions WHERE start_time < ?", (cutoff,))
            
            # Delete old performance data
            cursor.execute("DELETE FROM model_performance WHERE timestamp < ?", (cutoff,))
            
            # Delete old cache performance
            cursor.execute("DELETE FROM cache_performance WHERE timestamp < ?", (cutoff,))
            
            conn.commit()

# ====================
# PRODUCTION CACHE
# ====================

class CacheEntry:
    """Single cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.hits = 0
        self.last_accessed = time.time()
        self.size_bytes = len(pickle.dumps(value))
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Update access statistics"""
        self.hits += 1
        self.last_accessed = time.time()

class ProductionCache:
    """Context-aware production cache with compression and eviction"""
    
    def __init__(self, max_size_mb: int = 100, ttl_seconds: int = 3600,
                 compression: bool = True, logger: ProductionLogger = None):
        """
        Initialize production cache
        
        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Default TTL in seconds
            compression: Enable compression for large values
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.compression = compression
        self.logger = logger
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size': 0,
            'compression_savings': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, provider: str, model: str, messages: List[Dict],
                     temperature: float = None, max_tokens: int = None) -> str:
        """Generate cache key from request parameters"""
        # Create deterministic key
        key_data = {
            'provider': provider,
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value if beneficial"""
        pickled = pickle.dumps(value)
        
        if self.compression and len(pickled) > 1024:  # Only compress if > 1KB
            compressed = zlib.compress(pickled, level=6)
            
            if len(compressed) < len(pickled) * 0.9:  # Only use if 10% smaller
                self.stats['compression_savings'] += len(pickled) - len(compressed)
                return compressed
        
        return pickled
    
    def _decompress_value(self, data: bytes) -> Any:
        """Decompress value if needed"""
        try:
            # Try decompression first
            if self.compression:
                try:
                    decompressed = zlib.decompress(data)
                    return pickle.loads(decompressed)
                except zlib.error:
                    pass
            
            # Fall back to direct unpickling
            return pickle.loads(data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache decompression error: {e}")
            return None
    
    def get(self, provider: str, model: str, messages: List[Dict],
            temperature: float = None, max_tokens: int = None) -> Optional[str]:
        """Get cached response if available"""
        key = self._generate_key(provider, model, messages, temperature, max_tokens)
        
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['current_size'] -= entry.size_bytes
                    self.stats['misses'] += 1
                    
                    if self.logger:
                        self.logger.log_cache_event("expired", key, hit=False)
                    
                    return None
                
                # Update access stats
                entry.access()
                self.stats['hits'] += 1
                
                if self.logger:
                    self.logger.log_cache_event("hit", key, hit=True, 
                                               size_bytes=entry.size_bytes)
                
                # Decompress and return
                return self._decompress_value(entry.value)
            
            self.stats['misses'] += 1
            
            if self.logger:
                self.logger.log_cache_event("miss", key, hit=False)
            
            return None
    
    def set(self, provider: str, model: str, messages: List[Dict],
            response: str, temperature: float = None, max_tokens: int = None,
            ttl: int = None):
        """Cache a response"""
        key = self._generate_key(provider, model, messages, temperature, max_tokens)
        
        # Compress value
        compressed = self._compress_value(response)
        
        with self.cache_lock:
            # Check if we need to evict
            entry_size = len(compressed)
            self._evict_if_needed(entry_size)
            
            # Create and store entry
            entry = CacheEntry(key, compressed, ttl or self.default_ttl)
            self.cache[key] = entry
            self.stats['current_size'] += entry.size_bytes
            
            if self.logger:
                self.logger.log_cache_event("set", key, size_bytes=entry.size_bytes)
    
    def _evict_if_needed(self, new_size: int):
        """Evict entries if cache is full"""
        while self.stats['current_size'] + new_size > self.max_size_bytes:
            if not self.cache:
                break
            
            # LRU eviction
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].last_accessed)
            
            entry = self.cache[oldest_key]
            del self.cache[oldest_key]
            
            self.stats['current_size'] -= entry.size_bytes
            self.stats['evictions'] += 1
            
            if self.logger:
                self.logger.log_cache_event("evict", oldest_key, 
                                           size_bytes=entry.size_bytes)
    
    def _cleanup_worker(self):
        """Background thread to clean expired entries"""
        while True:
            time.sleep(60)  # Check every minute
            
            with self.cache_lock:
                expired_keys = []
                
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.cache[key]
                    del self.cache[key]
                    self.stats['current_size'] -= entry.size_bytes
                    
                    if self.logger:
                        self.logger.log_cache_event("cleanup", key)
    
    def clear(self):
        """Clear entire cache"""
        with self.cache_lock:
            self.cache.clear()
            self.stats['current_size'] = 0
            
            if self.logger:
                self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'current_size_mb': self.stats['current_size'] / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'entries': len(self.cache),
                'compression_savings_mb': self.stats['compression_savings'] / (1024 * 1024)
            }

# ====================
# PRODUCTION METRICS
# ====================

class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class Metric:
    """Single metric with history"""
    
    def __init__(self, name: str, metric_type: MetricType, 
                 window_size: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.values = deque(maxlen=window_size)
        self.timestamp = time.time()
        self.lock = threading.Lock()
    
    def add(self, value: float, timestamp: float = None):
        """Add value to metric"""
        with self.lock:
            ts = timestamp or time.time()
            self.values.append((ts, value))
            self.timestamp = ts
    
    def get_stats(self) -> Dict:
        """Get metric statistics"""
        with self.lock:
            if not self.values:
                return {'count': 0}
            
            values_only = [v[1] for v in self.values]
            
            if self.metric_type == MetricType.COUNTER:
                return {
                    'count': len(values_only),
                    'total': sum(values_only),
                    'rate': self._calculate_rate()
                }
            
            elif self.metric_type == MetricType.GAUGE:
                return {
                    'current': values_only[-1],
                    'min': min(values_only),
                    'max': max(values_only),
                    'avg': sum(values_only) / len(values_only)
                }
            
            elif self.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                sorted_values = sorted(values_only)
                count = len(sorted_values)
                
                return {
                    'count': count,
                    'min': sorted_values[0],
                    'max': sorted_values[-1],
                    'mean': sum(sorted_values) / count,
                    'p50': sorted_values[int(count * 0.5)],
                    'p95': sorted_values[int(count * 0.95)],
                    'p99': sorted_values[int(count * 0.99)]
                }
            
            return {}
    
    def _calculate_rate(self) -> float:
        """Calculate rate for counter metrics"""
        if len(self.values) < 2:
            return 0.0
        
        time_diff = self.values[-1][0] - self.values[0][0]
        if time_diff == 0:
            return 0.0
        
        return len(self.values) / time_diff

class ProductionMetrics:
    """Production metrics collection and reporting"""
    
    def __init__(self, export_interval: int = 60, logger: ProductionLogger = None):
        """
        Initialize metrics collector
        
        Args:
            export_interval: Interval in seconds to export metrics
        """
        self.metrics: Dict[str, Metric] = {}
        self.export_interval = export_interval
        self.logger = logger
        self.lock = threading.Lock()
        
        # Start export thread
        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        
        # Initialize standard metrics
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """Initialize standard metrics"""
        # API metrics
        self.create_metric("api_requests_total", MetricType.COUNTER)
        self.create_metric("api_errors_total", MetricType.COUNTER)
        self.create_metric("api_latency_ms", MetricType.TIMER)
        self.create_metric("api_tokens_input", MetricType.COUNTER)
        self.create_metric("api_tokens_output", MetricType.COUNTER)
        self.create_metric("api_cost_usd", MetricType.COUNTER)
        
        # Cache metrics
        self.create_metric("cache_hits", MetricType.COUNTER)
        self.create_metric("cache_misses", MetricType.COUNTER)
        self.create_metric("cache_size_bytes", MetricType.GAUGE)
        self.create_metric("cache_evictions", MetricType.COUNTER)
        
        # System metrics
        self.create_metric("active_sessions", MetricType.GAUGE)
        self.create_metric("memory_usage_mb", MetricType.GAUGE)
        self.create_metric("complexity_score", MetricType.HISTOGRAM)
        
        # Model routing metrics
        self.create_metric("model_switches", MetricType.COUNTER)
        self.create_metric("routing_decisions", MetricType.COUNTER)
    
    def create_metric(self, name: str, metric_type: MetricType) -> Metric:
        """Create or get a metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name, metric_type)
            return self.metrics[name]
    
    def increment(self, name: str, value: float = 1.0):
        """Increment a counter metric"""
        metric = self.create_metric(name, MetricType.COUNTER)
        metric.add(value)
    
    def gauge(self, name: str, value: float):
        """Set a gauge metric"""
        metric = self.create_metric(name, MetricType.GAUGE)
        metric.add(value)
    
    def histogram(self, name: str, value: float):
        """Add value to histogram"""
        metric = self.create_metric(name, MetricType.HISTOGRAM)
        metric.add(value)
    
    def timer(self, name: str, duration_ms: float):
        """Record timer duration"""
        metric = self.create_metric(name, MetricType.TIMER)
        metric.add(duration_ms)
    
    @contextmanager
    def time(self, name: str):
        """Context manager to time operations"""
        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        self.timer(name, duration_ms)
    
    def record_api_call(self, provider: str, model: str, success: bool,
                       latency_ms: float, input_tokens: int, output_tokens: int,
                       cost: float, complexity: int = None):
        """Record API call metrics"""
        # Base metrics
        self.increment("api_requests_total")
        self.timer("api_latency_ms", latency_ms)
        self.increment("api_tokens_input", input_tokens)
        self.increment("api_tokens_output", output_tokens)
        self.increment("api_cost_usd", cost)
        
        # Provider/model specific
        self.increment(f"api_requests_{provider}_total")
        self.increment(f"api_requests_{model}_total")
        self.timer(f"api_latency_{provider}_ms", latency_ms)
        
        if not success:
            self.increment("api_errors_total")
            self.increment(f"api_errors_{provider}_total")
        
        if complexity:
            self.histogram("complexity_score", complexity)
    
    def record_cache_event(self, event_type: str, size_bytes: int = None):
        """Record cache event"""
        if event_type == "hit":
            self.increment("cache_hits")
        elif event_type == "miss":
            self.increment("cache_misses")
        elif event_type == "eviction":
            self.increment("cache_evictions")
        
        if size_bytes:
            self.gauge("cache_size_bytes", size_bytes)
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get all metrics with statistics"""
        with self.lock:
            return {
                name: metric.get_stats() 
                for name, metric in self.metrics.items()
            }
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        all_metrics = self.get_all_metrics()
        
        # Calculate key metrics
        total_requests = all_metrics.get('api_requests_total', {}).get('total', 0)
        total_errors = all_metrics.get('api_errors_total', {}).get('total', 0)
        
        return {
            'total_requests': total_requests,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'avg_latency_ms': all_metrics.get('api_latency_ms', {}).get('mean', 0),
            'total_cost': all_metrics.get('api_cost_usd', {}).get('total', 0),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'total_tokens': (
                all_metrics.get('api_tokens_input', {}).get('total', 0) +
                all_metrics.get('api_tokens_output', {}).get('total', 0)
            )
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        all_metrics = self.get_all_metrics()
        hits = all_metrics.get('cache_hits', {}).get('total', 0)
        misses = all_metrics.get('cache_misses', {}).get('total', 0)
        total = hits + misses
        
        return hits / total if total > 0 else 0
    
    def _export_worker(self):
        """Background thread to export metrics"""
        while True:
            time.sleep(self.export_interval)
            
            try:
                # Export to logger
                if self.logger:
                    summary = self.get_summary()
                    self.logger.info("Metrics export", extra={
                        'metrics_summary': summary,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Could also export to Prometheus, CloudWatch, etc.
                self._export_to_monitoring()
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Metrics export error: {e}")
    
    def _export_to_monitoring(self):
        """Export metrics to monitoring system (Prometheus, CloudWatch, etc.)"""
        # This would integrate with your monitoring system
        # Example: push to Prometheus pushgateway, CloudWatch, etc.
        pass

# ====================
# PRODUCTION RATE LIMITER
# ====================

class RateLimitWindow:
    """Sliding window for rate limiting"""
    
    def __init__(self, window_seconds: int, max_requests: int):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def allow_request(self) -> Tuple[bool, int, float]:
        """
        Check if request is allowed
        
        Returns:
            (allowed, remaining_requests, reset_time)
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check if allowed
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                remaining = self.max_requests - len(self.requests)
                reset_time = self.requests[0] + self.window_seconds if self.requests else now
                return True, remaining, reset_time
            
            # Calculate when next request will be allowed
            reset_time = self.requests[0] + self.window_seconds
            return False, 0, reset_time

class ProductionRateLimiter:
    """Production rate limiter with per-provider and per-model limits"""
    
    def __init__(self, config: Dict = None, logger: ProductionLogger = None,
                 db: ProductionDB = None):
        """
        Initialize rate limiter
        
        Args:
            config: Rate limit configuration
        """
        self.logger = logger
        self.db = db
        
        # Default rate limits (requests per minute)
        default_config = {
            'global': {'window': 60, 'max_requests': 100},
            'providers': {
                'openai': {'window': 60, 'max_requests': 60},
                'anthropic': {'window': 60, 'max_requests': 50},
                'groq': {'window': 60, 'max_requests': 30},
                'google': {'window': 60, 'max_requests': 60},
                'deepseek': {'window': 60, 'max_requests': 100},
                'ollama': {'window': 60, 'max_requests': 1000}
            },
            'models': {
                'gpt-4o': {'window': 60, 'max_requests': 20},
                'claude-3-opus': {'window': 60, 'max_requests': 15}
            }
        }
        
        self.config = config or default_config
        self.windows: Dict[str, RateLimitWindow] = {}
        self.lock = threading.Lock()
        
        # Initialize windows
        self._init_windows()
        
        # Backoff strategy
        self.backoff_times: Dict[str, float] = {}
        self.max_backoff = 60  # Maximum backoff in seconds
    
    def _init_windows(self):
        """Initialize rate limit windows"""
        # Global window
        global_config = self.config.get('global', {})
        self.windows['global'] = RateLimitWindow(
            global_config.get('window', 60),
            global_config.get('max_requests', 100)
        )
        
        # Provider windows
        for provider, limits in self.config.get('providers', {}).items():
            self.windows[f'provider:{provider}'] = RateLimitWindow(
                limits.get('window', 60),
                limits.get('max_requests', 60)
            )
        
        # Model windows
        for model, limits in self.config.get('models', {}).items():
            self.windows[f'model:{model}'] = RateLimitWindow(
                limits.get('window', 60),
                limits.get('max_requests', 60)
            )
    
    def check_rate_limit(self, provider: str, model: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limits
        
        Returns:
            (allowed, limit_info)
        """
        with self.lock:
            # Check global limit
            global_allowed, global_remaining, global_reset = self.windows['global'].allow_request()
            
            if not global_allowed:
                self._handle_limit_exceeded('global', global_remaining, global_reset)
                return False, {
                    'limited_by': 'global',
                    'remaining': global_remaining,
                    'reset_time': global_reset,
                    'retry_after': global_reset - time.time()
                }
            
            # Check provider limit
            provider_key = f'provider:{provider}'
            if provider_key in self.windows:
                provider_allowed, provider_remaining, provider_reset = \
                    self.windows[provider_key].allow_request()
                
                if not provider_allowed:
                    self._handle_limit_exceeded(provider, provider_remaining, provider_reset)
                    return False, {
                        'limited_by': provider,
                        'remaining': provider_remaining,
                        'reset_time': provider_reset,
                        'retry_after': provider_reset - time.time()
                    }
            
            # Check model limit
            model_key = f'model:{model}'
            if model_key in self.windows:
                model_allowed, model_remaining, model_reset = \
                    self.windows[model_key].allow_request()
                
                if not model_allowed:
                    self._handle_limit_exceeded(model, model_remaining, model_reset)
                    return False, {
                        'limited_by': model,
                        'remaining': model_remaining,
                        'reset_time': model_reset,
                        'retry_after': model_reset - time.time()
                    }
            
            # All limits passed
            return True, {
                'allowed': True,
                'global_remaining': global_remaining
            }
    
    def _handle_limit_exceeded(self, limited_by: str, remaining: int, reset_time: float):
        """Handle rate limit exceeded"""
        # Log event
        if self.logger:
            self.logger.log_rate_limit(limited_by, remaining, 
                                      datetime.fromtimestamp(reset_time).isoformat())
        
        # Record in database
        if self.db:
            self.db.log_rate_limit_event(
                provider=limited_by if limited_by != 'global' else 'system',
                limit_type='rate_limit',
                remaining=remaining,
                reset_time=datetime.fromtimestamp(reset_time),
                action='blocked'
            )
    
    def wait_if_limited(self, provider: str, model: str, max_wait: float = 5.0) -> bool:
        """
        Wait if rate limited
        
        Returns:
            True if request can proceed, False if wait exceeded max_wait
        """
        allowed, info = self.check_rate_limit(provider, model)
        
        if allowed:
            return True
        
        retry_after = info.get('retry_after', 0)
        
        if retry_after > max_wait:
            return False
        
        if self.logger:
            self.logger.info(f"Rate limited, waiting {retry_after:.2f}s")
        
        time.sleep(retry_after)
        return True
    
    def apply_backoff(self, provider: str, error: Exception = None):
        """Apply exponential backoff for a provider"""
        key = f'backoff:{provider}'
        
        # Calculate next backoff time
        current_backoff = self.backoff_times.get(key, 0)
        
        if current_backoff == 0:
            next_backoff = 1  # Start with 1 second
        else:
            next_backoff = min(current_backoff * 2, self.max_backoff)
        
        self.backoff_times[key] = next_backoff
        
        if self.logger:
            self.logger.warning(f"Applying backoff for {provider}: {next_backoff}s", 
                              extra={'error': str(error) if error else None})
        
        time.sleep(next_backoff)
    
    def reset_backoff(self, provider: str):
        """Reset backoff for a provider after successful request"""
        key = f'backoff:{provider}'
        if key in self.backoff_times:
            del self.backoff_times[key]
    
    def get_limits_info(self) -> Dict:
        """Get current rate limit information"""
        info = {}
        
        for key, window in self.windows.items():
            with window.lock:
                now = time.time()
                cutoff = now - window.window_seconds
                
                # Clean old requests
                while window.requests and window.requests[0] < cutoff:
                    window.requests.popleft()
                
                current_usage = len(window.requests)
                
                info[key] = {
                    'current': current_usage,
                    'max': window.max_requests,
                    'window_seconds': window.window_seconds,
                    'usage_percentage': (current_usage / window.max_requests) * 100
                }
        
        return info
    
    def update_limits(self, provider: str = None, model: str = None, 
                     max_requests: int = None, window: int = None):
        """Dynamically update rate limits"""
        with self.lock:
            if provider:
                key = f'provider:{provider}'
                if key in self.windows:
                    if max_requests:
                        self.windows[key].max_requests = max_requests
                    if window:
                        self.windows[key].window_seconds = window
            
            if model:
                key = f'model:{model}'
                if key in self.windows:
                    if max_requests:
                        self.windows[key].max_requests = max_requests
                    if window:
                        self.windows[key].window_seconds = window
            
            if self.logger:
                self.logger.info("Rate limits updated", extra={
                    'provider': provider,
                    'model': model,
                    'max_requests': max_requests,
                    'window': window
                })

# ====================
# CONFIGURAZIONE BASE (original)
# ====================

@dataclass
class ModelConfig:
    """Configurazione completa modello"""
    provider: str
    model_id: str
    display_name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    quality_score: int  # 1-10 basato su dimensione e capacità
    speed: str  # slow, medium, fast, ultra_fast
    context_window: int
    best_for: List[str]
    requires_key: bool
    size_gb: float = 0.0  # Dimensione stimata in GB

# [Original MODEL_CATALOG remains the same]
MODEL_CATALOG = {
    # OPENAI
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4 Optimized",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=10,
        speed="medium",
        context_window=128000,
        best_for=["complex_reasoning", "coding", "analysis"],
        requires_key=True,
        size_gb=100  # Stima
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4 Mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=8,
        speed="fast",
        context_window=128000,
        best_for=["general", "cost_effective"],
        requires_key=True,
        size_gb=20  # Stima
    ),
    "gpt-3.5-turbo": ModelConfig(
        provider="openai",
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        quality_score=7,
        speed="fast",
        context_window=16385,
        best_for=["simple_tasks", "fast_responses"],
        requires_key=True,
        size_gb=10  # Stima
    ),
    
    # ANTHROPIC
    "claude-3-opus": ModelConfig(
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        quality_score=10,
        speed="slow",
        context_window=200000,
        best_for=["complex_analysis", "creative_writing", "research"],
        requires_key=True,
        size_gb=150  # Stima
    ),
    "claude-3-sonnet": ModelConfig(
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        quality_score=9,
        speed="medium",
        context_window=200000,
        best_for=["balanced", "coding", "reasoning"],
        requires_key=True,
        size_gb=70  # Stima
    ),
    "claude-3-haiku": ModelConfig(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        quality_score=7,
        speed="ultra_fast",
        context_window=200000,
        best_for=["simple_tasks", "fast_responses"],
        requires_key=True,
        size_gb=10  # Stima
    ),
    
    # GROQ
    "llama-3.3-70b": ModelConfig(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B (Groq)",
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        quality_score=9,
        speed="ultra_fast",
        context_window=128000,
        best_for=["fast_quality", "general"],
        requires_key=True,
        size_gb=70
    ),
    "mixtral-8x7b": ModelConfig(
        provider="groq",
        model_id="mixtral-8x7b-32768",
        display_name="Mixtral 8x7B (Groq)",
        cost_per_1k_input=0.00024,
        cost_per_1k_output=0.00024,
        quality_score=8,
        speed="ultra_fast",
        context_window=32768,
        best_for=["fast_responses", "general"],
        requires_key=True,
        size_gb=45
    ),
    
    # GOOGLE
    "gemini-2.0-flash": ModelConfig(
        provider="google",
        model_id="gemini-2.0-flash-exp",
        display_name="Gemini 2.0 Flash",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        quality_score=8,
        speed="ultra_fast",
        context_window=1048576,
        best_for=["long_context", "multimodal"],
        requires_key=True,
        size_gb=20  # Stima
    ),
    
    # DEEPSEEK
    "deepseek-chat": ModelConfig(
        provider="deepseek",
        model_id="deepseek-chat",
        display_name="DeepSeek Chat",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        quality_score=8,
        speed="fast",
        context_window=64000,
        best_for=["coding", "math", "chinese"],
        requires_key=True,
        size_gb=20  # Stima
    )
}

# [All original provider implementations remain the same]
# ====================
# PROVIDER BASE CLASSES
# ====================

class LLMProvider(ABC):
    """Classe base per tutti i provider LLM"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Controlla se il provider è disponibile"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        """Invia messaggi e ricevi risposta"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """Lista modelli disponibili"""
        pass

# ====================
# PROVIDER IMPLEMENTATIONS (original)
# ====================

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        data = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 2000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        if system:
            data["system"] = system
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]

class GroqProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

class GoogleProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        contents = []
        for msg in messages:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
        
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2000)
            }
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["gemini-2.0-flash-exp", "gemini-1.5-pro"]

class DeepSeekProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["deepseek-chat"]

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.client = ollama if OLLAMA_AVAILABLE else None
    
    def is_available(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            self.client.list()
            return True
        except:
            return False
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        if not self.client:
            raise Exception("Ollama not available")
            
        response = self.client.chat(
            model=model,
            messages=messages,
            options={
                'temperature': kwargs.get("temperature", 0.7),
                'num_predict': kwargs.get("max_tokens", 2000)
            }
        )
        return response['message']['content']
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        
        try:
            models = self.client.list()
            return [m['model'] for m in models['models']]
        except:
            return []

# [All original context, complexity evaluator, and auto-discovery code remains the same]
# ====================
# CONTEXT MANAGER (original)
# ====================

class ConversationContext:
    """Gestisce il contesto unificato della conversazione"""
    
    def __init__(self, max_tokens: int = 8000):
        self.messages: List[Dict] = []
        self.max_tokens = max_tokens
        self.model_history: List[str] = []
        self.token_count = 0
    
    def add_message(self, role: str, content: str, model_used: str = None):
        """Aggiunge messaggio mantenendo il contesto"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used
        }
        
        self.messages.append(msg)
        if model_used:
            self.model_history.append(model_used)
        
        # Stima tokens
        self.token_count += len(content.split()) * 1.3
        
        # Trim se necessario
        self._trim_context()
    
    def _trim_context(self):
        """Riduce il contesto se troppo lungo"""
        if self.token_count > self.max_tokens:
            system = [m for m in self.messages if m["role"] == "system"]
            recent = self.messages[-10:]
            
            self.messages = system + recent
            self.token_count = sum(len(m["content"].split()) * 1.3 for m in self.messages)
    
    def get_messages_for_api(self) -> List[Dict]:
        """Ritorna messaggi nel formato per API"""
        return [
            {"role": m["role"], "content": m["content"]} 
            for m in self.messages
        ]
    
    def get_summary(self) -> str:
        """Genera summary del contesto"""
        return f"Messages: {len(self.messages)}, Models used: {len(set(self.model_history))}, Tokens: ~{int(self.token_count)}"

# [All original auto-discovery and complexity evaluator code remains the same]
# ====================
# AUTO-DISCOVERY SYSTEM (original)
# ====================

class ModelAutoDiscovery:
    """Sistema di auto-discovery e benchmarking per modelli Ollama"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_suite = self._create_test_suite()
        self.benchmark_cache_file = Path.home() / ".llm-use" / "benchmarks.json"
        self.benchmark_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.benchmarks = self._load_benchmarks()
    
    def _create_test_suite(self) -> Dict[str, Dict]:
        """Suite di test per valutare capacità modelli"""
        return {
            "simple_greeting": {
                "prompt": "Hello! How are you?",
                "category": "simple",
                "expected_quality": "coherent_response",
                "max_tokens": 50
            },
            "basic_math": {
                "prompt": "What is 15 + 27?",
                "category": "math",
                "expected_quality": "correct_answer",
                "correct_answer": "42",
                "max_tokens": 20
            },
            "reasoning": {
                "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
                "category": "reasoning",
                "expected_quality": "logical_explanation",
                "max_tokens": 150
            },
            "coding": {
                "prompt": "Write a Python function to reverse a string. Just the function, no explanation.",
                "category": "coding",
                "expected_quality": "valid_code",
                "max_tokens": 100
            },
            "creativity": {
                "prompt": "Complete this story in one sentence: The robot looked at the sunset and suddenly...",
                "category": "creative",
                "expected_quality": "creative_continuation",
                "max_tokens": 50
            },
            "complex_analysis": {
                "prompt": "What are the main differences between supervised and unsupervised learning in 2 sentences?",
                "category": "technical",
                "expected_quality": "accurate_technical",
                "max_tokens": 100
            }
        }
    
    def _load_benchmarks(self) -> Dict:
        """Carica benchmark salvati"""
        if self.benchmark_cache_file.exists():
            try:
                with open(self.benchmark_cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_benchmarks(self):
        """Salva benchmark su file"""
        with open(self.benchmark_cache_file, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)
    
    def discover_and_benchmark_ollama_models(self, force_benchmark: bool = False) -> Dict[str, ModelConfig]:
        """Scopre e testa TUTTI i modelli Ollama installati"""
        discovered_models = {}
        
        if not OLLAMA_AVAILABLE:
            if self.verbose:
                print("⚠️ Ollama not available for discovery")
            return discovered_models
        
        try:
            # Ottieni lista modelli installati
            ollama_models = ollama.list()
            
            if self.verbose:
                print(f"\n🔍 Discovering Ollama models...")
                print(f"   Found {len(ollama_models['models'])} models installed\n")
            
            for model_info in ollama_models['models']:
                model_name = model_info['model']
                model_size_bytes = model_info.get('size', 0)
                model_size_gb = model_size_bytes / (1024**3) if model_size_bytes > 0 else 0
                
                model_base = model_name.split(':')[0]
                
                # Genera ID univoco
                model_id = f"ollama_{model_base}_{model_name.replace(':', '_')}"
                
                # Controlla cache benchmark
                use_cache = False
                if not force_benchmark and model_id in self.benchmarks:
                    benchmark_age_hours = (time.time() - self.benchmarks[model_id].get('timestamp', 0)) / 3600
                    if benchmark_age_hours < 24:
                        use_cache = True
                        if self.verbose:
                            print(f"✓ {model_name}: Using cached benchmark (age: {benchmark_age_hours:.1f}h)")
                
                if use_cache:
                    discovered_models[model_id] = self._create_config_from_benchmark(
                        model_name, 
                        self.benchmarks[model_id],
                        model_size_gb
                    )
                    continue
                
                # Esegui benchmark REALE
                if self.verbose:
                    print(f"\n🧪 Testing {model_name} ({model_size_gb:.1f}GB)")
                    print(f"   Running test suite:")
                
                benchmark_results = self._benchmark_model(model_name)
                benchmark_results['size_gb'] = model_size_gb
                
                # Mostra risultati in tempo reale
                if self.verbose:
                    print(f"\n   📊 Results:")
                    print(f"     Overall score: {benchmark_results['overall_score']:.2f}")
                    print(f"     Avg response time: {benchmark_results.get('avg_response_time', 0):.1f}s")
                    print(f"     Capabilities: {', '.join(benchmark_results['capabilities']) or 'none'}")
                    if benchmark_results.get('errors'):
                        print(f"     ⚠️  Errors: {len(benchmark_results['errors'])}")
                
                # Salva risultati
                self.benchmarks[model_id] = benchmark_results
                self._save_benchmarks()
                
                # Crea configurazione basata su TEST REALI
                config = self._create_config_from_benchmark(model_name, benchmark_results, model_size_gb)
                discovered_models[model_id] = config
                
                if self.verbose:
                    self._print_benchmark_summary(model_name, config)
                    print()
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error discovering Ollama models: {e}")
                import traceback
                traceback.print_exc()
        
        return discovered_models
    
    def _benchmark_model(self, model_name: str) -> Dict:
        """Esegue benchmark completo su un modello"""
        results = {
            "timestamp": time.time(),
            "model_name": model_name,
            "test_results": {},
            "timings": [],
            "capabilities": [],
            "errors": [],
            "overall_score": 0
        }
        
        total_score = 0
        test_count = 0
        
        for test_name, test_config in self.test_suite.items():
            try:
                if self.verbose:
                    print(f"     Running {test_name}...", end='', flush=True)
                
                start_time = time.time()
                
                # Esegui test e ASPETTA la risposta completa
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": test_config["prompt"]}],
                    stream=False,
                    options={
                        'temperature': 0.3,
                    }
                )
                
                elapsed = time.time() - start_time
                results["timings"].append(elapsed)
                
                # Ottieni risposta completa
                response_text = response['message']['content']
                
                if self.verbose:
                    print(f" ✓ {elapsed:.1f}s")
                    if elapsed > 10:
                        print(f"       (Long response time suggests large model)")
                
                # Valuta risposta
                score = self._evaluate_response(response_text, test_config)
                
                results["test_results"][test_name] = {
                    "score": score,
                    "time": elapsed,
                    "category": test_config["category"],
                    "response": response_text[:200],
                    "response_length": len(response_text)
                }
                
                # Aggiungi capability se test passato
                if score >= 0.7:
                    results["capabilities"].append(test_config["category"])
                
                total_score += score
                test_count += 1
                
            except Exception as e:
                error_msg = f"{test_name}: {str(e)}"
                results["errors"].append(error_msg)
                if self.verbose:
                    print(f" ❌ {error_msg}")
        
        # Calcola score complessivo
        if test_count > 0:
            results["overall_score"] = total_score / test_count
            results["avg_response_time"] = sum(results["timings"]) / len(results["timings"]) if results["timings"] else 1.0
            
            if self.verbose:
                print(f"     Average response time: {results['avg_response_time']:.1f}s")
        
        return results
    
    def _evaluate_response(self, response: str, test_config: Dict) -> float:
        """Valuta qualità della risposta (0-1)"""
        score = 0.0
        
        if not response or len(response.strip()) < 5:
            return 0.0
        
        category = test_config["category"]
        
        if category == "simple":
            score = 0.8 if len(response) > 10 else 0.4
        
        elif category == "math":
            if "correct_answer" in test_config:
                score = 1.0 if test_config["correct_answer"] in response else 0.0
            else:
                score = 0.5
        
        elif category == "reasoning":
            logic_words = ["because", "therefore", "however", "if", "then", "conclude"]
            matches = sum(1 for word in logic_words if word.lower() in response.lower())
            score = min(1.0, matches * 0.25)
        
        elif category == "coding":
            code_indicators = ["def ", "return", ":", "(", ")"]
            matches = sum(1 for indicator in code_indicators if indicator in response)
            score = min(1.0, matches * 0.2)
        
        elif category == "creative":
            score = 0.8 if len(response) > 20 else 0.4
        
        elif category == "technical":
            tech_terms = ["supervised", "unsupervised", "learning", "data", "model", "training"]
            matches = sum(1 for term in tech_terms if term.lower() in response.lower())
            score = min(1.0, matches * 0.2)
        
        else:
            score = 0.5
        
        return score
    
    def _calculate_quality_score(self, model_name: str, benchmark_results: Dict, size_gb: float) -> int:
        """
        Calcola quality score basato su PERFORMANCE REALE
        """
        test_score = benchmark_results.get("overall_score", 0.5)
        avg_time = benchmark_results.get("avg_response_time", 1.0)
        
        if avg_time > 15:
            size_factor = 9
        elif avg_time > 10:
            size_factor = 8
        elif avg_time > 5:
            size_factor = 7
        elif avg_time > 2:
            size_factor = 6
        elif avg_time > 1:
            size_factor = 5
        else:
            size_factor = 4
        
        capabilities = benchmark_results.get("capabilities", [])
        capability_bonus = len(capabilities) * 0.3
        
        error_penalty = len(benchmark_results.get("errors", [])) * 0.5
        
        final_score = (test_score * 5) + (size_factor * 0.5) + capability_bonus - error_penalty
        
        return min(10, max(1, round(final_score)))
    
    def _calculate_speed(self, avg_response_time: float) -> str:
        """Determina velocità basata SOLO su tempo risposta"""
        if avg_response_time < 0.5:
            return "ultra_fast"
        elif avg_response_time < 1.5:
            return "fast"
        elif avg_response_time < 5.0:
            return "medium"
        else:
            return "slow"
    
    def _create_config_from_benchmark(self, model_name: str, benchmark: Dict, size_gb: float) -> ModelConfig:
        """Crea ModelConfig basato sui risultati del benchmark"""
        
        model_base = model_name.split(':')[0]
        
        quality_score = self._calculate_quality_score(model_name, benchmark, size_gb)
        
        avg_time = benchmark.get("avg_response_time", 1.0)
        speed = self._calculate_speed(avg_time)
        
        capabilities = benchmark.get("capabilities", [])
        best_for = ["local", "private"]
        
        if "coding" in capabilities:
            best_for.append("coding")
        if "reasoning" in capabilities:
            best_for.append("reasoning")
        if "technical" in capabilities:
            best_for.append("technical")
        if "creative" in capabilities:
            best_for.append("creative")
        if "math" in capabilities:
            best_for.append("math")
        
        if len(best_for) == 2:
            best_for.append("general")
        
        context_window = self._estimate_context_window(model_base)
        display_name = self._generate_display_name(model_base, size_gb, quality_score)
        
        return ModelConfig(
            provider="ollama",
            model_id=model_name,
            display_name=display_name,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            quality_score=quality_score,
            speed=speed,
            context_window=context_window,
            best_for=best_for,
            requires_key=False,
            size_gb=size_gb
        )
    
    def _estimate_context_window(self, model_base: str) -> int:
        """Stima context window basata su modello"""
        known_contexts = {
            "llama3": 128000,
            "llama3.1": 128000,
            "llama3.2": 128000,
            "mistral": 32768,
            "qwen": 32768,
            "qwen2": 32768,
            "qwen2.5": 32768,
            "deepseek": 32768,
            "deepseek-r1": 32768,
            "gemma": 8192,
            "gemma2": 8192,
            "gemma3": 8192,
            "gpt": 32768,
            "mixtral": 32768
        }
        
        for key, context in known_contexts.items():
            if key in model_base.lower():
                return context
        
        return 8192
    
    def _generate_display_name(self, model_base: str, size_gb: float, quality_score: int) -> str:
        """Genera nome user-friendly"""
        name_parts = model_base.split('-')
        formatted_base = ' '.join(part.capitalize() for part in name_parts)
        
        if size_gb > 0:
            formatted_base += f" {size_gb:.0f}GB"
        
        if quality_score >= 9:
            quality_tag = "⭐"
        elif quality_score >= 7:
            quality_tag = "✓"
        else:
            quality_tag = ""
        
        return f"{formatted_base} (Local) {quality_tag}".strip()
    
    def _print_benchmark_summary(self, model_name: str, config: ModelConfig):
        """Stampa summary del benchmark"""
        print(f"   ✓ {config.display_name}")
        print(f"     Quality: {config.quality_score}/10 | Speed: {config.speed}")
        print(f"     Best for: {', '.join(config.best_for[:3])}")

# ====================
# COMPLEXITY EVALUATORS (original)
# ====================

class ComplexityEvaluator(ABC):
    """Base class per valutazione complessità"""
    
    @abstractmethod
    def evaluate(self, user_input: str) -> int:
        pass

class MistralComplexityEvaluator(ComplexityEvaluator):
    """Valuta complessità usando Mistral locale"""
    
    def __init__(self, ollama_provider):
        self.provider = ollama_provider
    
    def evaluate(self, user_input: str) -> int:
        prompt = f"""You are an expert at analyzing task complexity for LLM routing. Your job is to rate tasks from 1-10 to help select the right model.

<thinking_framework>
Analyze the input across these dimensions:
1. Cognitive Load: How much reasoning is required?
2. Technical Depth: How specialized is the knowledge needed?
3. Context Understanding: How much context interpretation is needed?
4. Problem Solving: How complex is the problem to solve?
5. Response Sophistication: How nuanced must the response be?
</thinking_framework>

<complexity_scale>
1-2: Trivial (greetings, basic arithmetic, yes/no questions)
3-4: Simple (factual questions, basic explanations, simple translations)
5-6: Moderate (detailed explanations, basic coding, standard analysis)
7-8: Complex (debugging, advanced reasoning, system design, error analysis)
9-10: Expert (research problems, architecture design, cutting-edge topics)
</complexity_scale>

<key_indicators>
Look for these patterns to guide your rating:
• Length and structure of the input
• Technical terminology and jargon
• Multiple interconnected concepts
• Need for creative or analytical thinking
• Presence of code, errors, or system outputs
• Requirement for specialized domain knowledge
</key_indicators>

<task_to_evaluate>
{user_input[:500]}{'...' if len(user_input) > 500 else ''}
</task_to_evaluate>

Analyze this task and determine its complexity level.
Think about what kind of model capabilities would be needed.
Consider the depth of understanding and reasoning required.

Output ONLY a single number from 1 to 10:"""

        try:
            response = self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model="mistral",
                temperature=0.1,
                max_tokens=50
            )
            
            numbers = [int(char) for char in response if char.isdigit()]
            if numbers:
                return min(10, max(1, numbers[-1]))
            return 5
            
        except:
            return 5

class HeuristicComplexityEvaluator(ComplexityEvaluator):
    """Advanced heuristic complexity evaluation with multi-dimensional analysis"""
    
    def __init__(self):
        self.patterns = {
            'error_indicators': {
                'patterns': [
                    r'error\s*:', r'exception', r'invalid\s+format', r'missing\s+[\"\']?\w+[\"\']?',
                    r'failed\s+to', r'cannot\s+\w+', r'unable\s+to', r'unexpected\s+\w+',
                    r'traceback', r'stack\s+trace', r'assertion\s+failed', r'undefined',
                    r'null\s+pointer', r'segmentation\s+fault', r'syntax\s+error'
                ],
                'weight': 4.0,
                'min_complexity': 7
            },
            
            'ai_output_patterns': {
                'patterns': [
                    r'thought\s*:', r'action\s*:', r'observation\s*:', r'reflection\s*:',
                    r'step\s+\d+\s*:', r'```[\w]*\n', r'here\'s\s+the\s+corrected',
                    r'apologies\s+for', r'let\s+me\s+clarify', r'to\s+summarize',
                    r'invalid\s+format.*missing', r'confused\s+about'
                ],
                'weight': 3.5,
                'min_complexity': 6
            },
            
            'code_patterns': {
                'patterns': [
                    r'def\s+\w+\s*KATEX_INLINE_OPEN', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
                    r'if\s+__name__\s*==', r'async\s+def', r'return\s+\w+', r'yield\s+',
                    r'lambda\s+\w+:', r'@\w+', r'try\s*:', r'except\s+\w+',
                    r'\w+\s*=\s*\w+\s*KATEX_INLINE_OPEN', r'for\s+\w+\s+in\s+', r'while\s+\w+\s*:'
                ],
                'weight': 2.5,
                'min_complexity': 5
            },
            
            'architecture_patterns': {
                'patterns': [
                    r'microservice', r'distributed\s+system', r'load\s+balanc', r'scalab',
                    r'kubernetes', r'docker', r'orchestrat', r'message\s+queue', r'event\s+driven',
                    r'design\s+pattern', r'architect', r'infrastructure', r'deployment',
                    r'high\s+availability', r'fault\s+toleran', r'consensus', r'replication'
                ],
                'weight': 4.5,
                'min_complexity': 8
            },
            
            'algorithm_patterns': {
                'patterns': [
                    r'algorithm', r'time\s+complexity', r'space\s+complexity', r'big\s*-?\s*o',
                    r'dynamic\s+programming', r'graph\s+traversal', r'tree\s+\w+', r'sort\s+\w+',
                    r'binary\s+search', r'hash\s+table', r'recursion', r'optimization',
                    r'np\s*-?\s*complete', r'heuristic', r'greedy\s+approach'
                ],
                'weight': 3.8,
                'min_complexity': 7
            },
            
            'simple_patterns': {
                'patterns': [
                    r'^hi\s*[!?]?$', r'^hello\s*[!?]?$', r'^hey\s*[!?]?$', r'^thanks?\s*[!?]?$',
                    r'^\d+\s*[\+\-\*/]\s*\d+\s*[?]?$', r'^what\s+is\s+\d+', r'^yes\s*[?]?$',
                    r'^no\s*[?]?$', r'^ok\s*[?]?$', r'^good\s*[?]?$'
                ],
                'weight': -3.0,
                'max_complexity': 2
            }
        }
        
        self.linguistic_features = {
            'subordinate_clauses': [r'\b(which|that|who|whom|whose|where|when|while|although|because|since|if|unless)\b'],
            'technical_terms': [r'\b(api|sdk|framework|library|protocol|interface|implementation|abstraction|encapsulation)\b'],
            'quantifiers': [r'\b(all|some|many|few|several|various|multiple|numerous)\b'],
            'modal_verbs': [r'\b(should|could|would|might|must|shall|may|can|will)\b'],
            'comparison': [r'\b(more|less|better|worse|faster|slower|higher|lower|greater|smaller)\s+than\b']
        }
    
    def evaluate(self, user_input: str) -> int:
        """Multi-dimensional complexity evaluation"""
        
        input_lower = user_input.lower()
        word_count = len(user_input.split())
        line_count = len(user_input.strip().split('\n'))
        char_count = len(user_input)
        
        base_complexity = self._calculate_base_complexity(word_count, line_count, char_count)
        
        pattern_score = 0
        max_min_complexity = 1
        min_max_complexity = 10
        
        for category, config in self.patterns.items():
            matches = sum(1 for pattern in config['patterns'] 
                         if re.search(pattern, input_lower, re.IGNORECASE))
            
            if matches > 0:
                pattern_score += config['weight'] * (1 - (1 / (1 + matches)))
                
                if 'min_complexity' in config:
                    max_min_complexity = max(max_min_complexity, config['min_complexity'])
                if 'max_complexity' in config:
                    min_max_complexity = min(min_max_complexity, config['max_complexity'])
        
        linguistic_score = self._analyze_linguistic_complexity(input_lower)
        structural_score = self._analyze_structural_complexity(user_input)
        density_score = self._calculate_information_density(user_input)
        
        final_complexity = (
            base_complexity * 0.20 +
            pattern_score * 0.35 +
            linguistic_score * 0.20 +
            structural_score * 0.15 +
            density_score * 0.10
        )
        
        final_complexity = max(final_complexity, max_min_complexity)
        final_complexity = min(final_complexity, min_max_complexity)
        
        return min(10, max(1, round(final_complexity)))
    
    def _calculate_base_complexity(self, words: int, lines: int, chars: int) -> float:
        """Sophisticated base complexity calculation"""
        word_score = 3 + (words ** 0.7) / 10
        line_bonus = min(2, (lines - 1) * 0.3) if lines > 1 else 0
        avg_word_length = chars / max(words, 1)
        density_factor = min(1.5, avg_word_length / 6)
        
        return word_score + line_bonus * density_factor
    
    def _analyze_linguistic_complexity(self, text: str) -> float:
        """Analyze linguistic features for complexity"""
        score = 0
        
        for feature, patterns in self.linguistic_features.items():
            matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                         for pattern in patterns)
            
            if feature == 'subordinate_clauses':
                score += matches * 0.8
            elif feature == 'technical_terms':
                score += matches * 0.6
            elif feature == 'quantifiers':
                score += matches * 0.3
            elif feature == 'modal_verbs':
                score += matches * 0.4
            elif feature == 'comparison':
                score += matches * 0.5
        
        return min(5, score)
    
    def _analyze_structural_complexity(self, text: str) -> float:
        """Analyze structural patterns"""
        score = 0
        
        if any(pattern in text for pattern in ['```', '{', '[', '<', '|']):
            score += 1.5
        
        if re.search(r'^\s*[\d\-\*\•]\s+', text, re.MULTILINE):
            score += 1.0
        
        if text.count('"') >= 2 or text.count("'") >= 2:
            score += 0.5
        
        if re.search(r'[A-Z]{2,}', text):
            score += 0.5
        
        nesting_depth = self._calculate_nesting_depth(text)
        score += min(2, nesting_depth * 0.5)
        
        return score
    
    def _calculate_nesting_depth(self, text: str) -> int:
        """Calculate maximum nesting depth of brackets/parentheses"""
        max_depth = 0
        current_depth = 0
        
        for char in text:
            if char in '([{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')]}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density of the text"""
        words = text.split()
        if not words:
            return 0
        
        unique_ratio = len(set(words)) / len(words)
        avg_length = sum(len(w) for w in words) / len(words)
        punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}')
        punct_density = punct_count / len(text) if text else 0
        
        density = (unique_ratio * 2 + avg_length / 5 + punct_density * 10)
        
        return min(3, density)

# ====================
# LLM-USE CORE WITH PRODUCTION FEATURES
# ====================

class LlmUse:
    """Sistema di routing universale con context preservation e production features"""
    
    def __init__(self, verbose: bool = True, enable_production: bool = True):
        self.verbose = verbose
        self.enable_production = enable_production
        
        # Initialize production components
        if enable_production:
            self.logger = ProductionLogger()
            self.db = ProductionDB()
            self.cache = ProductionCache(logger=self.logger)
            self.metrics = ProductionMetrics(logger=self.logger)
            self.rate_limiter = ProductionRateLimiter(logger=self.logger, db=self.db)
            
            # Create session
            self.session_id = self.db.create_session()
            self.logger.info(f"Production session created: {self.session_id}")
        else:
            self.logger = None
            self.db = None
            self.cache = None
            self.metrics = None
            self.rate_limiter = None
            self.session_id = None
        
        # Original initialization
        self.providers = self._initialize_providers()
        self.available_models = self._detect_available_models()
        self.context = ConversationContext()
        self.current_model = None
        self.stats = {
            "total_messages": 0,
            "total_cost": 0.0,
            "models_used": {},
            "provider_distribution": {}
        }
        
        self.router = self._setup_router()
        
        if verbose:
            self._print_initialization()
    
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Inizializza tutti i provider"""
        return {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "groq": GroqProvider(),
            "google": GoogleProvider(),
            "deepseek": DeepSeekProvider(),
            "ollama": OllamaProvider()
        }
    
    def _detect_available_models(self) -> Dict[str, ModelConfig]:
        """Rileva quali modelli sono realmente disponibili"""
        available = {}
        
        for model_name, config in MODEL_CATALOG.items():
            provider = self.providers.get(config.provider)
            
            if provider and provider.is_available():
                available[model_name] = config
        
        return available
    
    def _setup_router(self):
        """Setup del router per valutazione complessità"""
        if self.providers["ollama"].is_available():
            models = self.providers["ollama"].list_models()
            for model in models:
                if any(name in model.lower() for name in ["mistral", "gemma", "phi"]):
                    if self.verbose:
                        print(f"   Using {model} for complexity evaluation")
                    return MistralComplexityEvaluator(self.providers["ollama"])
        
        if self.verbose:
            print("   Using heuristic complexity evaluator")
        return HeuristicComplexityEvaluator()
    
    def _print_initialization(self):
        """Stampa stato inizializzazione"""
        print("\n" + "="*70)
        print("🌐 LLM-USE - Universal LLM Router")
        if self.enable_production:
            print("   🚀 Production Mode: ENABLED")
            print(f"   📊 Session ID: {self.session_id}")
        print("="*70)
        
        available_providers = [name for name, p in self.providers.items() if p.is_available()]
        print(f"\n✅ Active providers: {', '.join(available_providers)}")
        
        print(f"\n📦 Available models ({len(self.available_models)} total):")
        
        by_provider = {}
        for model_name, config in self.available_models.items():
            if config.provider not in by_provider:
                by_provider[config.provider] = []
            by_provider[config.provider].append(model_name)
        
        for provider, models in sorted(by_provider.items()):
            print(f"\n  {provider.upper()}:")
            for model in models[:3]:
                config = self.available_models[model]
                cost = "FREE" if config.cost_per_1k_input == 0 else f"${config.cost_per_1k_input}/1K"
                print(f"    • {config.display_name} ({cost})")
            if len(models) > 3:
                print(f"    ... and {len(models)-3} more")
        
        print("\n" + "="*70 + "\n")
    
    def evaluate_and_route(self, user_input: str) -> str:
        """Valuta complessità e seleziona modello ottimale"""
        complexity = self.router.evaluate(user_input)
        
        if self.metrics:
            self.metrics.histogram("complexity_score", complexity)
        
        selected_model = self._select_best_model(complexity, user_input)
        
        if self.verbose and selected_model != self.current_model:
            print(f"\n🔄 Switching to: {self.available_models[selected_model].display_name}")
            print(f"   Complexity: {complexity}/10")
            print(f"   Reason: {self._get_selection_reason(complexity)}")
            
            if self.metrics:
                self.metrics.increment("model_switches")
        
        if self.metrics:
            self.metrics.increment("routing_decisions")
        
        self.current_model = selected_model
        return selected_model
    
    def _select_best_model(self, complexity: int, user_input: str) -> str:
        """
        Seleziona modello basato su COMPLESSITÀ del task
        """
        if not self.available_models:
            raise Exception("No models available!")
        
        if complexity <= 2:
            min_quality = 1
            priority = "speed"
        elif complexity <= 4:
            min_quality = 6
            priority = "balanced"
        elif complexity <= 6:
            min_quality = 8
            priority = "balanced"
        elif complexity <= 8:
            min_quality = 9
            priority = "quality"
        else:
            min_quality = 9
            priority = "max_quality"
        
        suitable_models = [
            (name, config) for name, config in self.available_models.items()
            if config.quality_score >= min_quality
        ]
        
        if not suitable_models:
            suitable_models = sorted(
                self.available_models.items(), 
                key=lambda x: x[1].quality_score, 
                reverse=True
            )[:3]
        
        if priority == "speed":
            suitable_models.sort(key=lambda x: (
                {"ultra_fast": 0, "fast": 1, "medium": 2, "slow": 3}.get(x[1].speed, 4),
                -x[1].quality_score
            ))
        elif priority == "balanced":
            suitable_models.sort(key=lambda x: (
                -x[1].quality_score,
                {"fast": 0, "medium": 1, "ultra_fast": 2, "slow": 3}.get(x[1].speed, 4)
            ))
        elif priority == "quality":
            suitable_models.sort(key=lambda x: (
                -x[1].quality_score,
                {"medium": 0, "slow": 1, "fast": 2, "ultra_fast": 3}.get(x[1].speed, 4)
            ))
        else:
            suitable_models.sort(key=lambda x: -x[1].quality_score)
        
        selected = suitable_models[0][0]
        
        if self.verbose:
            print(f"\n   [DEBUG] Priority: {priority}, Min quality: {min_quality}")
            print(f"   [DEBUG] Suitable models: {len(suitable_models)}")
            if len(suitable_models) > 0:
                print(f"   [DEBUG] Top 3 choices:")
                for i, (name, config) in enumerate(suitable_models[:3], 1):
                    print(f"     {i}. {config.display_name} (Q:{config.quality_score}, S:{config.speed})")
        
        return selected
    
    def _get_selection_reason(self, complexity: int) -> str:
        """Genera spiegazione per la selezione"""
        if complexity <= 2:
            return "Very simple task - minimal model sufficient"
        elif complexity <= 4:
            return "Simple task - basic model adequate"
        elif complexity <= 6:
            return "Moderate task - balanced model needed"
        elif complexity <= 8:
            return "Complex task - high quality model required"
        else:
            return "Very complex task - maximum capability needed"
    
    def chat(self, user_input: str) -> str:
        """Chat principale con routing automatico e production features"""
        # Check cache first
        model_name = self.evaluate_and_route(user_input)
        config = self.available_models[model_name]
        
        # Check cache if enabled
        cached_response = None
        if self.cache:
            cached_response = self.cache.get(
                config.provider, config.model_id,
                self.context.get_messages_for_api(),
                temperature=0.7, max_tokens=2000
            )
            
            if cached_response:
                if self.verbose:
                    print(f"   ⚡ Using cached response")
                
                if self.metrics:
                    self.metrics.record_cache_event("hit")
                
                # Add to context
                self.context.add_message("user", user_input, None)
                self.context.add_message("assistant", cached_response, model_name)
                
                # Update stats
                self._update_stats(model_name, config, len(user_input), 
                                 len(cached_response), 0, cache_hit=True)
                
                return cached_response
        
        if self.cache:
            if self.metrics:
                self.metrics.record_cache_event("miss")
        
        # Check rate limits
        if self.rate_limiter:
            allowed = self.rate_limiter.wait_if_limited(config.provider, 
                                                       config.model_id, max_wait=5.0)
            if not allowed:
                if self.logger:
                    self.logger.warning(f"Rate limit exceeded for {config.provider}")
                
                # Try fallback
                return self._fallback_chat(user_input, exclude=model_name)
        
        # Prepare messages
        self.context.add_message("user", user_input, None)
        messages = self.context.get_messages_for_api()
        
        # Make API call
        provider = self.providers[config.provider]
        
        try:
            start_time = time.time()
            
            # API call with metrics
            if self.metrics:
                with self.metrics.time(f"api_call_{config.provider}"):
                    response = provider.chat(
                        messages=messages,
                        model=config.model_id,
                        temperature=0.7,
                        max_tokens=2000
                    )
            else:
                response = provider.chat(
                    messages=messages,
                    model=config.model_id,
                    temperature=0.7,
                    max_tokens=2000
                )
            
            elapsed = time.time() - start_time
            
            # Reset backoff on success
            if self.rate_limiter:
                self.rate_limiter.reset_backoff(config.provider)
            
            # Cache response
            if self.cache:
                self.cache.set(
                    config.provider, config.model_id, messages,
                    response, temperature=0.7, max_tokens=2000
                )
            
            # Add to context
            self.context.add_message("assistant", response, model_name)
            
            # Calculate tokens and cost
            input_tokens = len(user_input) // 4
            output_tokens = len(response) // 4
            cost = (input_tokens * config.cost_per_1k_input + 
                   output_tokens * config.cost_per_1k_output) / 1000
            
            # Update various tracking systems
            self._update_stats(model_name, config, len(user_input), 
                             len(response), elapsed)
            
            # Log to database
            if self.db:
                call_id = self.db.log_api_call(
                    self.session_id, config.provider, config.model_id,
                    user_input, response, input_tokens, output_tokens,
                    int(elapsed * 1000), cost, True,
                    complexity_score=self.router.evaluate(user_input)
                )
                
                self.db.update_session(
                    self.session_id,
                    messages=1,
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                    cost=cost
                )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_api_call(
                    config.provider, config.model_id, True,
                    elapsed * 1000, input_tokens, output_tokens, cost,
                    complexity=self.router.evaluate(user_input)
                )
            
            # Log API call
            if self.logger:
                self.logger.log_api_call(
                    config.provider, config.model_id,
                    input_tokens, output_tokens, elapsed, True
                )
            
            if self.verbose:
                self._print_response_info(config, elapsed)
            
            return response
            
        except Exception as e:
            error_msg = f"Error with {config.display_name}: {str(e)}"
            
            # Log error
            if self.logger:
                self.logger.error(error_msg, extra={
                    'provider': config.provider,
                    'model': config.model_id
                })
            
            # Record error in database
            if self.db:
                self.db.log_api_call(
                    self.session_id, config.provider, config.model_id,
                    user_input, None, 0, 0, 0, 0, False,
                    error_message=str(e)
                )
            
            # Record error metrics
            if self.metrics:
                self.metrics.record_api_call(
                    config.provider, config.model_id, False,
                    0, 0, 0, 0
                )
            
            # Apply backoff
            if self.rate_limiter:
                self.rate_limiter.apply_backoff(config.provider, e)
            
            if self.verbose:
                print(f"\n❌ {error_msg}")
            
            return self._fallback_chat(user_input, exclude=model_name)
    
    def _fallback_chat(self, user_input: str, exclude: str) -> str:
        """Fallback su altro modello se il principale fallisce"""
        fallback_models = [m for m in self.available_models.keys() if m != exclude]
        
        if not fallback_models:
            return "Sorry, no models are currently available."
        
        fallback = fallback_models[0]
        config = self.available_models[fallback]
        provider = self.providers[config.provider]
        
        if self.verbose:
            print(f"   Trying fallback: {config.display_name}")
        
        if self.logger:
            self.logger.info(f"Using fallback model: {config.display_name}")
        
        try:
            response = provider.chat(
                messages=self.context.get_messages_for_api(),
                model=config.model_id
            )
            
            self.context.add_message("assistant", response, fallback)
            return response
            
        except:
            return "I'm having trouble connecting to the models. Please try again."
    
    def _update_stats(self, model: str, config: ModelConfig, input_len: int, 
                     output_len: int, time: float, cache_hit: bool = False):
        """Aggiorna statistiche"""
        self.stats["total_messages"] += 1
        
        input_tokens = input_len / 4
        output_tokens = output_len / 4
        cost = (input_tokens * config.cost_per_1k_input + 
               output_tokens * config.cost_per_1k_output) / 1000
        self.stats["total_cost"] += cost
        
        if model not in self.stats["models_used"]:
            self.stats["models_used"][model] = 0
        self.stats["models_used"][model] += 1
        
        if config.provider not in self.stats["provider_distribution"]:
            self.stats["provider_distribution"][config.provider] = 0
        self.stats["provider_distribution"][config.provider] += 1
    
    def _print_response_info(self, config: ModelConfig, elapsed: float):
        """Stampa info sulla risposta"""
        cost_str = "FREE" if config.cost_per_1k_input == 0 else f"~${self.stats['total_cost']:.6f} total"
        print(f"   ⚡ {elapsed:.2f}s | {cost_str} | Context: {self.context.get_summary()}")
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche sessione"""
        stats = {
            **self.stats,
            "context_summary": self.context.get_summary(),
            "current_model": self.current_model,
            "available_models": len(self.available_models)
        }
        
        # Add production stats if enabled
        if self.enable_production:
            if self.cache:
                stats["cache"] = self.cache.get_stats()
            if self.metrics:
                stats["metrics"] = self.metrics.get_summary()
            if self.rate_limiter:
                stats["rate_limits"] = self.rate_limiter.get_limits_info()
            if self.db:
                stats["session"] = self.db.get_session_stats(self.session_id)
        
        return stats
    
    def clear_context(self):
        """Pulisce il contesto mantenendo le stats"""
        self.context = ConversationContext()
        if self.verbose:
            print("\n🗑️  Context cleared! Starting fresh conversation.")
    
    def save_conversation(self, filepath: str = None):
        """Salva conversazione su file"""
        if not filepath:
            filepath = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "messages": self.context.messages,
            "stats": self.stats,
            "model_history": self.context.model_history
        }
        
        if self.enable_production and self.db:
            data["session_data"] = self.db.get_session_stats(self.session_id)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"\n💾 Conversation saved to {filepath}")
    
    def get_analytics(self, hours: int = 24) -> Dict:
        """Get production analytics"""
        if not self.enable_production or not self.db:
            return {}
        
        return {
            "model_performance": self.db.get_model_analytics(hours),
            "error_analysis": self.db.get_error_analysis(hours),
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "metrics_summary": self.metrics.get_summary() if self.metrics else {}
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.enable_production:
            if self.db:
                self.db.update_session(self.session_id, end_time=True)
                self.db.cleanup_old_data(days=30)
            
            if self.logger:
                self.logger.info("Session ended", extra={
                    "session_id": self.session_id,
                    "total_messages": self.stats["total_messages"],
                    "total_cost": self.stats["total_cost"]
                })

# ====================
# LLM-USE V2 WITH AUTO-DISCOVERY
# ====================

class LLMUSEV2(LlmUse):
    """llm-use con auto-discovery, benchmarking e production features"""
    
    def __init__(self, verbose: bool = True, auto_discover: bool = True, 
                 enable_production: bool = True):
        # Initialize base class with production features
        super().__init__(verbose=False, enable_production=enable_production)
        
        self.verbose = verbose
        self.auto_discover = auto_discover
        
        # Discover models
        if auto_discover:
            self._discover_new_models()
        
        if verbose:
            self._print_initialization()
    
    def _discover_new_models(self):
        """Scopre e aggiunge nuovi modelli al catalogo"""
        if self.verbose:
            print("\n🔬 AUTO-DISCOVERY MODE")
            print("="*50)
        
        discoverer = ModelAutoDiscovery(verbose=self.verbose)
        
        discovered = discoverer.discover_and_benchmark_ollama_models()
        
        for model_id, config in discovered.items():
            self.available_models[model_id] = config
        
        if self.verbose and discovered:
            print(f"\n✅ Added {len(discovered)} Ollama models to catalog")
            print("="*50)
    
    def refresh_models(self):
        """Forza re-discovery dei modelli"""
        if self.verbose:
            print("\n🔄 Refreshing model catalog...")
        
        discoverer = ModelAutoDiscovery(verbose=self.verbose)
        discoverer.benchmarks = {}
        discoverer._save_benchmarks()
        
        self._discover_new_models()

# ====================
# INTERACTIVE INTERFACE
# ====================

def interactive_chat():
    """Chat interattiva con routing automatico e production features"""
    
    print("\n" + "🚀"*35)
    print("  LLM-USE PRODUCTION - INTELLIGENT ROUTING")
    print("  Production monitoring and caching enabled")
    print("🚀"*35 + "\n")
    
    # Setup with production features
    print("Initializing production system with auto-discovery...")
    router = LLMUSEV2(verbose=True, auto_discover=True, enable_production=True)
    
    if not router.available_models:
        print("\n❌ No models available! Please:")
        print("   • Set API keys (export OPENAI_API_KEY=...)")
        print("   • Or install Ollama models (ollama pull mistral)")
        return
    
    print("\n📝 Commands:")
    print("  /stats    - Show session statistics")
    print("  /analytics - Show production analytics")
    print("  /clear    - Clear conversation context")
    print("  /save     - Save conversation")
    print("  /models   - List all available models")
    print("  /refresh  - Re-discover models")
    print("  /quit     - Exit\n")
    
    print("💬 Start chatting! I'll automatically choose the best model for each task.\n")
    
    try:
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/stats':
                    stats = router.get_stats()
                    print("\n📊 SESSION STATS:")
                    print(f"   Messages: {stats['total_messages']}")
                    print(f"   Total cost: ${stats['total_cost']:.6f}")
                    print(f"   Models used: {len(stats['models_used'])}")
                    print(f"   {stats['context_summary']}")
                    
                    if 'cache' in stats:
                        cache_stats = stats['cache']
                        print(f"\n   Cache: {cache_stats['hit_rate']*100:.1f}% hit rate")
                        print(f"   Size: {cache_stats['current_size_mb']:.1f}/{cache_stats['max_size_mb']}MB")
                    
                    if 'metrics' in stats:
                        metrics = stats['metrics']
                        print(f"\n   Avg latency: {metrics['avg_latency_ms']:.0f}ms")
                        print(f"   Error rate: {metrics['error_rate']*100:.1f}%")
                    
                    if stats['models_used']:
                        print("\n   Model distribution:")
                        for model, count in sorted(stats['models_used'].items(), 
                                                  key=lambda x: x[1], reverse=True):
                            if model in router.available_models:
                                print(f"     • {router.available_models[model].display_name}: {count}x")
                    continue
                
                elif user_input.lower() == '/analytics':
                    analytics = router.get_analytics(hours=24)
                    
                    if analytics:
                        print("\n📈 PRODUCTION ANALYTICS (24h):")
                        
                        if 'model_performance' in analytics:
                            print("\n   Model Performance:")
                            for model_data in analytics['model_performance'][:5]:
                                print(f"     • {model_data['model']}: "
                                     f"{model_data['total_calls']} calls, "
                                     f"{model_data['avg_latency']:.0f}ms avg, "
                                     f"{model_data['success_rate']:.1f}% success")
                        
                        if 'error_analysis' in analytics:
                            errors = analytics['error_analysis']
                            if errors:
                                print(f"\n   Recent Errors:")
                                for error in errors[:3]:
                                    print(f"     • {error['model']}: {error['error_message'][:50]}... "
                                         f"({error['error_count']}x)")
                    continue
                
                elif user_input.lower() == '/clear':
                    router.clear_context()
                    continue
                
                elif user_input.lower() == '/save':
                    router.save_conversation()
                    continue
                
                elif user_input.lower() == '/models':
                    print("\n📦 AVAILABLE MODELS:")
                    
                    by_provider = {}
                    for name, config in router.available_models.items():
                        if config.provider not in by_provider:
                            by_provider[config.provider] = []
                        by_provider[config.provider].append((name, config))
                    
                    for provider, models in sorted(by_provider.items()):
                        print(f"\n  {provider.upper()} ({len(models)} models):")
                        for name, config in sorted(models, key=lambda x: -x[1].quality_score):
                            cost = "FREE" if config.cost_per_1k_input == 0 else f"${config.cost_per_1k_input}/1K"
                            size_str = f"{config.size_gb:.1f}GB" if hasattr(config, 'size_gb') and config.size_gb > 0 else "unknown"
                            print(f"    • {config.display_name}")
                            print(f"      Quality: {config.quality_score}/10 | Speed: {config.speed} | Size: {size_str} | Cost: {cost}")
                            print(f"      Best for: {', '.join(config.best_for[:3])}")
                    continue
                
                elif user_input.lower() == '/refresh':
                    router.refresh_models()
                    continue
                
                # Normal chat
                print("\n🤖 Assistant: ", end="")
                response = router.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted! Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if router.logger:
                    router.logger.error(f"Chat error: {e}")
    
    finally:
        # Cleanup on exit
        if hasattr(router, 'cleanup'):
            router.cleanup()

# ====================
# MAIN ENTRY POINT
# ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Test mode - simplified for production
            print("Running production tests...")
            router = LLMUSEV2(verbose=True, auto_discover=True, enable_production=True)
            
            test_inputs = [
                "Hi!",
                "What's 2+2?",
                "Explain quantum computing",
                "Write a Python web scraper"
            ]
            
            for test_input in test_inputs:
                print(f"\nTest: {test_input}")
                response = router.chat(test_input)
                print(f"Response: {response[:100]}...")
            
            # Print analytics
            print("\n" + "="*70)
            print("TEST ANALYTICS:")
            print(json.dumps(router.get_stats(), indent=2))
            
            router.cleanup()
            
        elif command == "benchmark":
            discoverer = ModelAutoDiscovery(verbose=True)
            discoverer.discover_and_benchmark_ollama_models(force_benchmark=True)
            
        elif command == "analytics":
            # Show stored analytics
            db = ProductionDB()
            analytics = {
                "model_performance": db.get_model_analytics(24),
                "error_analysis": db.get_error_analysis(24)
            }
            print(json.dumps(analytics, indent=2, default=str))
            
        elif command == "help":
            print("\nUsage:")
            print("  python llm-use-production.py          # Interactive chat")
            print("  python llm-use-production.py test     # Run tests")
            print("  python llm-use-production.py benchmark# Benchmark models")
            print("  python llm-use-production.py analytics# View analytics")
            print("  python llm-use-production.py help     # Show this help")
        else:
            print(f"Unknown command: {command}")
    else:
        interactive_chat()