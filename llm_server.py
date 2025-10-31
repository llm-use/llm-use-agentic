# llm_arena_server.py
"""
LLM-USE Web - Web Interface Backend
Enhanced with proper context management and features
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import traceback
import time
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import your LLM-USE system
from llm_use import LLMUSEV2, ConversationContext

# ====================
# DATA MODELS
# ====================

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    model: Optional[str] = None
    model_name: Optional[str] = None
    tokens: Optional[Dict] = None
    cost: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    mode: str = "single"  # single, battle, arena
    model1: Optional[str] = None
    model2: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    session_id: Optional[str] = None
    stream: bool = True
    force_model: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    tier: str  # Use tier instead of quality_score
    speed: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    context_window: int
    best_for: List[str]
    complexity_range: tuple

# ====================
# ENHANCED ARENA BACKEND
# ====================

class EnhancedArenaBackend:
    """Enhanced Backend for LLM Web interface with proper context management"""
    
    def __init__(self):
        print("🚀 Initializing Enhanced LLM-USE Web Backend...")
        
        # Initialize LLM-USE with production features
        self.llm_use = LLMUSEV2(
            verbose=True, 
            auto_discover=True,
            enable_production=True
        )
        
        # Session management with proper context
        self.sessions: Dict[str, Dict] = {}
        self.session_contexts: Dict[str, ConversationContext] = {}
        
        # Battle management
        self.battles: Dict[str, Dict] = {}
        self.battle_queue: List[str] = []  # For arena matchmaking
        
        # Leaderboard with ELO
        self.leaderboard: Dict[str, Dict] = {}
        self._init_leaderboard()
        
        # WebSocket connections
        self.connections: Dict[str, WebSocket] = {}
        
        # Analytics
        self.analytics = {
            "total_requests": 0,
            "total_battles": 0,
            "total_votes": 0,
            "provider_usage": defaultdict(int),
            "model_usage": defaultdict(int),
            "avg_response_times": defaultdict(list),
            "error_count": 0
        }
        
        # Rate limiting per session
        self.rate_limits: Dict[str, Dict] = {}
        
        print(f"✅ Arena ready with {len(self.llm_use.available_models)} models")
    
    def _init_leaderboard(self):
        """Initialize model leaderboard with ELO ratings"""
        for model_id, config in self.llm_use.available_models.items():
            # Calculate initial ELO based on tier
            tier_elo = {
                "expert": 1700,
                "professional": 1600,
                "competent": 1500,
                "basic": 1400,
                "minimal": 1300
            }
            
            initial_elo = tier_elo.get(config.tier, 1500)
            
            self.leaderboard[model_id] = {
                "name": config.display_name,
                "provider": config.provider,
                "tier": config.tier,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "elo": initial_elo,
                "peak_elo": initial_elo,
                "total_battles": 0,
                "avg_response_time": 0,
                "total_cost": 0,
                "user_ratings": [],
                "last_used": None
            }
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create new chat session with proper context"""
        session_id = str(uuid.uuid4())
        
        # Create session data
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "messages": [],  # Store for UI display
            "battles": [],
            "current_models": [],
            "stats": {
                "total_messages": 0,
                "total_cost": 0,
                "total_tokens": 0,
                "models_used": set(),
                "avg_complexity": 0
            },
            "preferences": {
                "preferred_speed": "medium",
                "preferred_quality": "balanced",
                "max_cost_per_request": None
            }
        }
        
        # Create dedicated context for this session
        self.session_contexts[session_id] = ConversationContext(max_tokens=8000)
        
        # Initialize rate limiting
        self.rate_limits[session_id] = {
            "requests": 0,
            "last_reset": time.time(),
            "limit": 100  # requests per minute
        }
        
        print(f"📝 Created session: {session_id}")
        return session_id
    
    def _check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limits"""
        if session_id not in self.rate_limits:
            return True
        
        limit_data = self.rate_limits[session_id]
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - limit_data["last_reset"] > 60:
            limit_data["requests"] = 0
            limit_data["last_reset"] = current_time
        
        if limit_data["requests"] >= limit_data["limit"]:
            return False
        
        limit_data["requests"] += 1
        return True
    
    def get_available_models(self, filters: Optional[Dict] = None) -> List[ModelInfo]:
        """Get list of available models with optional filtering"""
        models = []
        
        for model_id, config in self.llm_use.available_models.items():
            # Apply filters if provided
            if filters:
                if "provider" in filters and config.provider != filters["provider"]:
                    continue
                if "min_quality" in filters:
                    tier_order = {"expert": 5, "professional": 4, "competent": 3, "basic": 2, "minimal": 1}
                    if tier_order.get(config.tier, 0) < filters["min_quality"]:
                        continue
                if "max_cost" in filters and config.cost_per_1k_output > filters["max_cost"]:
                    continue
            
            models.append(ModelInfo(
                id=model_id,
                name=config.display_name,
                provider=config.provider,
                tier=config.tier,
                speed=config.speed,
                cost_per_1k_input=config.cost_per_1k_input,
                cost_per_1k_output=config.cost_per_1k_output,
                context_window=config.context_window,
                best_for=config.best_for,
                complexity_range=config.complexity_range
            ))
        
        # Sort by tier then by name
        tier_order = {"expert": 5, "professional": 4, "competent": 3, "basic": 2, "minimal": 1}
        models.sort(key=lambda x: (tier_order.get(x.tier, 0), x.name), reverse=True)
        
        return models
    
    # In EnhancedArenaBackend, modifica il metodo process_chat:

    async def process_chat(self, request: ChatRequest, websocket: WebSocket) -> Dict:
        """Process chat request with proper context management"""
        
        # Create or get session
        if not request.session_id:
            request.session_id = self.create_session()
        
        session = self.sessions.get(request.session_id)
        if not session:
            request.session_id = self.create_session()
            session = self.sessions[request.session_id]
        
        # Check rate limit
        if not self._check_rate_limit(request.session_id):
            await websocket.send_json({
                "type": "error",
                "error": "Rate limit exceeded. Please wait a moment."
            })
            return {"status": "error", "error": "Rate limit exceeded"}
        
        # Update analytics
        self.analytics["total_requests"] += 1
        
        # Get or create session context
        context = self.session_contexts.get(request.session_id)
        if not context:
            context = ConversationContext(max_tokens=8000)
            self.session_contexts[request.session_id] = context
            
            # 🔥 FIX: RICOSTRUISCI IL CONTESTO DAI MESSAGGI PRECEDENTI!
            # Aggiungi tutti i messaggi precedenti della sessione al context
            for msg in session["messages"]:
                if msg["role"] in ["user", "assistant"]:
                    context.add_message(
                        msg["role"], 
                        msg["content"], 
                        msg.get("model")
                    )
            
            print(f"📚 Rebuilt context with {len(session['messages'])} previous messages")
        
        # Add current user message to context
        context.add_message("user", request.message, None)
        
        # Also add to session messages for UI
        session["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Handle different modes
        if request.mode == "single":
            return await self._process_single_chat_fixed(request, session, context, websocket)
        elif request.mode == "battle":
            return await self._process_battle_chat(request, session, context, websocket)
        elif request.mode == "arena":
            return await self._process_arena_chat(request, session, context, websocket)
    
    async def _process_single_chat_fixed(self, request: ChatRequest, 
                                     session: Dict, 
                                     context: ConversationContext,
                                     websocket: WebSocket) -> Dict:
        """Process single model chat with FIXED context management"""
        
        try:
            # 🔥 FIX: RICOSTRUISCI IL CONTESTO COMPLETO DALLA SESSIONE
            # Questo assicura che abbiamo SEMPRE tutta la storia
            
            # Prima, assicuriamoci che il context sia sincronizzato con la sessione
            if len(session["messages"]) > 0:
                # Conta i messaggi nel context
                context_messages = context.get_messages_for_api()
                context_msg_count = len([m for m in context_messages if m["role"] != "system"])
                session_msg_count = len(session["messages"])
                
                # Se il context è indietro rispetto alla sessione, ricostruiscilo
                if context_msg_count < session_msg_count:
                    print(f"🔄 Context out of sync ({context_msg_count} vs {session_msg_count}), rebuilding...")
                    
                    # Svuota il context corrente
                    context.messages = []
                    context.token_count = 0
                    
                    # Ricostruisci dal session storage
                    for msg in session["messages"]:
                        if msg["role"] in ["user", "assistant"]:
                            context.add_message(
                                msg["role"],
                                msg["content"],
                                msg.get("model")
                            )
                    
                    # Aggiungi il messaggio corrente se non è già presente
                    if not session["messages"] or session["messages"][-1]["content"] != request.message:
                        context.add_message("user", request.message, None)
                    
                    print(f"✅ Context rebuilt with {len(session['messages'])} messages")
            
            # Ottieni i messaggi dal context (ora sincronizzato)
            messages_for_api = context.get_messages_for_api()
            
            # Se il context è vuoto, costruisci manualmente
            if len(messages_for_api) == 0 or (len(messages_for_api) == 1 and messages_for_api[0]["role"] == "system"):
                print("⚠️ Context is empty, building from session...")
                
                messages_for_api = []
                
                # Aggiungi system message
                messages_for_api.append({
                    "role": "system",
                    "content": """You are a helpful assistant with perfect memory of our conversation.

                IMPORTANT INSTRUCTIONS:
                1. ALWAYS remember what the user told you earlier in the conversation
                2. When asked about previous messages, look back at the conversation history
                3. Be specific when recalling previous messages - quote exactly what was said
                4. If the user asks "what did I ask before?" or similar, refer to their exact previous questions

                Remember: You have access to the ENTIRE conversation history. Use it!"""
                })
                
                # Aggiungi tutti i messaggi dalla sessione
                for msg in session["messages"]:
                    if msg["role"] in ["user", "assistant"]:
                        messages_for_api.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Aggiungi il messaggio corrente
                messages_for_api.append({
                    "role": "user",
                    "content": request.message
                })
            
            # Evaluate complexity
            complexity = self.llm_use.router.evaluate(request.message)
            
            # Model selection
            if request.model1 and request.model1 != "auto":
                # Manual selection
                model_id = request.model1
                if model_id not in self.llm_use.available_models:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Model {model_id} not available"
                    })
                    return {"status": "error", "error": "Model not available"}
                
                config = self.llm_use.available_models[model_id]
                selection_reason = "Manual selection"
            else:
                # Auto-routing based on complexity
                model_id = self.llm_use._select_best_model(complexity, request.message)
                config = self.llm_use.available_models[model_id]
                selection_reason = f"Complexity: {complexity}/10"
            
            # Send model selection info
            await websocket.send_json({
                "type": "model_selected",
                "model": config.display_name,
                "model_id": model_id,
                "complexity": complexity,
                "reason": selection_reason,
                "tier": config.tier if hasattr(config, 'tier') else 'unknown',
                "estimated_cost": self._estimate_cost(request.message, config)
            })
            
            # Get provider
            provider = self.llm_use.providers[config.provider]
            
            # Debug: mostra esattamente cosa stiamo inviando
            print(f"💬 Calling {config.display_name}")
            print(f"   Context messages: {len(messages_for_api)}")
            print(f"   Session history: {len(session['messages'])} messages")
            print(f"   Token estimate: ~{context.token_count} tokens")
            
            # Mostra anteprima dei messaggi per debug
            if self.llm_use.verbose:
                for i, msg in enumerate(messages_for_api[-3:]):  # Ultimi 3 messaggi
                    preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    print(f"     [{msg['role']}]: {preview}")
            
            # Start timing
            start_time = time.time()
            
            # Make API call
            try:
                response = provider.chat(
                    messages=messages_for_api,
                    model=config.model_id,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                elapsed_time = time.time() - start_time
                print(f"✅ Response received in {elapsed_time:.2f}s")
                
            except Exception as api_error:
                print(f"❌ API Error: {str(api_error)}")
                
                # Handle context length errors
                if any(word in str(api_error).lower() for word in ["context", "token", "length", "too long"]):
                    print("⚠️ Context too long, trimming...")
                    
                    # Mantieni solo system + ultimi N messaggi
                    trimmed_messages = messages_for_api[:1]  # System message
                    
                    # Prendi gli ultimi messaggi che stanno nel limite
                    if len(messages_for_api) > 10:
                        trimmed_messages.extend(messages_for_api[-10:])  # Ultimi 10
                    else:
                        trimmed_messages.extend(messages_for_api[1:])  # Tutti tranne system
                    
                    print(f"   Trimmed from {len(messages_for_api)} to {len(trimmed_messages)} messages")
                    
                    try:
                        response = provider.chat(
                            messages=trimmed_messages,
                            model=config.model_id,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens
                        )
                        elapsed_time = time.time() - start_time
                        
                        # Avvisa l'utente che il context è stato ridotto
                        await websocket.send_json({
                            "type": "warning",
                            "message": "Context was too long and has been trimmed to recent messages"
                        })
                        
                    except Exception as retry_error:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"API failed even after trimming: {str(retry_error)}"
                        })
                        return {"status": "error", "error": str(retry_error)}
                else:
                    # Errore non relativo al context
                    await websocket.send_json({
                        "type": "error",
                        "error": f"API error: {str(api_error)}"
                    })
                    return {"status": "error", "error": str(api_error)}
            
            # Stream response if requested
            if request.stream:
                await self._stream_response(response, websocket, "response_token")
            
            # IMPORTANTE: Aggiorna il context con la risposta
            context.add_message("assistant", response, model_id)
            
            # Calculate metrics
            input_tokens = sum(len(m["content"]) for m in messages_for_api) // 4
            output_tokens = len(response) // 4
            total_tokens = input_tokens + output_tokens
            cost = (input_tokens * config.cost_per_1k_input + 
                output_tokens * config.cost_per_1k_output) / 1000
            
            # Store message for UI
            # NOTA: NON duplicare il messaggio user, è già stato aggiunto
            assistant_message = {
                "role": "assistant",
                "content": response,
                "model": model_id,
                "model_name": config.display_name,
                "timestamp": datetime.now().isoformat(),
                "tokens": {"input": input_tokens, "output": output_tokens},
                "cost": cost,
                "response_time": elapsed_time
            }
            session["messages"].append(assistant_message)
            
            # Update session stats
            session["stats"]["total_messages"] = len(session["messages"])
            session["stats"]["total_cost"] += cost
            session["stats"]["total_tokens"] += total_tokens
            
            if isinstance(session["stats"]["models_used"], set):
                session["stats"]["models_used"].add(model_id)
            else:
                session["stats"]["models_used"] = {model_id}
            
            # Update analytics
            self.analytics["model_usage"][model_id] += 1
            self.analytics["provider_usage"][config.provider] += 1
            self.analytics["avg_response_times"][model_id].append(elapsed_time)
            
            # Update leaderboard stats
            if model_id in self.leaderboard:
                times_list = self.analytics["avg_response_times"][model_id]
                self.leaderboard[model_id]["avg_response_time"] = (
                    sum(times_list) / len(times_list) if times_list else 0
                )
                self.leaderboard[model_id]["total_cost"] += cost
                self.leaderboard[model_id]["last_used"] = datetime.now().isoformat()
            
            # Send completion with full stats
            await websocket.send_json({
                "type": "response_complete",
                "model": config.display_name,
                "model_id": model_id,
                "response": response,
                "session_id": request.session_id,  # ← AGGIUNGI QUESTA RIGA!
                "stats": {
                    "response_time": round(elapsed_time, 2),
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": total_tokens
                    },
                    "cost": round(cost, 6),
                    "context_length": len(messages_for_api),
                    "session_totals": {
                        "messages": len(session["messages"]),
                        "cost": round(session["stats"]["total_cost"], 6),
                        "tokens": session["stats"]["total_tokens"],
                        "models_used": len(session["stats"]["models_used"])
                    },
                    "conversation_turns": len([m for m in session["messages"] if m["role"] == "user"])
                }
            })
            
            print(f"📊 Session now has {len(session['messages'])} messages total")
            print(f"   Context is tracking {len(context.messages)} messages")
            
            return {
                "status": "success",
                "model": model_id,
                "model_name": config.display_name,
                "response": response,
                "cost": cost,
                "tokens": total_tokens,
                "session_id": request.session_id
            }
            
        except Exception as e:
            error_msg = f"Error in chat processing: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            self.analytics["error_count"] += 1
            
            await websocket.send_json({
                "type": "error",
                "error": error_msg,
                "traceback": traceback.format_exc() if request.session_id == "debug" else None
            })
            
            return {"status": "error", "error": error_msg}
    
    async def _process_battle_chat(self, request: ChatRequest, 
                                   session: Dict,
                                   context: ConversationContext,
                                   websocket: WebSocket) -> Dict:
        """Process battle mode with two models competing"""
        
        try:
            # Select models for battle
            if request.model1 and request.model2:
                model1_id = request.model1
                model2_id = request.model2
            else:
                # Auto-select contrasting models
                complexity = self.llm_use.router.evaluate(request.message)
                
                # Get primary model
                model1_id = self.llm_use._select_best_model(complexity, request.message)
                
                # Get contrasting model
                model2_id = self._select_contrasting_model(model1_id, complexity)
            
            config1 = self.llm_use.available_models[model1_id]
            config2 = self.llm_use.available_models[model2_id]
            
            # Create battle record
            battle_id = str(uuid.uuid4())[:8]
            self.battles[battle_id] = {
                "id": battle_id,
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat(),
                "models": {
                    "model1": {"id": model1_id, "name": config1.display_name, "tier": config1.tier},
                    "model2": {"id": model2_id, "name": config2.display_name, "tier": config2.tier}
                },
                "prompt": request.message,
                "responses": {},
                "metrics": {},
                "vote": None
            }
            
            # Send battle start
            await websocket.send_json({
                "type": "battle_start",
                "battle_id": battle_id,
                "model1": {"id": model1_id, "name": config1.display_name, "tier": config1.tier},
                "model2": {"id": model2_id, "name": config2.display_name, "tier": config2.tier}
            })
            
            # Generate responses in parallel
            messages_for_api = context.get_messages_for_api()
            
            # Model 1
            await websocket.send_json({
                "type": "model1_thinking",
                "model": config1.display_name
            })
            
            start1 = time.time()
            response1 = await self._generate_model_response(
                model1_id, messages_for_api, request, websocket, "model1_token"
            )
            time1 = time.time() - start1
            
            # Model 2
            await websocket.send_json({
                "type": "model2_thinking", 
                "model": config2.display_name
            })
            
            start2 = time.time()
            response2 = await self._generate_model_response(
                model2_id, messages_for_api, request, websocket, "model2_token"
            )
            time2 = time.time() - start2
            
            # Store results
            self.battles[battle_id]["responses"] = {
                "model1": response1,
                "model2": response2
            }
            
            self.battles[battle_id]["metrics"] = {
                "model1": {"time": time1, "length": len(response1)},
                "model2": {"time": time2, "length": len(response2)}
            }
            
            session["battles"].append(battle_id)
            self.analytics["total_battles"] += 1
            
            # Send completion
            await websocket.send_json({
                "type": "battle_complete",
                "battle_id": battle_id,
                "responses": self.battles[battle_id]["responses"],
                "metrics": self.battles[battle_id]["metrics"],
                "ready_to_vote": True
            })
            
            return {
                "status": "success",
                "battle_id": battle_id,
                "models": [model1_id, model2_id]
            }
            
        except Exception as e:
            error_msg = f"Battle error: {str(e)}"
            print(f"❌ {error_msg}")
            
            await websocket.send_json({
                "type": "error",
                "error": error_msg
            })
            
            return {"status": "error", "error": error_msg}
    
    async def _process_arena_chat(self, request: ChatRequest,
                                  session: Dict,
                                  context: ConversationContext,
                                  websocket: WebSocket) -> Dict:
        """Process arena mode (blind test)"""
        
        # Process as battle but hide model identities
        await websocket.send_json({
            "type": "arena_mode_start",
            "message": "Two models will compete blindly. Vote for the best!"
        })
        
        # Temporarily modify request to hide models
        original_mode = request.mode
        request.mode = "battle"
        
        result = await self._process_battle_chat(request, session, context, websocket)
        
        if result["status"] == "success":
            # Mark as arena battle
            self.battles[result["battle_id"]]["arena_mode"] = True
            
            await websocket.send_json({
                "type": "arena_ready",
                "battle_id": result["battle_id"],
                "message": "Responses ready! Vote for A or B without knowing the models."
            })
        
        request.mode = original_mode
        return result
    
    def _select_contrasting_model(self, primary_model: str, complexity: int) -> str:
        """Select a model that contrasts with the primary selection"""
        
        primary_config = self.llm_use.available_models[primary_model]
        candidates = []
        
        for model_id, config in self.llm_use.available_models.items():
            if model_id == primary_model:
                continue
            
            # Score based on contrast
            contrast_score = 0
            
            # Different provider is good
            if config.provider != primary_config.provider:
                contrast_score += 3
            
            # Different tier is interesting
            if config.tier != primary_config.tier:
                contrast_score += 2
            
            # Different speed characteristics
            if config.speed != primary_config.speed:
                contrast_score += 1
            
            # Still should handle the complexity
            min_c, max_c = config.complexity_range
            if min_c <= complexity <= max_c:
                contrast_score += 2
            
            candidates.append((model_id, contrast_score))
        
        # Sort by contrast score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0] if candidates else primary_model
    
    async def _generate_model_response(self, model_id: str, 
                                       messages: List[Dict],
                                       request: ChatRequest,
                                       websocket: WebSocket,
                                       token_type: str) -> str:
        """Generate response from specific model"""
        
        config = self.llm_use.available_models[model_id]
        provider = self.llm_use.providers[config.provider]
        
        try:
            response = provider.chat(
                messages=messages,
                model=config.model_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Stream if requested
            if request.stream:
                await self._stream_response(response, websocket, token_type)
            
            return response
            
        except Exception as e:
            error_msg = f"Model {config.display_name} error: {str(e)}"
            return error_msg
    
    async def _stream_response(self, response: str, websocket: WebSocket, token_type: str):
        """Stream response in chunks for better UX"""
        
        # Split into words for more natural streaming
        words = response.split()
        chunk_size = 3  # Words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            
            # Add space if not last chunk
            if i + chunk_size < len(words):
                chunk += ' '
            
            await websocket.send_json({
                "type": token_type,
                "content": chunk
            })
            
            # Variable delay for more natural feeling
            delay = 0.03 if len(chunk) < 10 else 0.05
            await asyncio.sleep(delay)
    
    def _estimate_cost(self, message: str, config) -> float:
        """Estimate cost for a request"""
        estimated_input_tokens = len(message) // 4
        estimated_output_tokens = 500  # Average estimate
        
        return (estimated_input_tokens * config.cost_per_1k_input + 
                estimated_output_tokens * config.cost_per_1k_output) / 1000
    
    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up old sessions to save memory"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        to_remove = []
        
        for session_id, session in self.sessions.items():
            created = datetime.fromisoformat(session["created"]).timestamp()
            if created < cutoff:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
            if session_id in self.session_contexts:
                del self.session_contexts[session_id]
            if session_id in self.rate_limits:
                del self.rate_limits[session_id]
        
        print(f"🧹 Cleaned up {len(to_remove)} old sessions")

    async def process_chat(self, request: ChatRequest, websocket: WebSocket) -> Dict:
        """Process chat request with proper context management and continuation support"""
        
        # Create or get session
        if not request.session_id:
            request.session_id = self.create_session()
        
        session = self.sessions.get(request.session_id)
        if not session:
            request.session_id = self.create_session()
            session = self.sessions[request.session_id]
        
        # Check rate limit
        if not self._check_rate_limit(request.session_id):
            await websocket.send_json({
                "type": "error",
                "error": "Rate limit exceeded. Please wait a moment."
            })
            return {"status": "error", "error": "Rate limit exceeded"}
        
        # Update analytics
        self.analytics["total_requests"] += 1
        
        # Get or create session context
        context = self.session_contexts.get(request.session_id)
        if not context:
            context = ConversationContext(max_tokens=8000)
            self.session_contexts[request.session_id] = context
            
            # Rebuild context from previous messages
            for msg in session["messages"]:
                if msg["role"] in ["user", "assistant"]:
                    context.add_message(
                        msg["role"], 
                        msg["content"], 
                        msg.get("model")
                    )
            
            print(f"📚 Rebuilt context with {len(session['messages'])} previous messages")
        
        # 🆕 CHECK FOR CONTINUATION REQUEST
        is_continuation = False
        force_model = None
        
        # Detect continuation patterns
        continuation_patterns = ['continua', 'continue', 'go on', 'prosegui', 'vai avanti', 'continua da dove']
        message_lower = request.message.lower().strip()
        
        if any(pattern in message_lower for pattern in continuation_patterns) and len(message_lower) < 30:
            # Check if this looks like a continuation request
            if session.get('last_model_used'):
                force_model = session['last_model_used']
                is_continuation = True
                
                model_name = self.llm_use.available_models[force_model].display_name if force_model in self.llm_use.available_models else force_model
                
                await websocket.send_json({
                    "type": "continuation_detected",
                    "model": force_model,
                    "model_name": model_name,
                    "message": f"Continuing with {model_name}"
                })
        
        # Override model if force_model provided in request
        if request.force_model:
            force_model = request.force_model
            is_continuation = True
        
        # Add current user message to context
        context.add_message("user", request.message, None)
        
        # Also add to session messages for UI
        session["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Handle different modes
        if request.mode == "single":
            return await self._process_single_chat_with_continuation(
                request, session, context, websocket, force_model, is_continuation
            )
        elif request.mode == "battle":
            return await self._process_battle_chat(request, session, context, websocket)
        elif request.mode == "arena":
            return await self._process_arena_chat(request, session, context, websocket)

    async def _process_single_chat_with_continuation(self, request: ChatRequest, 
                                                    session: Dict, 
                                                    context: ConversationContext,
                                                    websocket: WebSocket,
                                                    force_model: Optional[str] = None,
                                                    is_continuation: bool = False) -> Dict:
        """Process single model chat with continuation support"""
        
        try:
            # Rebuild context if needed
            if len(session["messages"]) > 0:
                context_messages = context.get_messages_for_api()
                context_msg_count = len([m for m in context_messages if m["role"] != "system"])
                session_msg_count = len(session["messages"])
                
                if context_msg_count < session_msg_count:
                    print(f"🔄 Context out of sync ({context_msg_count} vs {session_msg_count}), rebuilding...")
                    
                    context.messages = []
                    context.token_count = 0
                    
                    for msg in session["messages"]:
                        if msg["role"] in ["user", "assistant"]:
                            context.add_message(
                                msg["role"],
                                msg["content"],
                                msg.get("model")
                            )
                    
                    if not session["messages"] or session["messages"][-1]["content"] != request.message:
                        context.add_message("user", request.message, None)
                    
                    print(f"✅ Context rebuilt with {len(session['messages'])} messages")
            
            messages_for_api = context.get_messages_for_api()
            
            if len(messages_for_api) == 0 or (len(messages_for_api) == 1 and messages_for_api[0]["role"] == "system"):
                print("⚠️ Context is empty, building from session...")
                
                messages_for_api = []
                
                messages_for_api.append({
                    "role": "system",
                    "content": """You are a helpful assistant with perfect memory of our conversation.

    IMPORTANT INSTRUCTIONS:
    1. ALWAYS remember what the user told you earlier in the conversation
    2. When asked about previous messages, look back at the conversation history
    3. Be specific when recalling previous messages - quote exactly what was said
    4. If the user asks "what did I ask before?" or similar, refer to their exact previous questions
    5. When continuing a response, pick up exactly where you left off

    Remember: You have access to the ENTIRE conversation history. Use it!"""
                })
                
                for msg in session["messages"]:
                    if msg["role"] in ["user", "assistant"]:
                        messages_for_api.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                messages_for_api.append({
                    "role": "user",
                    "content": request.message
                })
            
            # CONTINUATION HANDLING
            if is_continuation and force_model:
                model_id = force_model
                
                if model_id not in self.llm_use.available_models:
                    print(f"⚠️ Forced model {model_id} not available, using auto-routing")
                    complexity = self.llm_use.router.evaluate(request.message)
                    model_id = self.llm_use._select_best_model(complexity, request.message)
                else:
                    complexity = 5  # Default for continuation
                
                config = self.llm_use.available_models[model_id]
                selection_reason = "Continuation with same model"
                
                await websocket.send_json({
                    "type": "model_selected",
                    "model": config.display_name,
                    "model_id": model_id,
                    "complexity": complexity,
                    "reason": selection_reason,
                    "tier": config.tier if hasattr(config, 'tier') else 'unknown',
                    "is_continuation": True,
                    "estimated_cost": self._estimate_cost(request.message, config)
                })
                
            else:
                # Normal routing logic
                complexity = self.llm_use.router.evaluate(request.message)
                
                if request.model1 and request.model1 != "auto":
                    model_id = request.model1
                    if model_id not in self.llm_use.available_models:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Model {model_id} not available"
                        })
                        return {"status": "error", "error": "Model not available"}
                    
                    config = self.llm_use.available_models[model_id]
                    selection_reason = "Manual selection"
                else:
                    model_id = self.llm_use._select_best_model(complexity, request.message)
                    config = self.llm_use.available_models[model_id]
                    selection_reason = f"Complexity: {complexity}/10"
                
                await websocket.send_json({
                    "type": "model_selected",
                    "model": config.display_name,
                    "model_id": model_id,
                    "complexity": complexity,
                    "reason": selection_reason,
                    "tier": config.tier if hasattr(config, 'tier') else 'unknown',
                    "estimated_cost": self._estimate_cost(request.message, config)
                })
            
            # SAVE LAST USED MODEL
            session['last_model_used'] = model_id
            session['last_model_name'] = config.display_name
            
            # Get provider and make API call
            provider = self.llm_use.providers[config.provider]
            
            print(f"💬 Calling {config.display_name}")
            print(f"   Context messages: {len(messages_for_api)}")
            print(f"   Session history: {len(session['messages'])} messages")
            print(f"   Token estimate: ~{context.token_count} tokens")
            print(f"   Is continuation: {is_continuation}")
            
            if self.llm_use.verbose:
                for i, msg in enumerate(messages_for_api[-3:]):
                    preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    print(f"     [{msg['role']}]: {preview}")
            
            start_time = time.time()
            
            try:
                response = provider.chat(
                    messages=messages_for_api,
                    model=config.model_id,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                elapsed_time = time.time() - start_time
                print(f"✅ Response received in {elapsed_time:.2f}s")
                
            except Exception as api_error:
                print(f"❌ API Error: {str(api_error)}")
                
                if any(word in str(api_error).lower() for word in ["context", "token", "length", "too long"]):
                    print("⚠️ Context too long, trimming...")
                    
                    trimmed_messages = messages_for_api[:1]
                    
                    if len(messages_for_api) > 10:
                        trimmed_messages.extend(messages_for_api[-10:])
                    else:
                        trimmed_messages.extend(messages_for_api[1:])
                    
                    print(f"   Trimmed from {len(messages_for_api)} to {len(trimmed_messages)} messages")
                    
                    try:
                        response = provider.chat(
                            messages=trimmed_messages,
                            model=config.model_id,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens
                        )
                        elapsed_time = time.time() - start_time
                        
                        await websocket.send_json({
                            "type": "warning",
                            "message": "Context was too long and has been trimmed to recent messages"
                        })
                        
                    except Exception as retry_error:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"API failed even after trimming: {str(retry_error)}"
                        })
                        return {"status": "error", "error": str(retry_error)}
                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"API error: {str(api_error)}"
                    })
                    return {"status": "error", "error": str(api_error)}
            
            if request.stream:
                await self._stream_response(response, websocket, "response_token")
            
            context.add_message("assistant", response, model_id)
            
            input_tokens = sum(len(m["content"]) for m in messages_for_api) // 4
            output_tokens = len(response) // 4
            total_tokens = input_tokens + output_tokens
            cost = (input_tokens * config.cost_per_1k_input + 
                output_tokens * config.cost_per_1k_output) / 1000
            
            assistant_message = {
                "role": "assistant",
                "content": response,
                "model": model_id,
                "model_name": config.display_name,
                "timestamp": datetime.now().isoformat(),
                "tokens": {"input": input_tokens, "output": output_tokens},
                "cost": cost,
                "response_time": elapsed_time
            }
            session["messages"].append(assistant_message)
            
            session["stats"]["total_messages"] = len(session["messages"])
            session["stats"]["total_cost"] += cost
            session["stats"]["total_tokens"] += total_tokens
            
            if isinstance(session["stats"]["models_used"], set):
                session["stats"]["models_used"].add(model_id)
            else:
                session["stats"]["models_used"] = {model_id}
            
            self.analytics["model_usage"][model_id] += 1
            self.analytics["provider_usage"][config.provider] += 1
            self.analytics["avg_response_times"][model_id].append(elapsed_time)
            
            if model_id in self.leaderboard:
                times_list = self.analytics["avg_response_times"][model_id]
                self.leaderboard[model_id]["avg_response_time"] = (
                    sum(times_list) / len(times_list) if times_list else 0
                )
                self.leaderboard[model_id]["total_cost"] += cost
                self.leaderboard[model_id]["last_used"] = datetime.now().isoformat()
            
            await websocket.send_json({
                "type": "response_complete",
                "model": config.display_name,
                "model_id": model_id,
                "response": response,
                "session_id": request.session_id,
                "is_truncated": len(response) >= (request.max_tokens * 3),  # Rough estimate
                "stats": {
                    "response_time": round(elapsed_time, 2),
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": total_tokens
                    },
                    "cost": round(cost, 6),
                    "context_length": len(messages_for_api),
                    "session_totals": {
                        "messages": len(session["messages"]),
                        "cost": round(session["stats"]["total_cost"], 6),
                        "tokens": session["stats"]["total_tokens"],
                        "models_used": len(session["stats"]["models_used"])
                    },
                    "conversation_turns": len([m for m in session["messages"] if m["role"] == "user"])
                }
            })
            
            print(f"📊 Session now has {len(session['messages'])} messages total")
            print(f"   Context is tracking {len(context.messages)} messages")
            
            return {
                "status": "success",
                "model": model_id,
                "model_name": config.display_name,
                "response": response,
                "cost": cost,
                "tokens": total_tokens,
                "session_id": request.session_id
            }
            
        except Exception as e:
            error_msg = f"Error in chat processing: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            self.analytics["error_count"] += 1
            
            await websocket.send_json({
                "type": "error",
                "error": error_msg,
                "traceback": traceback.format_exc() if request.session_id == "debug" else None
            })
            
            return {"status": "error", "error": error_msg}

# ====================
# FASTAPI APP
# ====================

app = FastAPI(
    title="LLM-USE Arena",
    version="2.0.0",
    description="Advanced LLM comparison and battle arena"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backend
arena_backend = EnhancedArenaBackend()

# Background task for cleanup
async def periodic_cleanup():
    """Run periodic cleanup tasks"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        arena_backend.cleanup_old_sessions()

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(periodic_cleanup())

# ====================
# API ENDPOINTS
# ====================

@app.get("/")
async def root():
    """Serve main HTML page"""
    html_path = Path("home.html")
    if not html_path.exists():
        return HTMLResponse(content=create_default_html())
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/api/models")
async def get_models(
    provider: Optional[str] = None,
    min_quality: Optional[int] = None,
    max_cost: Optional[float] = None
):
    """Get available models with optional filters"""
    filters = {}
    if provider:
        filters["provider"] = provider
    if min_quality:
        filters["min_quality"] = min_quality
    if max_cost:
        filters["max_cost"] = max_cost
    
    return arena_backend.get_available_models(filters)

@app.get("/api/leaderboard")
async def get_leaderboard(min_battles: int = 0):
    """Get model leaderboard"""
    return arena_backend.get_leaderboard(min_battles)

@app.get("/api/analytics")
async def get_analytics():
    """Get system analytics"""
    return arena_backend.get_analytics()

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = arena_backend.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Convert sets to lists for JSON serialization
    session_copy = session.copy()
    if "models_used" in session_copy["stats"]:
        session_copy["stats"]["models_used"] = list(session_copy["stats"]["models_used"])
    
    return session_copy

@app.get("/api/session/{session_id}/export")
async def export_session(session_id: str):
    """Export session as JSON"""
    session = arena_backend.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Prepare export data
    export_data = {
        "session": session,
        "timestamp": datetime.now().isoformat(),
        "messages_count": len(session["messages"]),
        "battles_count": len(session["battles"])
    }
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename=session_{session_id}.json"
        }
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time chat"""
    await websocket.accept()
    arena_backend.connections[client_id] = websocket
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process based on type
            if data["type"] == "chat":
                request = ChatRequest(**data["payload"])
                result = await arena_backend.process_chat(request, websocket)
                
            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
            elif data["type"] == "get_stats":
                stats = arena_backend.get_analytics()
                await websocket.send_json({"type": "stats", "data": stats})
                
    except WebSocketDisconnect:
        del arena_backend.connections[client_id]
        print(f"Client {client_id} disconnected")
        
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })

# ====================
# MAIN ENTRY POINT
# ====================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = "0.0.0.0"
    port = 8000
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print("\n" + "="*60)
    print("🎮 LLM-USE ARENA SERVER v2.0")
    print("="*60)
    print(f"📡 Starting server on http://{host}:{port}")
    print(f"📊 WebSocket endpoint: ws://{host}:{port}/ws/{{client_id}}")
    print(f"🌐 Open http://localhost:{port} in your browser")
    print("="*60 + "\n")
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
