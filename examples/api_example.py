#!/usr/bin/env python3
"""
API usage example for LLM-USE
"""

from llm_use import LLMUSEV2
import json

def analyze_text_complexity():
    """Example: Analyze how different texts are routed"""
    
    router = LLMUSEV2(verbose=False)
    
    test_cases = [
        "Hi!",
        "What's the weather like?",
        "Explain the theory of relativity",
        "Write a Python function to implement quicksort",
        "Design a microservices architecture for an e-commerce platform",
    ]
    
    print("🔍 Text Complexity Analysis\n")
    
    for text in test_cases:
        # This will trigger routing decision
        model = router.evaluate_and_route(text)
        config = router.available_models[model]
        
        print(f"Text: '{text[:50]}...' if len(text) > 50 else '{text}'")
        print(f"  → Model: {config.display_name}")
        print(f"  → Quality: {config.quality_score}/10")
        print(f"  → Speed: {config.speed}\n")

def cost_comparison():
    """Example: Compare costs across different models"""
    
    router = LLMUSEV2(verbose=False)
    
    test_prompt = "Write a blog post about AI"
    
    print("💰 Cost Comparison\n")
    
    # Track costs for different complexity levels
    for complexity in [2, 5, 8]:
        model = router._select_best_model(complexity, test_prompt)
        config = router.available_models[model]
        
        # Estimate cost (assuming ~500 tokens output)
        est_cost = (100 * config.cost_per_1k_input + 500 * config.cost_per_1k_output) / 1000
        
        print(f"Complexity {complexity}/10:")
        print(f"  Model: {config.display_name}")
        print(f"  Estimated cost: ${est_cost:.6f}")
        print(f"  Speed: {config.speed}\n")

def main():
    print("="*60)
    analyze_text_complexity()
    print("="*60)
    cost_comparison()
    print("="*60)

if __name__ == "__main__":
    main()
