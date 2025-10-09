#!/usr/bin/env python3
"""
Production monitoring example for LLM-USE
"""

from llm_use import LLMUSEV2
import time
import json

def simulate_production_load():
    """Simulate production traffic and monitor metrics"""
    
    print("🏭 Production Monitoring Example\n")
    
    # Initialize with production features
    router = LLMUSEV2(
        verbose=False,
        enable_production=True
    )
    
    # Simulate various requests
    test_queries = [
        ("What's 2+2?", "simple"),
        ("Explain machine learning", "moderate"),
        ("Debug this Python code: def factorial(n): return n * factorial(n-1)", "complex"),
        ("Hello!", "trivial"),
        ("Design a REST API", "complex"),
    ]
    
    print("📊 Simulating production traffic...\n")
    
    for query, complexity in test_queries:
        print(f"[{complexity.upper()}] Processing: '{query[:40]}...'")
        
        start = time.time()
        response = router.chat(query)
        elapsed = time.time() - start
        
        print(f"  ✓ Completed in {elapsed:.2f}s\n")
        time.sleep(0.5)  # Simulate time between requests
    
    # Get analytics
    print("\n" + "="*60)
    print("📈 PRODUCTION ANALYTICS")
    print("="*60)
    
    # Overall stats
    stats = router.get_stats()
    print("\n📊 Session Statistics:")
    print(f"  • Total messages: {stats['total_messages']}")
    print(f"  • Total cost: ${stats['total_cost']:.6f}")
    print(f"  • Models used: {len(stats['models_used'])}")
    
    # Cache performance
    if 'cache' in stats:
        cache = stats['cache']
        print(f"\n💾 Cache Performance:")
        print(f"  • Hit rate: {cache['hit_rate']*100:.1f}%")
        print(f"  • Size: {cache['current_size_mb']:.2f} MB")
        print(f"  • Entries: {cache['entries']}")
    
    # Metrics summary
    if 'metrics' in stats:
        metrics = stats['metrics']
        print(f"\n⚡ Performance Metrics:")
        print(f"  • Avg latency: {metrics['avg_latency_ms']:.0f}ms")
        print(f"  • Error rate: {metrics['error_rate']*100:.2f}%")
        print(f"  • Total tokens: {metrics['total_tokens']:,}")
    
    # Model distribution
    if stats['models_used']:
        print(f"\n🤖 Model Usage:")
        for model, count in stats['models_used'].items():
            config = router.available_models.get(model)
            if config:
                print(f"  • {config.display_name}: {count} requests")
    
    # Get 24h analytics if available
    analytics = router.get_analytics(hours=24)
    if analytics and 'model_performance' in analytics:
        print(f"\n📊 24-Hour Model Performance:")
        for perf in analytics['model_performance'][:3]:
            print(f"  • {perf['model']}: {perf['total_calls']} calls, "
                  f"{perf['avg_latency']:.0f}ms avg, "
                  f"{perf['success_rate']:.1f}% success")

def main():
    simulate_production_load()

if __name__ == "__main__":
    main()