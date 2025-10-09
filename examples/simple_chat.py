#!/usr/bin/env python3
"""
Simple chat example using LLM-USE
"""

from llm_use import LLMUSEV2

def main():
    # Initialize router
    print("🚀 Initializing LLM-USE...")
    router = LLMUSEV2(verbose=True)
    
    print("\n💬 Chat started (type 'quit' to exit)\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Get response
        response = router.chat(user_input)
        print(f"Assistant: {response}\n")
    
    # Show final stats
    stats = router.get_stats()
    print(f"\n📊 Final Stats:")
    print(f"   Messages: {stats['total_messages']}")
    print(f"   Total cost: ${stats['total_cost']:.6f}")

if __name__ == "__main__":
    main()