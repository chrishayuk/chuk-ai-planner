#!/usr/bin/env python3
"""
ChukLLM QuickStart Demo - Async Features
========================================

This demo showcases ChukLLM's async capabilities including concurrent requests,
streaming responses, and high-performance patterns for production applications.
"""

import asyncio
import time
from chuk_llm import (
    # Core async functions
    ask, stream,
    
    # Async provider functions
    ask_openai, ask_anthropic, ask_groq,
    
    # Model-specific async functions
    ask_openai_gpt4o_mini,
    ask_anthropic_claude_sonnet4_20250514,
    ask_anthropic_sonnet,  # Alias
    
    # Configuration
    configure,
)

# Import conversation separately to avoid the callable issue
from chuk_llm.api.conversation import conversation


async def demo_basic_async():
    """Demonstrate basic async functionality."""
    print("🚀 Basic Async Demo")
    print("=" * 40)
    
    # 1. Basic async call
    print("\n1️⃣ Basic async call:")
    response = await ask("What is 2 + 2?")
    print(f"   Q: What is 2 + 2?")
    print(f"   A: {response}")
    
    # 2. Provider-specific async calls
    print("\n2️⃣ Provider-specific async calls:")
    try:
        openai_response = await ask_openai("Tell me a quick fact about space")
        print(f"   🔹 OpenAI: {openai_response[:80]}...")
    except Exception as e:
        print(f"   ❌ OpenAI: {str(e)[:50]}...")
    
    try:
        anthropic_response = await ask_anthropic("Tell me a quick fact about the ocean")
        print(f"   🔹 Anthropic: {anthropic_response[:80]}...")
    except Exception as e:
        print(f"   ❌ Anthropic: {str(e)[:50]}...")
    
    # 3. Model-specific async calls
    print("\n3️⃣ Model-specific async calls:")
    try:
        gpt4o_response = await ask_openai_gpt4o_mini("What's the capital of France?")
        print(f"   🔹 GPT-4o Mini: {gpt4o_response}")
    except Exception as e:
        print(f"   ❌ GPT-4o Mini: {str(e)[:50]}...")
    
    try:
        sonnet_response = await ask_anthropic_sonnet("What's the capital of Germany?")
        print(f"   🔹 Claude Sonnet: {sonnet_response}")
    except Exception as e:
        print(f"   ❌ Claude Sonnet: {str(e)[:50]}...")
    
    print("\n✅ Basic async demo complete!")


async def demo_concurrent_requests():
    """Demonstrate concurrent requests for performance."""
    print("\n⚡ Concurrent Requests Demo")
    print("=" * 40)
    
    questions = [
        "What's the capital of France?",
        "What's 15 * 23?", 
        "Name a programming language",
        "What color is the sky?",
        "What's the largest planet?"
    ]
    
    # Sequential approach (slow)
    print("\n📊 Sequential requests:")
    start_time = time.time()
    sequential_responses = []
    for question in questions:
        response = await ask(question, max_tokens=20)
        sequential_responses.append(response)
    sequential_time = time.time() - start_time
    print(f"   ⏱️ Sequential time: {sequential_time:.2f}s")
    for i, response in enumerate(sequential_responses[:2]):  # Show first 2
        print(f"   {i+1}. {response}")
    
    # Concurrent approach (fast!)
    print("\n🚀 Concurrent requests:")
    start_time = time.time()
    
    # Create tasks for all requests
    tasks = [ask(question, max_tokens=20) for question in questions]
    
    # Run them concurrently
    concurrent_responses = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    
    print(f"   ⚡ Concurrent time: {concurrent_time:.2f}s")
    print(f"   🎯 Speedup: {sequential_time/concurrent_time:.1f}x faster!")
    
    for i, response in enumerate(concurrent_responses[:2]):  # Show first 2
        print(f"   {i+1}. {response}")
    
    print("\n✅ Concurrent requests demo complete!")


async def demo_streaming_responses():
    """Demonstrate streaming responses."""
    print("\n🌊 Streaming Responses Demo")
    print("=" * 40)
    
    # 1. Test what type of object stream returns
    print("\n🔍 Debugging stream function:")
    try:
        from chuk_llm.api.core import stream
        stream_obj = stream("test")
        print(f"   Stream returns: {type(stream_obj)}")
        print(f"   Has __aiter__: {hasattr(stream_obj, '__aiter__')}")
    except Exception as e:
        print(f"   Debug error: {e}")
    
    # 2. Basic streaming with proper error handling
    print("\n📝 Basic streaming:")
    print("🤖 Assistant: ", end="", flush=True)
    
    total_chars = 0
    start_time = time.time()
    
    try:
        # Try the direct approach first
        chunk_count = 0
        async for chunk in stream("Write a short haiku about coding"):
            print(chunk, end="", flush=True)
            total_chars += len(chunk)
            chunk_count += 1
            if chunk_count > 20:  # Safety break
                break
                
        print(f" (✅ {chunk_count} chunks)")
        
    except Exception as e:
        print(f"[Direct streaming failed: {e}]")
        
        # Fallback: simulate streaming with regular ask
        try:
            print("\n🔄 Fallback - simulated streaming: ", end="", flush=True)
            response = await ask("Write a short haiku about coding", max_tokens=30)
            
            # Simulate streaming by showing words one by one
            words = response.split()
            for word in words[:8]:  # Limit to first 8 words
                print(f"{word} ", end="", flush=True)
                await asyncio.sleep(0.1)  # Simulate delay
                total_chars += len(word) + 1
            
            print("(simulated)")
            
        except Exception as fallback_e:
            print(f"[Fallback failed: {fallback_e}]")
            total_chars = 50
    
    stream_time = time.time() - start_time
    print(f"\n   📊 Output {total_chars} chars in {stream_time:.2f}s")
    
    # 3. Test provider-specific streaming
    print("\n🎯 Provider-specific streaming test:")
    try:
        from chuk_llm import stream_openai
        print("   OpenAI streaming: ", end="", flush=True)
        
        chunk_count = 0
        async for chunk in stream_openai("Say hello", max_tokens=5):
            print(chunk, end="", flush=True)
            chunk_count += 1
            if chunk_count > 10:
                break
        
        print(f" (✅ {chunk_count} chunks)")
        
    except Exception as e:
        print(f"[Provider streaming error: {e}]")
    
    print("\n✅ Streaming demo complete!")


async def demo_provider_comparison():
    """Compare providers asynchronously."""
    print("\n⚖️ Async Provider Comparison Demo")
    print("=" * 40)
    
    question = "In exactly 10 words, explain artificial intelligence."
    
    # Define provider tasks
    provider_tasks = {
        "OpenAI": ask_openai(question),
        "Anthropic": ask_anthropic(question),
    }
    
    # Try adding Groq if available
    try:
        provider_tasks["Groq"] = ask_groq(question)
    except Exception:
        pass  # Groq not available or not configured
    
    print(f"\n🔍 Question: {question}")
    print("\n🏃‍♂️ Running concurrent requests...")
    
    start_time = time.time()
    
    # Run all providers concurrently
    results = await asyncio.gather(
        *provider_tasks.values(),
        return_exceptions=True  # Don't fail if one provider fails
    )
    
    total_time = time.time() - start_time
    
    # Display results
    for provider_name, result in zip(provider_tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"\n❌ {provider_name}: {str(result)[:60]}...")
        else:
            print(f"\n✅ {provider_name}: {result}")
    
    print(f"\n⏱️ All providers responded in {total_time:.2f}s")
    print("\n✅ Provider comparison demo complete!")


async def demo_async_conversations():
    """Demonstrate async conversations."""
    print("\n💬 Async Conversations Demo")
    print("=" * 40)
    
    # Single conversation
    print("\n🔹 Basic async conversation:")
    try:
        async with conversation(provider="anthropic") as chat:
            response1 = await chat.say("Hi! I'm interested in space exploration.")
            print(f"   👤 User: Hi! I'm interested in space exploration.")
            print(f"   🤖 Assistant: {response1[:100]}...")
            
            response2 = await chat.say("What should I study?")
            print(f"   👤 User: What should I study?")
            print(f"   🤖 Assistant: {response2[:100]}...")
    except Exception as e:
        print(f"   ❌ Conversation error: {str(e)[:60]}...")
    
    # Multiple concurrent conversations
    print("\n🔀 Concurrent conversations:")
    
    async def topic_conversation(topic, provider="openai"):
        """Have a conversation about a specific topic."""
        try:
            async with conversation(provider=provider) as chat:
                await chat.say(f"I want to learn about {topic}")
                response = await chat.say("Give me one key thing to know")
                return f"{topic}: {response[:60]}..."
        except Exception as e:
            return f"{topic}: Error - {str(e)[:40]}..."
    
    topics = ["Python programming", "machine learning", "quantum physics"]
    
    start_time = time.time()
    conversation_tasks = [topic_conversation(topic) for topic in topics]
    conversation_results = await asyncio.gather(*conversation_tasks, return_exceptions=True)
    conversation_time = time.time() - start_time
    
    for result in conversation_results:
        if isinstance(result, Exception):
            print(f"   ❌ Error: {str(result)[:60]}...")
        else:
            print(f"   🎯 {result}")
    
    print(f"   ⚡ All conversations completed in {conversation_time:.2f}s")
    
    print("\n✅ Async conversations demo complete!")


async def demo_streaming_conversation():
    """Demonstrate streaming in conversations."""
    print("\n🌊 Streaming Conversation Demo")
    print("=" * 40)
    
    try:
        async with conversation(provider="anthropic") as chat:
            # Set up context
            await chat.say("I'm learning about async programming in Python.")
            
            print("\n👤 User: Can you explain async/await in a simple way?")
            print("🤖 Assistant: ", end="", flush=True)
            
            # Final fix: Based on diagnostics, this should work
            streaming_worked = False
            
            try:
                # Get the async generator directly
                stream_gen = chat.stream_say("Can you explain async/await in a simple way? Keep it concise.")
                
                # Iterate without awaiting
                chunk_count = 0
                async for chunk in stream_gen:
                    if chunk and chunk != "[Streaming error: object async_generator can't be used in 'await' expression]":
                        print(chunk, end="", flush=True)
                        chunk_count += 1
                        streaming_worked = True
                    if chunk_count > 50:  # Safety
                        break
                
                if streaming_worked:
                    print(f" (✅ {chunk_count} chunks)")
                else:
                    # If we only got error messages, fall back
                    raise Exception("Only error messages received")
                    
            except Exception as e:
                if not streaming_worked:
                    print(f"\n🔄 Fallback to simulated streaming...")
                    
                    # Reliable fallback
                    response = await chat.say("Can you explain async/await in a simple way? Keep it concise.")
                    
                    # Simulate streaming for better UX
                    words = response.split()
                    print("   ", end="", flush=True)  # Indent for readability
                    
                    for i, word in enumerate(words[:25]):  # First 25 words
                        print(f"{word} ", end="", flush=True)
                        await asyncio.sleep(0.04)  # Smooth streaming feel
                    
                    if len(words) > 25:
                        print("... (truncated for demo)")
                    else:
                        print()
                    
                    print("✅ Simulated streaming completed")
                
    except Exception as e:
        print(f"\n❌ Conversation setup failed: {str(e)[:60]}...")
    
    print("\n✅ Streaming conversation demo complete!")


async def demo_error_handling():
    """Demonstrate async error handling."""
    print("\n🛡️ Async Error Handling Demo")
    print("=" * 40)
    
    # Test timeout handling
    print("\n⏱️ Testing with very short max_tokens:")
    try:
        response = await ask("Write a long essay about the universe", max_tokens=5)
        print(f"   ✅ Short response: {response}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:60]}...")
    
    # Test provider error handling
    print("\n🔧 Testing invalid provider:")
    try:
        response = await ask("Hello", provider="nonexistent")
        print(f"   Unexpected success: {response}")
    except Exception as e:
        print(f"   ✅ Graceful error: {str(e)[:60]}...")
    
    # Test concurrent error handling
    print("\n🏃‍♂️ Testing concurrent with some failures:")
    tasks = [
        ask("Hello", provider="openai"),  # Should work
        ask("Hello", provider="nonexistent"),  # Should fail
        ask("Hello", provider="anthropic"),  # Should work
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"   ❌ Task {i+1}: {str(result)[:50]}...")
        else:
            print(f"   ✅ Task {i+1}: {result[:30]}...")
    
    print("\n✅ Error handling demo complete!")


async def demo_performance_patterns():
    """Demonstrate high-performance async patterns."""
    print("\n🏎️ Performance Patterns Demo")
    print("=" * 40)
    
    # Pattern 1: Batching requests
    print("\n📦 Batch processing pattern:")
    
    async def process_batch(questions, batch_size=3):
        """Process questions in batches to avoid overwhelming the API."""
        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_tasks = [ask(q, max_tokens=15) for q in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Small delay between batches (good practice for rate limiting)
            if i + batch_size < len(questions):
                await asyncio.sleep(0.1)
        
        return results
    
    questions = [f"What is {i}+{i}?" for i in range(1, 8)]
    
    start_time = time.time()
    batch_results = await process_batch(questions, batch_size=3)
    batch_time = time.time() - start_time
    
    successful = sum(1 for r in batch_results if not isinstance(r, Exception))
    print(f"   ✅ Processed {successful}/{len(questions)} questions in {batch_time:.2f}s")
    
    # Pattern 2: Timeout handling
    print("\n⏰ Timeout handling pattern:")
    
    async def ask_with_timeout(question, timeout=5.0):
        """Ask with a timeout to prevent hanging."""
        try:
            return await asyncio.wait_for(ask(question, max_tokens=20), timeout=timeout)
        except asyncio.TimeoutError:
            return "Error: Request timed out"
    
    timeout_result = await ask_with_timeout("What is the meaning of life?", timeout=10.0)
    print(f"   ⏱️ Timeout result: {timeout_result[:50]}...")
    
    # Pattern 3: Parallel model comparison
    print("\n🔄 Parallel model comparison pattern:")
    
    async def compare_models(question):
        """Compare responses from different models in parallel."""
        model_tasks = {
            "GPT-4o Mini": ask_openai_gpt4o_mini(question),
            "Claude Sonnet": ask_anthropic_sonnet(question),
        }
        
        results = await asyncio.gather(*model_tasks.values(), return_exceptions=True)
        
        comparison = {}
        for model_name, result in zip(model_tasks.keys(), results):
            if isinstance(result, Exception):
                comparison[model_name] = f"Error: {str(result)[:30]}..."
            else:
                comparison[model_name] = result[:50] + "..." if len(result) > 50 else result
        
        return comparison
    
    comparison_result = await compare_models("What's 5+5?")
    for model, response in comparison_result.items():
        print(f"   🔹 {model}: {response}")
    
    print("\n✅ Performance patterns demo complete!")


async def demo_model_specific_functions():
    """Demonstrate model-specific async functions."""
    print("\n🎯 Model-Specific Functions Demo")
    print("=" * 40)
    
    question = "What's 2+2? Explain your reasoning."
    
    print("🤖 OpenAI Models:")
    print("-" * 40)
    
    try:
        print("GPT-4o Mini:")
        response = await ask_openai_gpt4o_mini(question)
        print(f"   {response}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")
    
    print("\n🧠 Anthropic Models:")
    print("-" * 40)
    
    try:
        print("Claude Sonnet 4:")
        response = await ask_anthropic_claude_sonnet4_20250514(question)
        print(f"   {response}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")
    
    # Using aliases for convenience
    print("\n✨ Using Convenient Aliases:")
    print("-" * 40)
    
    try:
        print("Anthropic 'sonnet' alias:")
        response = await ask_anthropic_sonnet(question)
        print(f"   {response}")
        print()
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")
    
    # Parallel requests example
    print("\n⚡ Parallel Model Requests Example:")
    print("-" * 40)
    print("Asking both models simultaneously...")
    
    # Create tasks for parallel execution
    tasks = [
        ask_openai_gpt4o_mini("What's 3+3?"),
        ask_anthropic_sonnet("What's 4+4?")
    ]
    
    # Wait for both to complete
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(["OpenAI", "Anthropic"]):
            if isinstance(results[i], Exception):
                print(f"   ❌ {result}: {str(results[i])[:50]}...")
            else:
                print(f"   ✅ {result}: {results[i]}")
    except Exception as e:
        print(f"   ❌ Parallel execution failed: {str(e)[:50]}...")
    
    print("\n✅ Model-specific functions demo complete!")


async def main():
    """Run all async demos."""
    print("🎯 ChukLLM QuickStart - Async Features")
    print("This demo shows ChukLLM's async capabilities for high-performance applications!")
    print("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables")
    print()
    
    try:
        await demo_basic_async()
        await demo_concurrent_requests()
        await demo_streaming_responses()
        await demo_provider_comparison()
        await demo_async_conversations()
        await demo_streaming_conversation()
        await demo_error_handling()
        await demo_performance_patterns()
        await demo_model_specific_functions()
        
        print("\n🎉 All async demos completed successfully!")
        print("\n💡 Key async advantages:")
        print("   • Concurrent requests: Up to 5x faster than sequential")
        print("   • Streaming responses: Real-time output as it's generated")
        print("   • Non-blocking operations: Perfect for web applications")
        print("   • Error resilience: Individual failures don't block others")
        print("   • Resource efficiency: Better CPU and memory usage")
        print("   • Scalability: Handle hundreds of concurrent requests")
        print("\n🚀 Perfect for:")
        print("   • Web applications (FastAPI, Django Async)")
        print("   • Chatbots and interactive systems")
        print("   • Batch processing and data analysis")
        print("   • Real-time streaming applications")
        print("   • High-throughput production systems")
        print("\n🎯 Model-specific functions available:")
        print("   • ask_openai_gpt4o_mini() - Fast, efficient OpenAI model")
        print("   • ask_anthropic_claude_sonnet4_20250514() - Latest Claude Sonnet")
        print("   • ask_anthropic_sonnet() - Convenient alias for Sonnet")
        print("   • Plus many more provider and model-specific functions!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n💡 Make sure you have API keys configured:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'")


if __name__ == "__main__":
    asyncio.run(main())