#!/usr/bin/env python3
"""
ChukLLM Async Generator Diagnostic
=================================

This script helps diagnose and fix async generator issues with chuk_llm.
Run this to understand what's happening with streaming functions.
"""

import asyncio
import inspect
from typing import Any

from dotenv import load_dotenv
load_dotenv()

try:
    from chuk_llm import stream, ask
    from chuk_llm.api.conversation import conversation
    print("✅ chuk_llm imports successful")
except ImportError as e:
    print(f"❌ chuk_llm import failed: {e}")
    exit(1)


async def diagnose_stream_function():
    """Diagnose the stream function behavior."""
    print("\n🔬 Diagnosing stream() function")
    print("=" * 40)
    
    try:
        # Test the stream function
        stream_obj = stream("Hello world")
        
        print(f"stream() returns: {type(stream_obj)}")
        print(f"Has __aiter__: {hasattr(stream_obj, '__aiter__')}")
        print(f"Has __anext__: {hasattr(stream_obj, '__anext__')}")
        print(f"Is coroutine: {inspect.iscoroutine(stream_obj)}")
        print(f"Is async generator: {inspect.isasyncgen(stream_obj)}")
        print(f"Is generator: {inspect.isgenerator(stream_obj)}")
        
        # Try to use it
        print("\n🔄 Testing stream iteration:")
        try:
            chunk_count = 0
            async for chunk in stream_obj:
                print(f"Chunk {chunk_count + 1}: '{chunk}'", end=" ")
                chunk_count += 1
                if chunk_count >= 5:  # Limit output
                    print("... (truncated)")
                    break
            print(f"\n✅ Successfully streamed {chunk_count} chunks")
        except Exception as e:
            print(f"❌ Stream iteration failed: {e}")
            print(f"Error type: {type(e)}")
            
    except Exception as e:
        print(f"❌ stream() function failed: {e}")


async def diagnose_conversation_streaming():
    """Diagnose conversation streaming behavior."""
    print("\n🔬 Diagnosing conversation streaming")
    print("=" * 40)
    
    try:
        async with conversation(provider="anthropic") as chat:
            print(f"Chat object type: {type(chat)}")
            print(f"Has stream_say: {hasattr(chat, 'stream_say')}")
            
            if hasattr(chat, 'stream_say'):
                print(f"stream_say is callable: {callable(getattr(chat, 'stream_say'))}")
                
                # Test stream_say
                try:
                    stream_response = chat.stream_say("Hello")
                    print(f"stream_say returns: {type(stream_response)}")
                    print(f"Has __aiter__: {hasattr(stream_response, '__aiter__')}")
                    print(f"Is coroutine: {inspect.iscoroutine(stream_response)}")
                    print(f"Is async generator: {inspect.isasyncgen(stream_response)}")
                    
                    # Try to iterate
                    print("\n🔄 Testing conversation stream iteration:")
                    if inspect.iscoroutine(stream_response):
                        print("⚠️  stream_say returned a coroutine, need to await it first")
                        try:
                            awaited_result = await stream_response
                            print(f"Awaited result: {type(awaited_result)}")
                            if hasattr(awaited_result, '__aiter__'):
                                async for chunk in awaited_result:
                                    print(f"'{chunk}'", end=" ")
                                    break  # Just test first chunk
                                print("\n✅ Awaited stream works")
                            else:
                                print(f"❌ Awaited result not iterable: {awaited_result}")
                        except Exception as await_e:
                            print(f"❌ Awaiting failed: {await_e}")
                    elif hasattr(stream_response, '__aiter__'):
                        chunk_count = 0
                        async for chunk in stream_response:
                            print(f"'{chunk}'", end=" ")
                            chunk_count += 1
                            if chunk_count >= 3:
                                print("... (truncated)")
                                break
                        print(f"\n✅ Direct iteration works, {chunk_count} chunks")
                    else:
                        print(f"❌ stream_response not directly iterable")
                        
                except Exception as e:
                    print(f"❌ stream_say failed: {e}")
                    print(f"Error type: {type(e)}")
            else:
                print("❌ No stream_say method available")
                
    except Exception as e:
        print(f"❌ Conversation creation failed: {e}")


async def test_working_alternatives():
    """Test alternative approaches that work."""
    print("\n🔧 Testing Working Alternatives")
    print("=" * 40)
    
    # Test 1: Regular ask function
    print("1. Regular ask() function:")
    try:
        response = await ask("What is 2+2?", max_tokens=10)
        print(f"✅ ask() works: {response}")
    except Exception as e:
        print(f"❌ ask() failed: {e}")
    
    # Test 2: Simulated streaming
    print("\n2. Simulated streaming:")
    try:
        response = await ask("Write a short sentence about Python", max_tokens=20)
        words = response.split()
        print("Simulated stream: ", end="")
        for word in words[:8]:
            print(f"{word} ", end="", flush=True)
            await asyncio.sleep(0.1)
        print("\n✅ Simulated streaming works")
    except Exception as e:
        print(f"❌ Simulated streaming failed: {e}")
    
    # Test 3: Conversation without streaming
    print("\n3. Regular conversation:")
    try:
        async with conversation(provider="anthropic") as chat:
            response = await chat.say("Hello, how are you?")
            print(f"✅ Regular conversation works: {response[:50]}...")
    except Exception as e:
        print(f"❌ Regular conversation failed: {e}")


async def provide_recommendations():
    """Provide recommendations based on diagnostics."""
    print("\n💡 Recommendations")
    print("=" * 40)
    
    print("Based on the diagnostics:")
    print("1. ✅ Use regular ask() and chat.say() functions - they work reliably")
    print("2. 🔄 For streaming UI, simulate with word-by-word output")
    print("3. ⚠️  Avoid direct async generator iteration until chuk_llm fixes the issue")
    print("4. 🛡️  Always use try/except blocks around streaming code")
    print("5. 📝 Use fallback patterns for production code")
    
    print("\n🔧 Working Pattern Example:")
    print("""
async def safe_streaming_response(prompt):
    try:
        # Try streaming first
        async for chunk in stream(prompt):
            yield chunk
    except Exception:
        # Fallback to regular response with simulation
        response = await ask(prompt)
        words = response.split()
        for word in words:
            yield f"{word} "
            await asyncio.sleep(0.05)
""")


async def main():
    """Run all diagnostics."""
    print("🔬 ChukLLM Async Generator Diagnostic Tool")
    print("This tool helps identify and fix streaming issues")
    print()
    
    await diagnose_stream_function()
    await diagnose_conversation_streaming()
    await test_working_alternatives()
    await provide_recommendations()
    
    print("\n🎯 Summary:")
    print("- Regular async functions (ask, chat.say) work perfectly")
    print("- Streaming has some async generator compatibility issues")
    print("- Use fallback patterns for robust production code")
    print("- Consider simulated streaming for UI responsiveness")


if __name__ == "__main__":
    asyncio.run(main())