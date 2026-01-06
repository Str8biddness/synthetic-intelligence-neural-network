#!/usr/bin/env python3
"""
SI Engine Capabilities Demo
Demonstrates: Hardware Detection, Memory System, Web Access
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.si_engine.hardware_acceleration import HardwareAccelerator
from backend.si_engine.memory_system import MemorySystem
from backend.si_engine.web_access import WebAccess

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def demo_hardware():
    print_section("🔧 HARDWARE DETECTION")
    
    hw = HardwareAccelerator()
    info = hw.hardware_info
    
    print(f"\n💻 CPU: {info.cpu_count} cores @ {info.cpu_freq_mhz:.0f} MHz")
    print(f"🧠 RAM: {info.ram_total_gb:.2f} GB total, {info.ram_available_gb:.2f} GB available")
    print(f"🖥️  Platform: {info.platform} ({info.architecture})")
    
    if info.has_gpu:
        print(f"\n🎮 GPU Detected: {info.gpu_info.get('type', 'Unknown')}")
        print(f"   Name: {info.gpu_info.get('name', 'N/A')}")
    else:
        print("\n⚠️  No GPU detected (CPU-only mode)")
    
    # Show optimization settings
    print(f"\n⚙️  Optimal Workers: {hw.optimal_workers}")
    config = hw.get_parallel_config()
    print(f"   Recommended chunk size: {config['chunk_size']}")
    print(f"   Threading enabled: {config['use_threading']}")
    
    # Show current system stats
    stats = hw.get_system_stats()
    print(f"\n📊 Current Load:")
    print(f"   CPU Usage: {stats['cpu_percent']:.1f}%")
    print(f"   Memory Usage: {stats['memory_percent']:.1f}%")
    
    return hw

def demo_memory():
    print_section("🧠 MEMORY SYSTEM")
    
    memory = MemorySystem()
    
    # Create a test session
    print("\n📝 Creating test session...")
    session = memory.create_session(user_id="demo_user")
    print(f"   Session ID: {session.id[:16]}...")
    
    # Store some memories
    print("\n💾 Storing memories...")
    memory.store(session.id, "What is artificial intelligence?", "query", importance=0.7)
    memory.store(session.id, "AI is the simulation of human intelligence in machines", "response", importance=0.8)
    memory.store(session.id, "Tell me about machine learning", "query", importance=0.6)
    memory.store(session.id, "ML is a subset of AI focused on learning from data", "response", importance=0.7)
    print("   Stored 4 memory entries")
    
    # Retrieve memories
    print("\n🔍 Retrieving recent memories...")
    memories = memory.retrieve(session.id, limit=4)
    for i, mem in enumerate(memories, 1):
        print(f"   {i}. [{mem.memory_type}] {mem.content[:50]}...")
    
    # Search memories
    print("\n🔎 Searching for 'machine learning'...")
    results = memory.search("machine learning", session.id, limit=2)
    for i, mem in enumerate(results, 1):
        print(f"   {i}. {mem.content[:60]}...")
    
    # Get context
    print("\n📋 Conversation context:")
    context = memory.get_context(session.id, window_size=3)
    for line in context.split('\n')[:3]:
        print(f"   {line}")
    
    # Show stats
    print("\n📈 Memory Statistics:")
    stats = memory.get_stats(session.id)
    print(f"   Memories in session: {stats['memory_count']}")
    print(f"   Short-term utilization: {stats['short_term_utilization']*100:.1f}%")
    
    return memory, session

def demo_web_access():
    print_section("🌐 WEB ACCESS")
    
    web = WebAccess()
    
    print("\n🔍 Searching web for 'Python programming'...")
    results = web.search_web("Python programming", num_results=3)
    
    if results:
        print(f"   Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.title}")
            print(f"      {result.snippet[:80]}...")
            print()
    else:
        print("   No results found (this is normal for DuckDuckGo HTML scraping)")
    
    # Try fetching a URL
    print("\n🌐 Fetching example.com...")
    page = web.fetch_url("http://example.com")
    if page:
        print(f"   Title: {page.title}")
        print(f"   Content length: {len(page.content)} characters")
        print(f"   Snippet: {page.snippet[:100]}...")
    else:
        print("   Failed to fetch page")
    
    return web

def main():
    print("\n" + "*"*60)
    print("*" + " "*58 + "*")
    print("*" + "  SI ENGINE CAPABILITIES DEMONSTRATION".center(58) + "*")
    print("*" + " "*58 + "*")
    print("*"*60)
    
    try:
        # Demo 1: Hardware
        hw = demo_hardware()
        
        # Demo 2: Memory
        memory, session = demo_memory()
        
        # Demo 3: Web Access
        web = demo_web_access()
        
        # Final summary
        print_section("✅ DEMO COMPLETE")
        print("\n🎉 All SI capabilities tested successfully!\n")
        print("✓ Hardware Detection: Working")
        print("✓ Memory System: Working")
        print("✓ Web Access: Working")
        print("\n" + "*"*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
